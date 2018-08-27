#!/usr/bin/env python
# Copyright 2017 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import collections
import glob
import h5py
import os
import pdb
import random
import shutil
import subprocess
import unittest

import numpy as np
import pandas as pd
import pysam
import tensorflow as tf

from basenji import tfrecord_util
from basenji.dna_io import hot1_dna

class TestData(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.targets_file = 'data/targets32.txt'
    cls.fasta_file = 'data/hg19_mini.fa'
    gaps_file = 'data/hg19_mini_gaps.bed'
    cls.out_dir = 'data_out'

    cls.seq_len = 131072
    cls.pool_width = 128
    stride_train = 0.8
    stride_test = 0.8

    test_pct = 0.12
    valid_pct = 0.12

    umap_bed_file = 'data/umap.bed'
    umap_t = 0.5

    # clean output directory
    if os.path.isdir(cls.out_dir):
      shutil.rmtree(cls.out_dir)

    # run command
    cmd = 'basenji_data.py'
    cmd += ' --local -p 16'
    cmd += ' -o %s' % cls.out_dir
    cmd += ' -g %s' % gaps_file
    cmd += ' -l %d' % cls.seq_len
    cmd += ' --stride_train %f' % stride_train
    cmd += ' --stride_test %f' % stride_test
    cmd += ' -v %f' % valid_pct
    cmd += ' -t %f' % test_pct
    cmd += ' -u %s' % umap_bed_file
    cmd += ' --umap_t %f' % umap_t
    cmd += ' -w %d' % cls.pool_width
    cmd += ' %s' % cls.fasta_file
    cmd += ' %s' % cls.targets_file
    print(cmd)
    subprocess.call(cmd, shell=True)


  def test_output(self):
    """Test that the output is generated."""
    train_tfrs = len(glob.glob('%s/tfrecords/train-*.tfr' % self.out_dir))
    self.assertGreater(train_tfrs, 0)

    valid_tfrs = len(glob.glob('%s/tfrecords/valid-*.tfr' % self.out_dir))
    self.assertGreater(valid_tfrs, 0)

    test_tfrs = len(glob.glob('%s/tfrecords/test-*.tfr' % self.out_dir))
    self.assertGreater(test_tfrs, 0)


  def test_seqs(self):
    """Test that the one hot coded sequences match."""
    # read sequence coordinates
    seqs_bed_file = '%s/sequences.bed' % self.out_dir
    seq_coords = read_seq_coords(seqs_bed_file)

    # read one hot coding from TF Records
    train_tfrs_str = '%s/tfrecords/train-0.tfr' % self.out_dir
    seqs_1hot, _ = self.read_tfrecords(train_tfrs_str)

    # open FASTA
    fasta_open = pysam.Fastafile(self.fasta_file)

    # check random sequences
    seq_indexes = random.sample(range(seqs_1hot.shape[0]), 32)
    for si in seq_indexes:
      sc = seq_coords[si]

      seq_fasta = fasta_open.fetch(sc.chr, sc.start, sc.end)
      seq_1hot_dna = hot1_dna(seqs_1hot[si])
      self.assertEqual(seq_fasta, seq_1hot_dna)

  def test_targets(self, atol=1e-3):
    """Test that the targets match."""
    # read sequence coordinates
    seqs_bed_file = '%s/sequences.bed' % self.out_dir
    seq_coords = read_seq_coords(seqs_bed_file)

    # read targets
    targets_df = pd.read_table(self.targets_file)

    # read one hot coding from TF Records
    train_tfrs_str = '%s/tfrecords/train-0.tfr' % self.out_dir
    _, targets = self.read_tfrecords(train_tfrs_str)

    # check random sequences
    seq_indexes = random.sample(range(targets.shape[0]), 8)
    target_indexes = random.sample(range(targets_df.shape[0]), 8)
    for si in seq_indexes:
      sc = seq_coords[si]
      for ti in target_indexes:
        # read coverage
        cov_h5 = h5py.File(targets_df.file.iloc[ti], 'r')
        seq_cov_nt = cov_h5[sc.chr][sc.start:sc.end]

        # set NaN's to zero
        seq_cov_nt = np.nan_to_num(seq_cov_nt)

        # reshape and sum
        seq_cov = seq_cov_nt.reshape((-1,self.pool_width))
        seq_cov = seq_cov.sum(axis=1, dtype='float32').astype('float16')

        # compute diff
        cov_targets_diff = np.abs(seq_cov - targets[si,:,ti])

        # compute close mask
        close_mask = (cov_targets_diff < atol)

        # guess that non-close were unmappable and set to null
        seq_cov_umap = seq_cov.copy()
        seq_cov_umap[~close_mask] = np.percentile(seq_cov, q=[25])[0]

        # compare
        try:
          np.testing.assert_allclose(targets[si,:,ti], seq_cov_umap,
                                     rtol=1, atol=atol)
        except AssertionError:
          pdb.set_trace()

  def read_tfrecords(self, tfrs_str):
    target_len = self.seq_len//self.pool_width

    seqs_1hot = []
    targets = []

    # read TF Records
    dataset = tf.data.Dataset.list_files(tfrs_str)
    dataset = dataset.flat_map(file_to_records)
    dataset = dataset.batch(1)
    dataset = dataset.map(parse_proto)

    iterator = dataset.make_one_shot_iterator()
    next_op = iterator.get_next()

    with tf.Session() as sess:
      next_datum = sess.run(next_op)
      while next_datum:
        seq_1hot = next_datum[tfrecord_util.TFR_INPUT].reshape((-1,4))
        targets1 = next_datum[tfrecord_util.TFR_OUTPUT].reshape(target_len,-1)

        seqs_1hot.append(seq_1hot)
        targets.append(targets1)

        try:
          next_datum = sess.run(next_op)
        except tf.errors.OutOfRangeError:
          next_datum = False

    seqs_1hot = np.array(seqs_1hot)
    targets = np.array(targets)

    return seqs_1hot, targets


def file_to_records(filename):
    return tf.data.TFRecordDataset(filename, compression_type='ZLIB')

def parse_proto(example_protos):
  features = {
    tfrecord_util.TFR_INPUT: tf.FixedLenFeature([], tf.string),
    tfrecord_util.TFR_OUTPUT: tf.FixedLenFeature([], tf.string)
  }
  parsed_features = tf.parse_example(example_protos, features=features)
  seq = tf.decode_raw(parsed_features[tfrecord_util.TFR_INPUT], tf.uint8)
  targets = tf.decode_raw(parsed_features[tfrecord_util.TFR_OUTPUT], tf.float16)
  return {tfrecord_util.TFR_INPUT: seq, tfrecord_util.TFR_OUTPUT: targets}

def read_seq_coords(bed_file):
  mseqs = []
  for line in open(bed_file):
    a = line.split()
    mseqs.append(ModelSeq(a[0], int(a[1]), int(a[2])))
  return mseqs

ModelSeq = collections.namedtuple('ModelSeq', ['chr', 'start', 'end'])

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  unittest.main()