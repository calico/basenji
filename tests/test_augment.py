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
from optparse import OptionParser

import os
import pdb
import subprocess
import unittest

import h5py
import numpy as np
import tensorflow as tf

from basenji import augmentation
from basenji import dna_io
from basenji import params
from basenji import tfrecord_util

class TestAugmentation(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.seq_length = 131072
    cls.target_length = 1024
    cls.tfr_data_dir = 'data_out'
    cls.targets_file = 'data/targets32.txt'

    # convert TFRecords to HDF5
    cls.data_h5 = 'data_out/data.h5'
    cmd_list = ['tfr_hdf5.py', cls.targets_file, cls.tfr_data_dir, cls.data_h5]
    # subprocess.call(cmd_list)

  def test_stochastic(self):
    # get HDF5 data
    hdf5_open = h5py.File(self.data_h5)
    hdf5_seqs = hdf5_open['valid_in']
    hdf5_targets = hdf5_open['valid_out']

    # get TFR data
    tfr_pattern = '%s/tfrecords/valid-0.tfr' % self.tfr_data_dir
    next_op = make_data_op(tfr_pattern, self.seq_length, self.target_length)

    # define augmentation
    augment_shifts = [-2, -1, 0, 1, 2]
    next_op = augmentation.augment_stochastic(next_op, True, augment_shifts)

    # initialize counters
    augment_counts = {}
    for fwdrc in [True, False]:
      for shift in augment_shifts:
        augment_counts[(fwdrc,shift)] = 0

    # choose # sequences
    max_seqs = min(64, hdf5_seqs.shape[0])
    si = 0

    # iterate over data
    si = 0
    with tf.Session() as sess:
      next_datum = sess.run(next_op)
      while next_datum:
        # parse TFRecord
        seqs_tfr = next_datum['sequence'][0]
        targets_tfr = next_datum['label'][0]

        # parse HDF5
        seqs_h5 = hdf5_seqs[si].astype('float32')
        targets_h5 = hdf5_targets[si].astype('float32')

        # expand dim
        seqs1_h5 = np.reshape(seqs_h5, (1, seqs_h5.shape[0], seqs_h5.shape[1]))

        # check augmentations for matches
        matched = False
        for fwdrc in [True, False]:
          for shift in augment_shifts:
            # modify sequence
            seqs_h5_aug = dna_io.hot1_augment(seqs1_h5, fwdrc, shift)[0]

            # modify targets
            if fwdrc:
              targets_h5_aug = targets_h5
            else:
              targets_h5_aug = targets_h5[::-1,:]

            # check match
            if np.array_equal(seqs_tfr, seqs_h5_aug) and np.allclose(targets_tfr, targets_h5_aug):
              #  print(si, fwdrc, shift)
              matched = True
              augment_counts[(fwdrc,shift)] += 1

        # assert augmentation found
        self.assertTrue(matched)

        try:
          next_datum = sess.run(next_op)
          si += 1
        except tf.errors.OutOfRangeError:
          next_datum = False

    hdf5_open.close()

    # verify all augmentations appear
    for fwdrc in [True, False]:
      for shift in augment_shifts:
        # print(fwdrc, shift, augment_counts[(fwdrc,shift)])
        self.assertGreater(augment_counts[(fwdrc,shift)], 0)


  def test_deterministic(self):
    # get HDF5 data
    hdf5_open = h5py.File(self.data_h5)
    hdf5_seqs = hdf5_open['valid_in']
    hdf5_targets = hdf5_open['valid_out']

    # get TFR data
    tfr_pattern = '%s/tfrecords/valid-0.tfr' % self.tfr_data_dir
    next_op = make_data_op(tfr_pattern, self.seq_length, self.target_length)

    # define augmentation
    augment_shifts = [-2, -1, 0, 1, 2]
    next_op_list = augmentation.augment_deterministic_set(next_op, True, augment_shifts)

    # initialize counters
    augment_counts = {}
    for fwdrc in [True, False]:
      for shift in augment_shifts:
        augment_counts[(fwdrc,shift)] = 0

    # choose # sequences
    max_seqs = min(32, hdf5_seqs.shape[0])
    si = 0

    # iterate over data
    with tf.Session() as sess:
      next_datums = sess.run(next_op_list)
      while next_datums and si < max_seqs:
        for next_datum in next_datums:
          # parse TFRecord
          seqs_tfr = next_datum['sequence'][0]
          targets_tfr = next_datum['label'][0]

          # parse HDF5
          seqs_h5 = hdf5_seqs[si].astype('float32')
          targets_h5 = hdf5_targets[si].astype('float32')

          # expand dim
          seqs1_h5 = np.reshape(seqs_h5, (1, seqs_h5.shape[0], seqs_h5.shape[1]))

          # check augmentation
          matched = False
          for fwdrc in [True, False]:
            for shift in augment_shifts:
              # modify sequence
              seqs_h5_aug = dna_io.hot1_augment(seqs1_h5, fwdrc, shift)[0]

              # modify targets
              if fwdrc:
                targets_h5_aug = targets_h5
              else:
                targets_h5_aug = targets_h5[::-1,:]

              # check match
              if np.array_equal(seqs_tfr, seqs_h5_aug) and np.allclose(targets_tfr, targets_h5_aug):
                # print(si, fwdrc, shift)
                matched = True
                augment_counts[(fwdrc,shift)] += 1

          # assert augmentation found
          self.assertTrue(matched)

        try:
          next_datums = sess.run(next_op_list)
          si += 1
        except tf.errors.OutOfRangeError:
          next_datums = False

    hdf5_open.close()

    # verify all augmentations appear
    for fwdrc in [True, False]:
      for shift in augment_shifts:
        #print(fwdrc, shift, augment_counts[(fwdrc,shift)])
        self.assertEqual(max_seqs, augment_counts[(fwdrc,shift)])


def make_data_op(tfr_pattern, seq_length, target_length):
  dataset = tf.data.Dataset.list_files(tfr_pattern)

  def file_to_records(filename):
    return tf.data.TFRecordDataset(filename, compression_type='ZLIB')
  dataset = dataset.flat_map(file_to_records)

  def parse_proto(example_protos):
    features = {
      tfrecord_util.TFR_INPUT: tf.FixedLenFeature([], tf.string),
      tfrecord_util.TFR_OUTPUT: tf.FixedLenFeature([], tf.string)
    }
    parsed_features = tf.parse_example(example_protos, features=features)

    seq = tf.decode_raw(parsed_features[tfrecord_util.TFR_INPUT], tf.uint8)
    seq = tf.reshape(seq, [1, seq_length, -1])
    seq = tf.cast(seq, tf.float32)

    targets = tf.decode_raw(parsed_features[tfrecord_util.TFR_OUTPUT], tf.float16)
    targets = tf.reshape(targets, (1, target_length, -1))
    targets = tf.cast(targets, tf.float32)

    na = tf.zeros(targets.shape[:-1], dtype=tf.bool)

    return {'sequence': seq, 'label': targets, 'na':na}

  dataset = dataset.batch(1)
  dataset = dataset.map(parse_proto)

  iterator = dataset.make_one_shot_iterator()
  try:
    next_op = iterator.get_next()
  except tf.errors.OutOfRangeError:
    print('TFRecord pattern %s is empty' % self.tfr_pattern, file=sys.stderr)
    exit(1)

  return next_op

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  unittest.main()