#!/usr/bin/env python
# Copyright 2019 Calico LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
from optparse import OptionParser
import os
import sys

import h5py
import numpy as np
import pdb
import pysam

from basenji_data import ModelSeq
from basenji.dna_io import dna_1hot, dna_1hot_index

import tensorflow as tf

"""
basenji_data_write.py

Write TF Records for batches of model sequences.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <fasta_file> <seqs_bed_file> <seqs_cov_dir> <tfr_file>'
  parser = OptionParser(usage)
  parser.add_option('-d', dest='decimals',
      default=None, type='int',
      help='Round values to given decimals [Default: %default]')
  parser.add_option('-s', dest='start_i',
      default=0, type='int',
      help='Sequence start index [Default: %default]')
  parser.add_option('-e', dest='end_i',
      default=None, type='int',
      help='Sequence end index [Default: %default]')
  parser.add_option('--te', dest='target_extend',
      default=None, type='int', help='Extend targets vector [Default: %default]')
  parser.add_option('-u', dest='umap_npy',
      help='Unmappable array numpy file')
  parser.add_option('--umap_clip', dest='umap_clip',
      default=1, type='float',
      help='Clip values at unmappable positions to distribution quantiles, eg 0.25. [Default: %default]')
  parser.add_option('--umap_tfr', dest='umap_tfr',
      default=False, action='store_true',
      help='Save umap array into TFRecords [Default: %default]')
  parser.add_option('-x', dest='extend_bp',
      default=0, type='int',
      help='Extend sequences on each side [Default: %default]')
  (options, args) = parser.parse_args()

  if len(args) != 4:
    parser.error('Must provide input arguments.')
  else:
    fasta_file = args[0]
    seqs_bed_file = args[1]
    seqs_cov_dir = args[2]
    tfr_file = args[3]

  ################################################################
  # read model sequences

  model_seqs = []
  for line in open(seqs_bed_file):
    a = line.split()
    model_seqs.append(ModelSeq(a[0],int(a[1]),int(a[2]),None))

  if options.end_i is None:
    options.end_i = len(model_seqs)

  num_seqs = options.end_i - options.start_i

  ################################################################
  # determine sequence coverage files

  seqs_cov_files = []
  ti = 0
  seqs_cov_file = '%s/%d.h5' % (seqs_cov_dir, ti)
  while os.path.isfile(seqs_cov_file):
    seqs_cov_files.append(seqs_cov_file)
    ti += 1
    seqs_cov_file = '%s/%d.h5' % (seqs_cov_dir, ti)

  if len(seqs_cov_files) == 0:
    print('Sequence coverage files not found, e.g. %s' % seqs_cov_file, file=sys.stderr)
    exit(1)

  seq_pool_len = h5py.File(seqs_cov_files[0], 'r')['targets'].shape[1]
  num_targets = len(seqs_cov_files)

  ################################################################
  # read targets

  # initialize targets
  targets = np.zeros((num_seqs, seq_pool_len, num_targets), dtype='float16')

  # read each target
  for ti in range(num_targets):
    seqs_cov_open = h5py.File(seqs_cov_files[ti], 'r')
    targets[:,:,ti] = seqs_cov_open['targets'][options.start_i:options.end_i,:]
    seqs_cov_open.close()

  ################################################################
  # modify unmappable

  if options.umap_npy is not None and options.umap_clip < 1:
    unmap_mask = np.load(options.umap_npy)

    for si in range(num_seqs):
      msi = options.start_i + si

      # determine unmappable null value
      seq_target_null = np.percentile(targets[si], q=[100*options.umap_clip], axis=0)[0]

      # set unmappable positions to null
      targets[si,unmap_mask[msi,:],:] = np.minimum(targets[si,unmap_mask[msi,:],:], seq_target_null)

  elif options.umap_npy is not None and options.umap_tfr:
    unmap_mask = np.load(options.umap_npy)

  ################################################################
  # write TFRecords

  # open FASTA
  fasta_open = pysam.Fastafile(fasta_file)

  # define options
  tf_opts = tf.io.TFRecordOptions(compression_type='ZLIB')

  with tf.io.TFRecordWriter(tfr_file, tf_opts) as writer:
    for si in range(num_seqs):
      msi = options.start_i + si
      mseq = model_seqs[msi]
      mseq_start = mseq.start - options.extend_bp
      mseq_end = mseq.end + options.extend_bp

      # read FASTA
      # seq_dna = fasta_open.fetch(mseq.chr, mseq.start, mseq.end)
      seq_dna = fetch_dna(fasta_open, mseq.chr, mseq_start, mseq_end)

      # one hot code (N's as zero)
      # seq_1hot = dna_1hot(seq_dna, n_uniform=False, n_sample=False)
      seq_1hot = dna_1hot_index(seq_dna) # more efficient, but fighting inertia

      # truncate decimals (which aids compression)
      if options.decimals is not None:
        targets_si = targets[si].astype('float32')
        targets_si = np.around(targets_si, decimals=options.decimals)
        targets_si = targets_si.astype('float16')
        # targets_si = rround(targets[si], decimals=options.decimals)
      else:
        targets_si = targets[si]

      assert(np.isinf(targets_si).sum() == 0)

      # hash to bytes
      features_dict = {
        'sequence': feature_bytes(seq_1hot),
        'target': feature_bytes(targets_si)
        }

      # add unmappability
      if options.umap_tfr:
        features_dict['umap'] = feature_bytes(unmap_mask[msi,:])

      # write example
      example = tf.train.Example(features=tf.train.Features(feature=features_dict))
      writer.write(example.SerializeToString())

    fasta_open.close()


def tround(a, decimals):
  """ Truncate to the specified number of decimals. """
  return np.true_divide(np.floor(a * 10**decimals), 10**decimals)

def rround(a, decimals):
  """ Round to the specified number of decimals, randomly sampling
      the last digit according to a bernoulli RV. """
  a_dtype = a.dtype
  a = a.astype('float32')
  dec_probs = (a - tround(a, decimals)) * 10**decimals
  dec_bin = np.random.binomial(n=1, p=dec_probs)
  a_dec = tround(a, decimals) + dec_bin / 10**decimals
  return np.around(a_dec.astype(a_dtype), decimals)

def fetch_dna(fasta_open, chrm, start, end):
  """Fetch DNA when start/end may reach beyond chromosomes."""

  # initialize sequence
  seq_len = end - start
  seq_dna = ''

  # add N's for left over reach
  if start < 0:
    seq_dna = 'N'*(-start)
    start = 0

  # get dna
  seq_dna += fasta_open.fetch(chrm, start, end)

  # add N's for right over reach
  if len(seq_dna) < seq_len:
    seq_dna += 'N'*(seq_len-len(seq_dna))

  return seq_dna


def feature_bytes(values):
  """Convert numpy arrays to bytes features."""
  values = values.flatten().tobytes()
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def feature_floats(values):
  """Convert numpy arrays to floats features.
     Requires more space than bytes for float16"""
  values = values.flatten().tolist()
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
