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

from __future__ import print_function
from optparse import OptionParser
import os
import random
import sys
import time

import h5py
import numpy as np
import seaborn as sns

import tensorflow as tf
import basenji

"""
basenji_norm.py

Compute prediction summary statistics for normalization.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <model_file> <test_hdf5_file>'
  parser = OptionParser(usage)
  parser.add_option('-l', dest='sample_len',
      default=1, type='int',
      help='Uniformly sample across the seq length [Default: %default]')
  parser.add_option('--mc', dest='mc_n',
      default=0, type='int',
      help='Monte carlo test iterations [Default: %default]')
  parser.add_option('-o', dest='out_dir',
      default='test_out',
      help='Output directory for test statistics [Default: %default]')
  parser.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Average the fwd and rc predictions [Default: %default]')
  parser.add_option('-s', dest='sample_seqs',
      default=1., type='float',
      help='Sample sequences [Default: %default]')
  parser.add_option('--save', dest='save',
      default=False, action='store_true',
      help='Save predictions to HDF5. [Default: %default]')
  parser.add_option('--shifts', dest='shifts',
      default='0',
      help='Ensemble prediction shifts [Default: %default]')
  parser.add_option('--train', dest='train',
      default=False, action='store_true',
      help='Process the training set [Default: %default]')
  parser.add_option('-v', dest='valid',
      default=False, action='store_true',
      help='Process the validation set [Default: %default]')
  (options, args) = parser.parse_args()

  if len(args) != 3:
    parser.error('Must provide parameters, model, and test data HDF5')
  else:
    params_file = args[0]
    model_file = args[1]
    test_hdf5_file = args[2]

  if not os.path.isdir(options.out_dir):
    os.mkdir(options.out_dir)

  options.shifts = [int(shift) for shift in options.shifts.split(',')]

  #######################################################
  # load data
  #######################################################
  data_open = h5py.File(test_hdf5_file)

  if options.train:
    test_seqs = data_open['train_in']
    test_targets = data_open['train_out']
    if 'train_na' in data_open:
      test_na = data_open['train_na']

  elif options.valid:
    test_seqs = data_open['valid_in']
    test_targets = data_open['valid_out']
    test_na = None
    if 'valid_na' in data_open:
      test_na = data_open['valid_na']

  else:
    test_seqs = data_open['test_in']
    test_targets = data_open['test_out']
    test_na = None
    if 'test_na' in data_open:
      test_na = data_open['test_na']

  if options.sample_seqs < 1:
    sample_n = int(test_seqs.shape[0]*options.sample_seqs)
    print('Sampling %d sequences' % sample_n)
    sample_indexes = sorted(np.random.choice(np.arange(test_seqs.shape[0]),
                                              size=sample_n, replace=False))
    test_seqs = test_seqs[sample_indexes]
    test_targets = test_targets[sample_indexes]
    if test_na is not None:
      test_na = test_na[sample_indexes]

  target_labels = [tl.decode('UTF-8') for tl in data_open['target_labels']]

  #######################################################
  # model parameters and placeholders

  job = basenji.dna_io.read_job_params(params_file)

  job['seq_length'] = test_seqs.shape[1]
  job['seq_depth'] = test_seqs.shape[2]
  job['num_targets'] = test_targets.shape[2]
  job['target_pool'] = int(np.array(data_open.get('pool_width', 1)))

  model = basenji.seqnn.SeqNN()
  model.build(job)

  #######################################################
  # predict

  # initialize batcher
  batcher_test = basenji.batcher.Batcher(test_seqs, test_targets, test_na,
                                         model.batch_size, model.target_pool)

  # initialize saver
  saver = tf.train.Saver()

  with tf.Session() as sess:
    # load variables into session
    saver.restore(sess, model_file)

    # test
    t0 = time.time()
    test_preds = model.predict(sess, batcher_test, rc=options.rc,
                          shifts=options.shifts, mc_n=options.mc_n,
                          down_sample=options.sample_len)
    print('SeqNN test: %ds' % (time.time() - t0))

  if options.save:
    preds_h5 = h5py.File('%s/preds.h5' % options.out_dir, 'w')
    preds_h5.create_dataset('preds', data=test_preds)
    preds_h5.close()

  #######################################################
  # normalize

  target_norms = np.ones(test_preds.shape[-1], dtype='float64')

  # compute target means
  target_means = test_preds.mean(axis=(0,1), dtype='float64')

  # determine target categories
  target_categories = set()
  for tl in target_labels:
    target_categories.add(tl.split(':')[0])

  category_out = open('%s/categories.txt' % options.out_dir, 'w')

  # normalize within category
  for target_category in target_categories:
    # determine targets in this category
    category_mask = np.zeros(len(target_norms), dtype='bool')
    for ti, tl in enumerate(target_labels):
      category_mask[ti] = (tl.split(':')[0] == target_category)

    # compute category median
    category_median = np.median(target_means[category_mask])
    print('%-10s  %4d  %.4f' % (target_category, category_mask.sum(), category_median),
      file=category_out)

    # set normalization factors
    target_norms[category_mask] = target_means[category_mask] / category_median

  category_out.close()

  # print normalization factors
  norm_out = open('%s/normalization.txt' % options.out_dir, 'w')
  print('\n'.join([str(tu) for tu in target_norms]), file=norm_out)
  norm_out.close()


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
