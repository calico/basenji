#!/usr/bin/env python
# Copyright 2020 Calico LLC
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
import json
import os
import pdb
import sys
import time

import h5py
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import tensorflow as tf
if tf.__version__[0] == '1':
  tf.compat.v1.enable_eager_execution()

from quantile_normalization import quantile_normalize

from basenji import dataset
from basenji import seqnn


"""
basenji_test_specificity.py

Test the accuracy of a trained model on targets/predictions normalized across targets.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <model_file> <data_dir>'
  parser = OptionParser(usage)
  parser.add_option('--head', dest='head_i',
      default=0, type='int',
      help='Parameters head to test [Default: %default]')
  parser.add_option('-o', dest='out_dir',
      default='test_out',
      help='Output directory for test statistics [Default: %default]')
  parser.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Average the fwd and rc predictions [Default: %default]')
  parser.add_option('--save', dest='save',
      default=False, action='store_true',
      help='Save targets and predictions numpy arrays [Default: %default]')
  parser.add_option('--shifts', dest='shifts',
      default='0',
      help='Ensemble prediction shifts [Default: %default]')
  parser.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')
  parser.add_option('--split', dest='split_label',
      default='test',
      help='Dataset split label for eg TFR pattern [Default: %default]')
  parser.add_option('--tfr', dest='tfr_pattern',
      default=None,
      help='TFR pattern string appended to data_dir/tfrecords for subsetting [Default: %default]')
  parser.add_option('-v', dest='high_var_pct',
      default=1.0, type='float',
      help='Highly variable site proportion to take [Default: %default]')
  (options, args) = parser.parse_args()

  if len(args) != 3:
    parser.error('Must provide parameters, model, and test data HDF5')
  else:
    params_file = args[0]
    model_file = args[1]
    data_dir = args[2]

  if not os.path.isdir(options.out_dir):
    os.mkdir(options.out_dir)

  # parse shifts to integers
  options.shifts = [int(shift) for shift in options.shifts.split(',')]

  #######################################################
  # targets

  # read table
  if options.targets_file is None:
    options.targets_file = '%s/targets.txt' % data_dir
  targets_df = pd.read_csv(options.targets_file, index_col=0, sep='\t')
  num_targets = targets_df.shape[0]

  # classify
  target_classes = []
  for ti in range(num_targets):
    description = targets_df.iloc[ti].description
    if description.find(':') == -1:
      tc = '*'
    else:
      desc_split = description.split(':')
      if desc_split[0] == 'CHIP':
        tc = '/'.join(desc_split[:2])
      else:
        tc = desc_split[0]
    target_classes.append(tc)
  targets_df['class'] = target_classes
  target_classes = sorted(set(target_classes))
  print(target_classes)

  #######################################################
  # model

  # read parameters
  with open(params_file) as params_open:
    params = json.load(params_open)
  params_model = params['model']
  params_train = params['train']

  # construct eval data
  eval_data = dataset.SeqDataset(data_dir,
    split_label=options.split_label,
    batch_size=params_train['batch_size'],
    mode='eval',
    tfr_pattern=options.tfr_pattern)

  # initialize model
  seqnn_model = seqnn.SeqNN(params_model)
  seqnn_model.restore(model_file, options.head_i)
  seqnn_model.build_ensemble(options.rc, options.shifts)
  seqnn_model.downcast()

  #######################################################
  # targets/predictions

  # option to read from disk?

  # predict
  eval_preds = seqnn_model.predict(eval_data, verbose=1)
  print('')
  
  # targets
  eval_targets = eval_data.numpy(return_inputs=False, return_outputs=True)

  # flatten
  eval_preds = np.reshape(eval_preds, (-1,num_targets))
  eval_targets = np.reshape(eval_targets, (-1,num_targets))

  #######################################################
  # process classes

  targets_spec = np.zeros(num_targets)

  for tc in target_classes:
    class_mask = np.array(targets_df['class'] == tc)
    num_targets_class = class_mask.sum()

    if num_targets_class == 1:
      targets_spec[class_mask] = np.nan
    else:
      # slice class
      eval_preds_class = eval_preds[:,class_mask].astype('float32')
      eval_targets_class = eval_targets[:,class_mask].astype('float32')

      # highly variable filter
      if options.high_var_pct < 1:
        eval_targets_var = eval_targets_class.var(axis=1)
        high_var_t = np.percentile(eval_targets_var, 100*(1-options.high_var_pct))
        high_var_mask = (eval_targets_var >= high_var_t)

        eval_preds_class = eval_preds_class[high_var_mask]
        eval_targets_class = eval_targets_class[high_var_mask]

      # quantile normalize
      eval_preds_norm = quantile_normalize(eval_preds_class)
      eval_targets_norm = quantile_normalize(eval_targets_class)

      # mean normalize
      eval_preds_norm = eval_preds_norm - eval_preds_norm.mean(axis=-1, keepdims=True)
      eval_targets_norm = eval_targets_norm - eval_targets_norm.mean(axis=-1, keepdims=True)

      # compute correlations
      pearsonr_class = np.zeros(num_targets_class)
      for ti in range(num_targets_class):
        pearsonr_class[ti] = pearsonr(eval_preds_norm[:,ti], eval_targets_norm[:,ti])[0]

      # save
      targets_spec[class_mask] = pearsonr_class

      # print
      print('%-15s  %4d  %.4f' % (tc, num_targets_class, pearsonr_class[ti]))

  # write target-level statistics
  targets_acc_df = pd.DataFrame({
      'index': targets_df.index,
      'pearsonr': targets_spec,
      'identifier': targets_df.identifier,
      'description': targets_df.description
      })
  targets_acc_df.to_csv('%s/acc.txt'%options.out_dir, sep='\t',
                        index=False, float_format='%.5f')


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
