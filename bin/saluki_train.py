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

import json
import os
import shutil
import sys
import time

import numpy as np
import tensorflow as tf

from basenji import dataset
from basenji import rnann
from basenji import trainer

"""
saluki_train.py

Train Saluki model using given parameters and data on RNA sequence.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <data_dir>'
  parser = OptionParser(usage)
  parser.add_option('-o', dest='out_dir',
      default='train_out',
      help='Output directory for test statistics [Default: %default]')
  (options, args) = parser.parse_args()

  if len(args) != 2:
    parser.error('Must provide parameters and data directory.')
  else:
    params_file = args[0]
    data_dir = args[1]

  # read model parameters
  with open(params_file) as params_open:
    params = json.load(params_open)
  params_model = params['model']
  params_train = params['train']

  # read data stats
  data_stats_file = '%s/statistics.json' % data_dir
  with open(data_stats_file) as data_stats_open:
    data_stats = json.load(data_stats_open)
  num_folds = data_stats['num_folds']

  # set seq length
  # params_model['seq_length'] = data_stats['length_%s' % params_model['rna_mode']]
  # params_model['num_features'] = data_stats.get('num_features',0)
  params_model['seq_length'] = data_stats['length_t']

  os.makedirs(options.out_dir, exist_ok=True)
  if params_file != '%s/params.json' % options.out_dir:
    shutil.copy(params_file, '%s/params.json' % options.out_dir)
  
  # for each fold
  for fi in range(num_folds):
    fold_out_dir = '%s/f%d' % (options.out_dir,fi)
    os.makedirs(fold_out_dir, exist_ok=True)

    # divide train/test
    split_labels_eval = ['fold%d'%fi]
    split_labels_train = []
    for fj in range(num_folds):
      if fi != fj:
        split_labels_train.append('fold%d'%fj)

    # initialize train data
    train_data = dataset.RnaDataset(data_dir,
      split_labels=split_labels_train,
      batch_size=params_train['batch_size'],
      shuffle_buffer=params_train.get('shuffle_buffer',1024),
      mode='train')
    # rna_mode=params_model['rna_mode'],

    # initialize eval data
    eval_data = dataset.RnaDataset(data_dir,
      split_labels=split_labels_eval,
      batch_size=params_train['batch_size'],
      mode='eval')
    # rna_mode=params_model['rna_mode'],

    # initialize model
    seqnn_model = rnann.RnaNN(params_model)

    # initialize trainer
    seqnn_trainer = trainer.RnaTrainer(params_train, train_data, 
                                       eval_data, fold_out_dir)

    # compile model
    seqnn_trainer.compile(seqnn_model)

    # train model
    seqnn_trainer.fit(seqnn_model)

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
