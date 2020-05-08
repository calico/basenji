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
if tf.__version__[0] == '1':
  tf.compat.v1.enable_eager_execution()

from basenji import dataset
from basenji import seqnn
from basenji import trainer

"""
basenji_train.py

Train Basenji model using given parameters and data.
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
  parser.add_option('--restore', dest='restore',
      help='Restore model and continue training [Default: %default]')
  parser.add_option('--trunk', dest='trunk',
      default=False, action='store_true',
      help='Restore only model trunk [Default: %default]')
  parser.add_option('--tfr_train', dest='tfr_train_pattern',
      default='train-*.tfr',
      help='Training TFRecord pattern string appended to data_dir [Default: %default]')
  parser.add_option('--tfr_eval', dest='tfr_eval_pattern',
      default='valid-*.tfr',
      help='Evaluation TFRecord pattern string appended to data_dir [Default: %default]')
  (options, args) = parser.parse_args()

  if len(args) != 2:
    parser.error('Must provide parameters and data directory.')
  else:
    params_file = args[0]
    data_dir = args[1]

  if not os.path.isdir(options.out_dir):
    os.mkdir(options.out_dir)
  if params_file != '%s/params.json' % options.out_dir:
    shutil.copy(params_file, '%s/params.json' % options.out_dir)

  # read model parameters
  with open(params_file) as params_open:
    params = json.load(params_open)
  params_model = params['model']
  params_train = params['train']

  # read data parameters
  data_stats_file = '%s/statistics.json' % data_dir
  with open(data_stats_file) as data_stats_open:
    data_stats = json.load(data_stats_open)

  # load data
  tfr_train_full = '%s/tfrecords/%s' % (data_dir, options.tfr_train_pattern)
  train_data = dataset.SeqDataset(tfr_train_full,
    params_train['batch_size'],
    data_stats['seq_length'],
    data_stats['target_length'],
    tf.estimator.ModeKeys.TRAIN)
  tfr_eval_full = '%s/tfrecords/%s' % (data_dir, options.tfr_eval_pattern)
  eval_data = dataset.SeqDataset(tfr_eval_full,
    params_train['batch_size'],
    data_stats['seq_length'],
    data_stats['target_length'],
    tf.estimator.ModeKeys.EVAL)

  if params_train.get('num_gpu', 1) == 1:
    ########################################
    # one GPU

    # initialize model
    seqnn_model = seqnn.SeqNN(params_model)

    # restore
    if options.restore:
      seqnn_model.restore(options.restore, options.trunk)

    # initialize trainer
    seqnn_trainer = trainer.Trainer(params_train, train_data, 
                                    eval_data, options.out_dir)

    # compile model
    seqnn_trainer.compile(seqnn_model)

    # train model
    seqnn_trainer.fit(seqnn_model)

  else:
    ########################################
    # two GPU

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():

      # initialize model
      seqnn_model = seqnn.SeqNN(params_model)

      # restore
      if options.restore:
        seqnn_model.restore(options.restore, options.trunk)

      # initialize trainer
      seqnn_trainer = trainer.Trainer(params_train, train_data,
                                      eval_data, options.out_dir)

      # compile model
      seqnn_trainer.compile(seqnn_model)

    # train model
    seqnn_trainer.fit(seqnn_model)

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
