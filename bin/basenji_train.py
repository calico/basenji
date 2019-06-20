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

import json
import os
import pdb
from queue import Queue
import sys
from threading import Thread
import time

from absl import app, flags
import numpy as np




################################################################################

# parameters and data
flags.DEFINE_string('params', '', 'Parameter JSON')
flags.DEFINE_string('train_data', '', 'train tfrecord file')
flags.DEFINE_string('eval_data', '', 'test tfrecord file')

# ensembling/augmentation
flags.DEFINE_boolean('augment_rc', False, 'Augment training with reverse complement.')
flags.DEFINE_boolean('ensemble_rc', False, 'Ensemble prediction with reverse complement.')
flags.DEFINE_string('augment_shifts', '0', 'Augment training with shifted sequences.')
flags.DEFINE_string('ensemble_shifts', '0', 'Ensemble prediction with shifted sequences.')
flags.DEFINE_integer('ensemble_mc', 0, 'Ensemble monte carlo samples.')

# training modes
flags.DEFINE_string('restart', None, 'Restart training the model')

# eval options
flags.DEFINE_boolean('metrics_thread', False, 'Evaluate validation metrics in a separate thread.')
flags.DEFINE_boolean('r', False, 'Compute validation set PearsonrR.')
flags.DEFINE_boolean('r2', False, 'Compute validation set R2.')
flags.DEFINE_float('metrics_sample', 1.0, 'Sample sequence positions for computing metrics.')

FLAGS = flags.FLAGS


################################################################################

def main(_):
  # I could write some additional code around this to check for common
  # problems, such as with num_targets.
  with open(FLAGS.params) as params_open:
    params = json.load(params_open)
  params_model = params['model']
  params_train = params['train']

  if params_train.get('use_gpu',1) == False:
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    print("  ")
    print(" training on CPU ")
    print("  ")

  import tensorflow as tf
  if tf.__version__[0] == '1':
    tf.compat.v1.enable_eager_execution()
  from basenji import dataset
  from basenji import seqnn
  from basenji import trainer


  # load data
  train_data = dataset.SeqDataset(FLAGS.train_data,
    params_train['batch_size'],
    params_model['seq_length'],
    params_model['target_length'],
    tf.estimator.ModeKeys.TRAIN)
  eval_data = dataset.SeqDataset(FLAGS.eval_data,
    params_train['batch_size'],
    params_model['seq_length'],
    params_model['target_length'],
    tf.estimator.ModeKeys.EVAL)


  if params_train.get('num_gpu',1) == 1:
    ########################################
    # one GPU

    # initialize model
    seqnn_model = seqnn.SeqNN(params_model)

    # initialize trainer
    seqnn_trainer = trainer.Trainer(params_train, train_data, eval_data)

    # compile model
    seqnn_trainer.compile(seqnn_model.model)

    # train model
    seqnn_trainer.fit(seqnn_model.model)

  else:
    ########################################
    # two GPU

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():

      # initialize model
      seqnn_model = seqnn.SeqNN(params_model)

      # initialize trainer
      seqnn_trainer = trainer.Trainer(params_train, train_data, eval_data)

      # compile model
      seqnn_trainer.compile(seqnn_model.model, None)

    # train model
    seqnn_trainer.fit(seqnn_model.model)

if __name__ == '__main__':
  app.run(main)
