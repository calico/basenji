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

import pdb
import sys
import time

import numpy as np
import tensorflow as tf
try:
  import tensorflow_probability as tfp
except ImportError:
  pass

from basenji import augmentation
from basenji import blocks
from basenji import layers
from basenji import params

class SeqNN():

  def __init__(self, job, log_dir):
    self.hp = params.make_hparams(job)
    self.log_dir = log_dir
    self.build_model()


  def build_model(self, save_reprs=False):
    self.sequence = tf.keras.Input(shape=(self.hp.seq_length, 4), name='sequence')
    self.genome = tf.keras.Input(shape=(None,), name='genome')
    current = self.sequence

    ###################################################
    # convolutions
    ###################################################
    layer_reprs = [current]
    for li in range(self.hp.cnn_layers):
      with tf.variable_scope('layer%d' % li):
        skip_inputs = layer_reprs[-self.hp.cnn_params[li].skip_layers] if self.hp.cnn_params[li].skip_layers > 0 else None

        # build convolution block
        current = blocks.conv_pool(
          current,
          filters=self.hp.cnn_params[li].filters,
          kernel_size=self.hp.cnn_params[li].filter_size,
          activation=self.hp.nonlinearity,
          strides=self.hp.cnn_params[li].stride,
          l2_weight=self.hp.cnn_l2_scale,
          momentum=self.hp.batch_norm_momentum,
          dropout=self.hp.cnn_params[li].dropout,
          pool_size=self.hp.cnn_params[li].pool,
          skip_inputs=skip_inputs,
          concat=self.hp.cnn_params[li].concat)

        # save representation
        layer_reprs.append(current)

    if save_reprs:
      self.layer_reprs = layer_reprs

    # final activation
    if self.hp.nonlinearity == 'relu':
      current = tf.keras.layers.ReLU()(current)
    elif self.hp.nonlinearity == 'gelu':
      current = layers.GELU()(current)
    else:
      print('Unrecognized activation "%s"' % self.hp.nonlinearity, file=sys.stderr)
      exit(1)

    ###################################################
    # slice out side buffer
    ###################################################

    # update batch buffer to reflect pooling
    seq_length = current.shape[1].value
    pool_preds = self.hp.seq_length // seq_length
    assert self.hp.seq_end_ignore % pool_preds == 0, (
        'seq_end_ignore %d not divisible'
        ' by the CNN pooling %d') % (self.hp.seq_end_ignore, pool_preds)
    seq_end_ignore_pool = self.hp.seq_end_ignore // pool_preds

    # Cropping1D?

    # slice out buffer
    seq_length = current.shape[1]
    current = layers.SliceCenter(
      left=seq_end_ignore_pool,
      right=seq_length-seq_end_ignore_pool)(current)
    seq_length = current.shape[1]

    ###################################################
    # final layer
    ###################################################
    with tf.variable_scope('final'):
      current = tf.keras.layers.Dense(
        units=self.hp.sum_targets,
        activation=None,
        use_bias=True,
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l1(self.hp.final_l1_scale)
        )(current)

    # transform for reverse complement
    # current = layers.SwitchReverse()([current, input_reverse])

    ###################################################
    # link
    ###################################################
    # float 32 exponential clip max
    exp_max = 50

    # choose link
    if self.hp.link == 'softplus':
      current = layers.Softplus(exp_max)(current)

    else:
      print('Unknown link function %s' % self.hp.link, file=sys.stderr)
      exit(1)

    # clip
    if self.hp.target_clip is not None:
      current = layers.Clip(0, self.hp.target_clip)(current)

    self.preds = current

    ###################################################
    # compile model
    ###################################################
    self.model = tf.keras.Model(inputs=[self.sequence,self.genome], outputs=self.preds)
    print(self.model.summary())


  def train(self, trainer):
    self.model.compile(loss='poisson',
                       optimizer=trainer.optimizer,
                       metrics=[tf.keras.metrics.mean_absolute_error])

    callbacks = [
      tf.keras.callbacks.EarlyStopping(patience=trainer.patience, monitor='val_loss'),
      tf.keras.callbacks.TensorBoard(log_dir=self.log_dir)
    ]

    self.model.fit(
      trainer.train_data.dataset,
      epochs=trainer.train_epochs,
      steps_per_epoch=trainer.train_epoch_batches,
      callbacks=callbacks,
      validation_data=trainer.eval_data.dataset,
      validation_steps=trainer.eval_epoch_batches)


  # def fit(self, train_data, eval_data, train_epochs, train_epoch_batches=None):
  #   callbacks = [
  #     tf.keras.callbacks.EarlyStopping(patience=20, monitor='val_loss'),
  #     tf.keras.callbacks.TensorBoard(log_dir=self.log_dir)
  #   ]

  #   if train_epoch_batches is None:
  #     train_epoch_batches = train_data.batches_per_epoch()

  #   self.model.fit(
  #     train_data.dataset,
  #     epochs=train_epochs,
  #     steps_per_epoch=train_epoch_batches,
  #     callbacks=callbacks,
  #     validation_data=eval_data.dataset,
  #     validation_steps=eval_data.batches_per_epoch())
