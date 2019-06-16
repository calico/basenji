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

  def __init__(self, params):
    for key, value in params.items():
      self.__setattr__(key, value)
    self.build_model()

  def build_block(self, current, block_params):
    """Construct a SeqNN block.

    Args:

    Returns:
      current
    """
    block_args = {}

    # set global defaults
    global_vars = ['activation', 'batch_norm', 'bn_momentum',
      'l2_scale', 'l1_scale']
    for gv in global_vars:
      gv_value = getattr(self, gv, False)
      if gv_value:
        block_args[gv] = gv_value

    # extract name
    block_name = block_params['name']
    del block_params['name']

    # set remaining params
    block_args.update(block_params)

    # switch for block
    if block_name[0].islower():
      block_func = blocks.name_func[block_name]
      current = block_func(current, **block_args)
    else:
      block_func = blocks.keras_func[block_name]
      current = block_func(**block_args)(current)

    return current

  def build_model(self, save_reprs=False):
    ###################################################
    # inputs
    ###################################################
    self.sequence = tf.keras.Input(shape=(self.seq_length, 4), name='sequence')
    self.genome = tf.keras.Input(shape=(1,), name='genome')
    current = self.sequence

    ###################################################
    # build convolution blocks
    ###################################################
    for bi, block_params in enumerate(self.blocks):
      current = self.build_block(current, block_params)

    # final activation
    current = layers.activate(current, self.activation)

    # TEMP to include in the graph for model saving
    # genome_repeat = tf.keras.layers.RepeatVector(1024)(tf.cast(self.genome, tf.float32))
    # current = tf.keras.layers.Add()([current, genome_repeat])

    ###################################################
    # slice center (replace w/ Cropping1D?)
    ###################################################
    current_length = current.shape[1]
    target_diff = self.target_length - current_length
    target_diff2 = target_diff // 2
    if target_diff2 < 0:
      print('Model over-pools to %d for target_length %d.' % \
        (current_length, self.target_length), file=sys.stderr)
      exit(1)
    elif target_diff2 > 0:
      current = layers.SliceCenter(
        left=target_diff2,
        right=current_length-target_diff2)(current)

    ###################################################
    # final layer
    ###################################################
    self.sum_targets = np.sum(self.num_targets)

    current = tf.keras.layers.Dense(
      units=self.sum_targets,
      activation=None,
      use_bias=True,
      kernel_initializer='he_normal',
      )(current)
      # kernel_regularizer=tf.keras.regularizers.l1(self.pred_l1_scale)

    # transform for reverse complement
    # current = layers.SwitchReverse()([current, input_reverse])

    ###################################################
    # link
    ###################################################
    # float 32 exponential clip max
    exp_max = 50

    # choose link
    current = layers.Softplus(exp_max)(current)

    self.preds = current

    ###################################################
    # compile model
    ###################################################
    self.model = tf.keras.Model(inputs=[self.sequence,self.genome], outputs=self.preds)
    print(self.model.summary())
