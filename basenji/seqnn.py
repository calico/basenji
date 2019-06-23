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

from basenji import blocks
from basenji import layers
from basenji import metrics

class SeqNN():

  def __init__(self, params):
    self.set_defaults()
    for key, value in params.items():
      self.__setattr__(key, value)
    self.build_model()
    self.ensemble = None

  def set_defaults(self):
    # only necessary for my bespoke parameters
    # others are best defaulted closer to the source
    self.augment_rc = False
    self.augment_shift = 0

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
    sequence = tf.keras.Input(shape=(self.seq_length, 4), name='sequence')
    # self.genome = tf.keras.Input(shape=(1,), name='genome')
    current = sequence

    # augmentation
    if self.augment_rc:
      current, reverse_bool = layers.StochasticReverseComplement()(current)
    current = layers.StochasticShift(self.augment_shift)(current)

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
      bias_initializer='zeros'
      )(current)
      # kernel_regularizer=tf.keras.regularizers.l1(self.pred_l1_scale)

    if self.augment_rc:
      # transform back from reverse complement
      current = layers.SwitchReverse()([current, reverse_bool])

    ###################################################
    # link
    ###################################################

    # choose link
    link = getattr(self,'link', None)
    if (link is None) or (link is 'softplus'):
      # float 32 exponential clip max
      exp_max = 50
      current = layers.Softplus(exp_max)(current)
    elif link is 'identity' or 'linear':
      current = current  #tf.identity(current, name='preds') led to a strange JSON error

    self.preds = current

    ###################################################
    # compile model
    ###################################################
    # self.model = tf.keras.Model(inputs=[sequence,self.genome], outputs=self.preds)
    self.model = tf.keras.Model(inputs=sequence, outputs=self.preds)
    print(self.model.summary())


  def build_ensemble(self, ensemble_rc=False, ensemble_shifts=[0]):
    """ Build ensemble of models computing on augmented input sequences. """
    if ensemble_rc or len(ensemble_shifts) > 1:
      # sequence input
      sequence = tf.keras.Input(shape=(self.seq_length, 4), name='sequence')
      sequences = [sequence]

      if len(ensemble_shifts) > 1:
        # generate shifted sequences
        sequences = layers.EnsembleShift(ensemble_shifts)(sequences)

      if ensemble_rc:
        # generate reverse complements and indicators
        sequences_rev = layers.EnsembleReverseComplement()(sequences)
      else:
        sequences_rev = [(seq,tf.constant(False)) for seq in sequences]

      # predict each sequence
      preds = [layers.SwitchReverse()([self.model(seq), rp]) for (seq,rp) in sequences_rev]

      # create layer
      preds_avg = tf.keras.layers.Average()(preds)

      # create meta model
      self.ensemble = tf.keras.Model(inputs=sequence, outputs=preds_avg)


  def evaluate(self, seq_data):
    """ Evaluate model on SeqDataset. """
    # choose ensemble if built
    if self.ensemble is None:
      model = self.model
    else:
      model = self.ensemble

    # compile with dense metrics
    num_targets = self.model.output_shape[-1]
    loss = getattr(self,'loss','poisson')
    print('eval loss:', loss)
    model.compile(loss= loss,
                  optimizer=tf.keras.optimizers.SGD(),
                  metrics=[metrics.PearsonR(num_targets, summarize=False),
                           metrics.R2(num_targets, summarize=False)])

    # evaluate
    return model.evaluate(seq_data.dataset)


  def predict(self, seq_data):
    """ Predict targets for SeqDataset. """
    if self.ensemble is None:
      return self.model.predict(seq_data.dataset)
    else:
      return self.ensemble(seq_data.dataset)


  def restore(self, model_file):
    """ Restore weights from saved model. """
    self.model.load_weights(model_file)
