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
    print('block_args',block_args)

    # switch for block
    if block_name[0].islower():
      block_func = blocks.name_func[block_name]
      current = block_func(current, **block_args)

    else:
      block_args= {} # keras layers don't necessarily understand global vars
      block_args.update(block_params)
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
    for bi, block_params in enumerate(self.trunk):
      current = self.build_block(current, block_params)

    # final activation
    current = layers.activate(current, self.activation)

    # TEMP to include in the graph for model saving
    # genome_repeat = tf.keras.layers.RepeatVector(1024)(tf.cast(self.genome, tf.float32))
    # current = tf.keras.layers.Add()([current, genome_repeat])

    ###################################################
    # slice center (replace w/ Cropping1D?)
    ###################################################

    slicegf = False
    if slicegf:
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


    if self.augment_rc:
      print('self.augment_rc')
      # transform back from reverse complement
      current = layers.SwitchReverse()([current, reverse_bool]) ### needs to change for hic


    self.trunk_output = current

    ###################################################
    # heads
    ###################################################

    if not isinstance(self.head, list):
        self.head = [self.head]

    self.head_output = []

    for hi, head in enumerate(self.head):

        if not isinstance(head, list):
            head = [head]
            print('head is not a list')

        # reset to trunk output
        current = self.trunk_output

        # build blocks
        #print(enumerate(head))
        print('building head')
        for bi, block_params in enumerate(head):
            print(current)
            current = self.build_block(current, block_params)


        # save head output
        self.head_output.append(current)

    ###################################################
    # compile model(s)
    ###################################################
    # self.model = tf.keras.Model(inputs=sequence, outputs=self.preds)
    self.models = []
    for ho in self.head_output:
        self.models.append(tf.keras.Model(inputs=sequence, outputs=ho))
    self.model = self.models[0]
    print(self.model.summary())


  def build_ensemble(self, ensemble_rc=False, ensemble_shifts=[0]):
    print('build ensemble')
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


  def evaluate(self, seq_data, head_i=0):
    """ Evaluate model on SeqDataset. """
    # choose model
    if self.ensemble is None:
      model = self.models[head_i]
    else:
      model = self.ensemble
      print('model = self.ensemble')
    # compile with dense metrics
    num_targets = self.model.output_shape[-1]
    model.compile(loss='poisson',
                  optimizer=tf.keras.optimizers.SGD(),
                  metrics=[metrics.PearsonR(num_targets, summarize=False),
                           metrics.R2(num_targets, summarize=False)])

    # evaluate
    return model.evaluate(seq_data.dataset)


  def predict(self, seq_data, head_i=0):
    """ Predict targets for SeqDataset. """
    # choose model
    if self.ensemble is None:
      model = self.models[head_i]
    else:
      model = self.ensemble

    return model.predict(seq_data.dataset)


  def restore(self, model_file):
    """ Restore weights from saved model. """
    self.model.load_weights(model_file)
