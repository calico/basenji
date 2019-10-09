# Copyright 2019 Calico LLC
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

from natsort import natsorted
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
    self.embed = None

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

    # extract name
    block_name = block_params['name']
    del block_params['name']

    # if Keras, get block variables names
    pass_all_globals = True
    if block_name[0].isupper():
      pass_all_globals = False
      block_func = blocks.keras_func[block_name]
      block_varnames = block_func.__init__.__code__.co_varnames

    # set global defaults
    global_vars = ['activation', 'batch_norm', 'bn_momentum',
      'l2_scale', 'l1_scale']
    for gv in global_vars:
      gv_value = getattr(self, gv, False)
      if gv_value and (pass_all_globals or gv in block_varnames):
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
      requires_reversal = True
    else:
      requires_reversal = False
    current = layers.StochasticShift(self.augment_shift)(current)

    ###################################################
    # build convolution blocks
    ###################################################
    for bi, block_params in enumerate(self.trunk):
      current = self.build_block(current, block_params)

    # final activation
    current = layers.activate(current, self.activation)

    # ORIGINAL
    # if self.augment_rc:
    #   current = layers.SwitchReverse()([current, reverse_bool])

    trunk_output = current
    self.model_trunk = tf.keras.Model(inputs=sequence, outputs=trunk_output)

    ###################################################
    # heads
    ###################################################
    irreversible_blocks = ['upper_triu']

    head_keys = natsorted([v for v in vars(self) if v.startswith('head')])
    self.heads = [getattr(self, hk) for hk in head_keys]

    self.head_output = []
    for hi, head in enumerate(self.heads):
      if not isinstance(head, list):
          head = [head]

      # reset to trunk output
      current = trunk_output

      # build blocks
      for bi, block_params in enumerate(head):
          if requires_reversal and block_params['name'] in irreversible_blocks:
            # transform back from reverse complement before next block
            current = layers.SwitchReverse()([current, reverse_bool])
            requires_reversal = False
          
          current = self.build_block(current, block_params)

      if requires_reversal:
        # transform back from reverse complement
        current = layers.SwitchReverse()([current, reverse_bool])

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


  def build_embed(self, conv_layer_i):
    if conv_layer_i == -1:
      self.embed = tf.keras.Model(inputs=self.model.inputs,
                                  outputs=self.model.inputs)
    else:
      conv_layer = self.get_bn_layer(conv_layer_i)
      self.embed = tf.keras.Model(inputs=self.model.inputs,
                                  outputs=conv_layer.output)


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


  def evaluate(self, seq_data, head_i=0, loss='poisson'):
    """ Evaluate model on SeqDataset. """
    # choose model
    if self.ensemble is None:
      model = self.models[head_i]
    else:
      model = self.ensemble

    # compile with dense metrics
    num_targets = self.model.output_shape[-1]
    model.compile(optimizer=tf.keras.optimizers.SGD(),
                  loss=loss,
                  metrics=[metrics.PearsonR(num_targets, summarize=False),
                           metrics.R2(num_targets, summarize=False)])

    # evaluate
    return model.evaluate(seq_data.dataset)


  def get_bn_layer(self, bn_layer_i):
    """ Return specified batch normalization layer. """
    bn_layers = [layer for layer in self.model.layers if layer.name.startswith('batch_normalization')]
    return bn_layers[bn_layer_i]


  def get_conv_layer(self, conv_layer_i):
    """ Return specified convolution layer. """
    conv_layers = [layer for layer in self.model.layers if layer.name.startswith('conv')]
    return conv_layers[conv_layer_i]


  def get_conv_weights(self, conv_layer_i):
    """ Return kernel weights for specified convolution layer. """
    conv_layer = self.get_conv_layer(conv_layer_i)
    weights = conv_layer.weights[0].numpy()
    weights = np.transpose(weights, [2,1,0])
    return weights


  def num_targets(self, head_i=0):
    return self.models[head_i].output_shape[-1]


  def predict(self, seq_data, head_i=0, **kwargs):
    """ Predict targets for SeqDataset. """
    # choose model
    if self.embed is not None:
      model = self.embed
    elif self.ensemble is not None:
      model = self.ensemble
    else:
      model = self.models[head_i]

    dataset = getattr(seq_data, 'dataset', None)
    if dataset is None:
      dataset = seq_data

    return model.predict(dataset, **kwargs)


  def restore(self, model_file, trunk=False):
    """ Restore weights from saved model. """
    if trunk:
      self.model_trunk.load_weights(model_file)
    else:
      self.model.load_weights(model_file)


  def save(self, model_file, trunk=False):
    if trunk:
      self.model_trunk.save(model_file)
    else:
      self.model.save(model_file)
