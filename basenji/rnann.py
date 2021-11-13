# Copyright 2021 Calico LLC
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

import numpy as np
import tensorflow as tf

from basenji import layers
from basenji import metrics

class RnaNN:
  def __init__(self, params):
    self.set_defaults()
    for key, value in params.items():
      self.__setattr__(key, value)
    self.build_model()
    self.ensemble = None

  def set_defaults(self):
    # only necessary for my bespoke parameters
    # others are best defaulted closer to the source
    self.seq_depth = 6
    self.augment_shift = [0]
    self.num_targets = 1
    self.initializer = 'he_normal'
    self.l2_scale = 0
    self.activation = 'relu'
    self.residual = False
    self.heads = 1
    self.go_backwards = True
    self.num_dlayers = 0

  def build_model(self):
    ###################################################
    # inputs
    ###################################################
    sequence = tf.keras.Input(shape=(self.seq_length, self.seq_depth), name='sequence')
    current = sequence

    # augmentation
    if self.augment_shift != [0]:
      current = layers.StochasticShift(self.augment_shift, symmetric=False)(current)
    
    ###################################################
    # initial
    ###################################################

    # RNA convolution
    current = tf.keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, padding='valid',
                                     kernel_initializer=self.initializer, use_bias=False,
                                     kernel_regularizer=tf.keras.regularizers.l2(self.l2_scale))(current)
    if self.residual:
      initial = current
      current = tf.keras.layers.LayerNormalization(epsilon=self.ln_epsilon)(current)
      current = layers.activate(current, self.activation)
      current = tf.keras.layers.Conv1D(filters=self.filters, kernel_size=1, padding='valid',
                                       kernel_initializer=self.initializer,
                                       kernel_regularizer=tf.keras.regularizers.l2(self.l2_scale))(current)
      current = tf.keras.layers.Dropout(self.dropout)(current)
      current = layers.Scale()(current)
      current = tf.keras.layers.Add()([initial,current])

    # middle convolutions
    for mi in range(self.num_layers):
      current = tf.keras.layers.LayerNormalization(epsilon=self.ln_epsilon)(current)
      current = layers.activate(current, self.activation)
      current = tf.keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, padding='valid',
                                       kernel_initializer=self.initializer,
                                       kernel_regularizer=tf.keras.regularizers.l2(self.l2_scale))(current)
      current = tf.keras.layers.Dropout(self.dropout)(current)
      if self.residual:
        initial = current
        current = tf.keras.layers.LayerNormalization(epsilon=self.ln_epsilon)(current)
        current = layers.activate(current, self.activation)
        current = tf.keras.layers.Conv1D(filters=self.filters, kernel_size=1, padding='valid',
                                         kernel_initializer=self.initializer,
                                         kernel_regularizer=tf.keras.regularizers.l2(self.l2_scale))(current)
        current = tf.keras.layers.Dropout(self.dropout)(current)
        current = layers.Scale()(current)
        current = tf.keras.layers.Add()([initial,current])
      current = tf.keras.layers.MaxPooling1D()(current)

    # dilated residual convolutions
    # drate = 1.0
    # for di in range(self.num_dlayers):
    #   initial = current
    #   drate *= 2.0
    #   current = tf.keras.layers.LayerNormalization(epsilon=self.ln_epsilon)(current)
    #   current = layers.activate(current, self.activation)
    #   current = tf.keras.layers.Conv1D(filters=self.filters, kernel_size=3, padding='same',
    #                                    kernel_initializer=self.initializer, dilation_rate=int(np.round(drate)),
    #                                    kernel_regularizer=tf.keras.regularizers.l2(self.l2_scale))(current)
    #   current = tf.keras.layers.Dropout(self.dropout)(current)
    #   current = layers.Scale()(current)
    #   current = tf.keras.layers.Add()([initial,current])

    # aggregate sequence
    current = tf.keras.layers.LayerNormalization(epsilon=self.ln_epsilon)(current)
    current = layers.activate(current, self.activation)
    current = tf.keras.layers.LSTM(self.filters, go_backwards=self.go_backwards, kernel_initializer=self.initializer,
                                   kernel_regularizer=tf.keras.regularizers.l2(self.l2_scale))(current)
    # current = tf.keras.layers.LSTM(self.filters,
    #   return_sequences=True,
    #   kernel_initializer=self.initializer,
    #   kernel_regularizer=tf.keras.regularizers.l2(self.l2_scale))(current)
    # current = layers.LengthAverage()(current, sequence)

    # penultimate
    current = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)(current)
    current = layers.activate(current, self.activation)
    current = tf.keras.layers.Dense(self.filters,
                                    kernel_initializer=self.initializer,
                                    kernel_regularizer=tf.keras.regularizers.l2(self.l2_scale))(current)
    current = tf.keras.layers.Dropout(self.dropout)(current)
    if self.residual:
      initial = current
      current = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)(current)
      current = layers.activate(current, self.activation)
      current = tf.keras.layers.Dense(self.filters,
                                      kernel_initializer=self.initializer,
                                      kernel_regularizer=tf.keras.regularizers.l2(self.l2_scale))(current)
      current = tf.keras.layers.Dropout(self.dropout)(current)
      current = layers.Scale()(current)
      current = tf.keras.layers.Add()([initial,current])

    # final representation
    current = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)(current)
    current = layers.activate(current, self.activation)

    ###################################################
    # compile model(s)
    ###################################################
    self.models = []
    for hi in range(self.heads):
      prediction = tf.keras.layers.Dense(self.num_targets,
                                         kernel_initializer=self.initializer)(current)
      self.models.append(tf.keras.Model(inputs=sequence, outputs=prediction))
    
    self.model = self.models[0]
    print(self.model.summary())

  def build_ensemble(self, ensemble_shifts=[0]):
    """ Build ensemble of models computing on augmented input sequences. """
    if len(ensemble_shifts) > 1:
      # sequence input
      sequence = tf.keras.Input(shape=(self.seq_length, 4), name='sequence')
      sequences = [sequence]

      # generate shifted sequences
      sequences = layers.EnsembleShift(ensemble_shifts)(sequences)

      # predict each sequence
      preds = [self.model(seq) for seq in sequences]

      # create layer
      preds_avg = tf.keras.layers.Average()(preds)

      # create meta model
      self.ensemble = tf.keras.Model(inputs=sequence, outputs=preds_avg)

  def evaluate(self, seq_data, head_i=None, loss='mse'):
    """ Evaluate model on SeqDataset. """

    # choose model
    if self.ensemble is not None:
      model = self.ensemble
    elif head_i is not None:
      model = self.models[head_i]
    else:
      model = self.model

    # compile with dense metrics
    num_targets = model.output_shape[-1]
   
    model.compile(optimizer=tf.keras.optimizers.SGD(),
                  loss=loss,
                  metrics=[metrics.PearsonR(num_targets, summarize=False),
                           metrics.R2(num_targets, summarize=False)])

    # evaluate
    return model.evaluate(seq_data.dataset)

  def predict(self, seq_data, head_i=None, generator=False, **kwargs):
    """ Predict targets for SeqDataset. """
    # choose model
    if self.ensemble is not None:
      model = self.ensemble
    elif head_i is not None:
      model = self.models[head_i]
    else:
      model = self.model

    dataset = getattr(seq_data, 'dataset', None)
    if dataset is None:
      dataset = seq_data

    if generator:
      return model.predict_generator(dataset, **kwargs)
    else:
      return model.predict(dataset, **kwargs)

  def restore(self, model_file, head_i=0, trunk=False):
    """ Restore weights from saved model. """
    if trunk:
      self.model_trunk.load_weights(model_file)
    else:
      self.models[head_i].load_weights(model_file)
      self.model = self.models[head_i]

  def save(self, model_file, trunk=False):
    if trunk:
      self.model_trunk.save(model_file, include_optimizer=False)
    else:
      self.model.save(model_file, include_optimizer=False)
