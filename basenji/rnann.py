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

class RnaNN:
  def __init__(self, params):
    self.set_defaults()
    for key, value in params.items():
      self.__setattr__(key, value)
    self.build_model()
    self.ensemble = None

  def set_defaults(self):
    # only necessary for my bespoke parameters
    # others are  best defaulted closer to the source
    self.augment_shift = [0]

  def build_model(self):
    ###################################################
    # inputs
    ###################################################
    seq_depth = 5 if self.rna_mode == 'full' else 4
    sequence = tf.keras.Input(shape=(self.seq_length, seq_depth), name='sequence')
    features = tf.keras.Input(shape=(self.num_features,), name='features')
    current = sequence

    # augmentation
    if self.augment_shift != [0]:
      current = layers.StochasticShift(self.augment_shift, symmetric=False)(current)
    
    ###################################################
    # initial
    ###################################################

    # RNA convolution
    current = tf.keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, padding='valid',
                                     kernel_initializer='he_normal',
                                     kernel_regularizer=tf.keras.regularizers.l2(self.l2_scale))(current)
    current = tf.keras.layers.LayerNormalization(epsilon=self.ln_epsilon)(current)
    current = tf.keras.layers.Dropout(self.dropout)(current)
    current = tf.keras.layers.ReLU()(current)
    current = tf.keras.layers.MaxPooling1D()(current)

    # middle convolutions
    for mi in range(self.num_layers-1):
      current = tf.keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, padding='valid',
                                       kernel_initializer='he_normal',
                                       kernel_regularizer=tf.keras.regularizers.l2(self.l2_scale))(current)
      current = tf.keras.layers.LayerNormalization(epsilon=self.ln_epsilon)(current)
      current = tf.keras.layers.Dropout(self.dropout)(current)
      current = tf.keras.layers.ReLU()(current)
      current = tf.keras.layers.MaxPooling1D()(current)

    # aggregate sequence
    current = tf.keras.layers.LSTM(self.filters, go_backwards=True, kernel_initializer='he_normal',
                                   kernel_regularizer=tf.keras.regularizers.l2(self.l2_scale))(current)

    # concat features
    current = tf.keras.layers.Concatenate()([current, features])

    # penultimate
    current = tf.keras.layers.Dense(self.filters,
                                    kernel_initializer='he_normal',
                                    kernel_regularizer=tf.keras.regularizers.l2(self.l2_scale))(current)
    current = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)(current)
    current = tf.keras.layers.Dropout(self.dropout)(current)
    current = tf.keras.layers.ReLU()(current)

    # final
    prediction = tf.keras.layers.Dense(1)(current)

    ###################################################
    # compile model(s)
    ###################################################
    self.model = tf.keras.Model(inputs=[sequence,features], outputs=prediction)
    print(self.model.summary())