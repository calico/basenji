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
    self.num_targets = [1]
    self.initializer = 'he_normal'
    self.l2_scale = 0
    self.activation = 'relu'
    self.residual = False
    self.heads = 1
    self.go_backwards = True
    self.rnn_type = 'gru'

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

    # aggregate sequence
    current = tf.keras.layers.LayerNormalization(epsilon=self.ln_epsilon)(current)
    current = layers.activate(current, self.activation)
    rnn_layer = tf.keras.layers.GRU
    if self.rnn_type == 'lstm':
      rnn_layer = tf.keras.layers.LSTM
    current = rnn_layer(self.filters, go_backwards=self.go_backwards, kernel_initializer=self.initializer,
                        kernel_regularizer=tf.keras.regularizers.l2(self.l2_scale))(current)

    # attention = tf.keras.layers.LayerNormalization(epsilon=self.ln_epsilon)(current)
    # attention = layers.activate(attention, self.activation)
    # attention = tf.keras.layers.Dense(units=self.filters,
    #                                   kernel_initializer=self.initializer,
    #                                   kernel_regularizer=tf.keras.regularizers.l2(self.l2_scale))(attention)
    # attention = tf.keras.layers.Softmax(axis=-2)(attention)
    # current *= attention
    # current = tf.keras.layers.GlobalAveragePooling1D()(current)

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
    if not isinstance(self.num_targets, list):
      self.num_targets = [self.num_targets]
    self.heads = len(self.num_targets)

    self.models = []
    for hi in range(self.heads):
      prediction = tf.keras.layers.Dense(self.num_targets[hi],
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


  def gradients(self, seq_1hot, head_i=None, dtype='float16'):
    """ Compute input gradients sequence. """
    # choose model
    if self.ensemble is not None:
      model = self.ensemble
    elif head_i is not None:
      model = self.models[head_i]
    else:
      model = self.model

    # verify tensor shape
    seq_1hot = seq_1hot.astype('float32')
    seq_1hot = tf.convert_to_tensor(seq_1hot, dtype=tf.float32)
    if len(seq_1hot.shape) < 3:
      seq_1hot = tf.expand_dims(seq_1hot, axis=0)

    with tf.GradientTape() as tape:
      tape.watch(seq_1hot)

      # predict
      preds = model(seq_1hot, training=False)

    # compute jacboian
    grads = tape.jacobian(preds, seq_1hot)
    grads = tf.squeeze(grads)
    # grads = tf.transpose(grads, [1,2,0])

    # I'm confused about which access would be which for multi-task

    # convert numpy dtype
    grads = grads.numpy()

    # zero mean each position
    grads[:,:4] = grads[:,:4] - grads[:,:4].mean(axis=-1, keepdims=True)

    return grads.astype(dtype)


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
