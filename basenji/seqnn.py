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

  def set_defaults(self):
    # only necessary for my bespoke parameters
    # others are best defaulted closer to the source
    self.augment_rc = False
    self.augment_shift = [0]
    self.strand_pair = []
    self.verbose = True

  def build_block(self, current, block_params):
    """Construct a SeqNN block.
    Args:
    Returns:
      current
    """
    block_args = {}

    # extract name
    block_name = block_params['name']

    # save upper_tri flatten
    self.preds_triu |= (block_name == 'upper_tri')
        
    # if Keras, get block variables names
    pass_all_globals = True
    if block_name[0].isupper():
      pass_all_globals = False
      block_func = blocks.keras_func[block_name]
      block_varnames = block_func.__init__.__code__.co_varnames

    # set global defaults
    global_vars = ['activation', 'batch_norm', 'bn_momentum', 'norm_type',
      'l2_scale', 'l1_scale', 'padding', 'kernel_initializer']
    for gv in global_vars:
      gv_value = getattr(self, gv, False)
      if gv_value and (pass_all_globals or gv in block_varnames):
        block_args[gv] = gv_value

    # set remaining params
    block_args.update(block_params)
    del block_args['name']

    # save representations
    if block_name.find('tower') != -1:
      block_args['reprs'] = self.reprs

    # U-net helper
    if block_name[-5:] == '_unet':
      # find matching representation
      unet_repr = None
      for seq_repr in reversed(self.reprs[:-1]):
        if seq_repr.shape[1] == current.shape[1]*2:
          unet_repr = seq_repr
          break
      if unet_repr is None:
        print('Could not find matching representation for length %d' % current.shape[1], sys.stderr)
        exit(1)
      block_args['unet_repr'] = unet_repr

    # switch for block
    if block_name[0].islower():
      block_func = blocks.name_func[block_name]
      current = block_func(current, **block_args)

    else:
      block_func = blocks.keras_func[block_name]
      current = block_func(**block_args)(current)

    return current

  def build_model(self, save_reprs=True):
    ###################################################
    # inputs
    ###################################################
    sequence = tf.keras.Input(shape=(self.seq_length, 4), name='sequence')
    current = sequence

    # augmentation
    if self.augment_rc:
      current, reverse_bool = layers.StochasticReverseComplement()(current)
    if self.augment_shift != [0]:
      current = layers.StochasticShift(self.augment_shift)(current)
    self.preds_triu = False
    
    ###################################################
    # build convolution blocks
    ###################################################
    self.reprs = []
    for bi, block_params in enumerate(self.trunk):
      current = self.build_block(current, block_params)
      if save_reprs:
        self.reprs.append(current)

    # final activation
    current = layers.activate(current, self.activation)

    # make model trunk
    trunk_output = current
    self.model_trunk = tf.keras.Model(inputs=sequence, outputs=trunk_output)

    ###################################################
    # heads
    ###################################################
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
        current = self.build_block(current, block_params)

      if hi < len(self.strand_pair):
        strand_pair = self.strand_pair[hi]
      else:
        strand_pair = None

      # transform back from reverse complement
      if self.augment_rc:
        if self.preds_triu:
          current = layers.SwitchReverseTriu(self.diagonal_offset)([current, reverse_bool])
        else:
          current = layers.SwitchReverse(strand_pair)([current, reverse_bool])

      # save head output
      self.head_output.append(current)

    ###################################################
    # compile model(s)
    ###################################################
    self.models = []
    for ho in self.head_output:
      self.models.append(tf.keras.Model(inputs=sequence, outputs=ho))
    self.model = self.models[0]
    if self.verbose: print(self.model.summary())

    ###################################################
    # track pooling/striding and cropping
    ###################################################
    self.model_strides = []
    self.target_lengths = []
    self.target_crops = []
    for model in self.models:
      # determine model stride
      self.model_strides.append(1)
      for layer in self.model.layers:
        if hasattr(layer, 'strides') or hasattr(layer, 'size'):
          stride_factor = layer.input_shape[1] / layer.output_shape[1]
          self.model_strides[-1] *= stride_factor
      self.model_strides[-1] = int(self.model_strides[-1])

      # determine predictions length before cropping
      if type(sequence.shape[1]) == tf.compat.v1.Dimension:
        target_full_length = sequence.shape[1].value // self.model_strides[-1]
      else:
        target_full_length = sequence.shape[1] // self.model_strides[-1]

      # determine predictions length after cropping
      self.target_lengths.append(model.outputs[0].shape[1])
      if type(self.target_lengths[-1]) == tf.compat.v1.Dimension:
        self.target_lengths[-1] = self.target_lengths[-1].value
      self.target_crops.append((target_full_length - self.target_lengths[-1])//2)
    
    if self.verbose:
      print('model_strides', self.model_strides)
      print('target_lengths', self.target_lengths)
      print('target_crops', self.target_crops)


  def build_embed(self, conv_layer_i, batch_norm=True):
    if conv_layer_i == -1:
      self.model = self.model_trunk

    else:
      if batch_norm:
        conv_layer = self.get_bn_layer(conv_layer_i)
      else:
        conv_layer = self.get_conv_layer(conv_layer_i)

      self.model = tf.keras.Model(inputs=self.model.inputs,
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

      if len(self.strand_pair) == 0:
        strand_pair = None
      else:
        strand_pair = self.strand_pair[0]

      # predict each sequence
      if self.preds_triu:
        preds = [layers.SwitchReverseTriu(self.diagonal_offset)
                  ([self.model(seq), rp]) for (seq,rp) in sequences_rev]
      else:
        if len(self.model.output_shape) == 2:
          # length collapsed, skip reversal
          preds = [self.model(seq) for (seq,rp) in sequences_rev]
          assert(strand_pair is None)  # not implemented yet
        else:
          preds = [layers.SwitchReverse(strand_pair)([self.model(seq), rp]) for (seq,rp) in sequences_rev]

      # create layer
      preds_avg = tf.keras.layers.Average()(preds)

      # create meta model
      self.ensemble = tf.keras.Model(inputs=sequence, outputs=preds_avg)


  def build_sad(self):
    # sequence input
    sequence = tf.keras.Input(shape=(self.seq_length, 4), name='sequence')

    # predict
    predictions = self.model(sequence)
    preds_len = predictions.shape[1]

    # sum pool
    sad = preds_len * tf.keras.layers.GlobalAveragePooling1D()(predictions)

    # replace model
    self.model = tf.keras.Model(inputs=sequence, outputs=sad)


  def build_slice(self, target_slice=None, target_sum=False):
    if target_slice is not None or target_sum:
      # sequence input
      sequence = tf.keras.Input(shape=(self.seq_length, 4), name='sequence')

      # predict
      predictions = self.model(sequence)

      # slice
      if target_slice is None:
        predictions_slice = predictions
      else:
        predictions_slice = tf.gather(predictions, target_slice, axis=-1)

      # sum
      if target_sum:
        predictions_sum = tf.reduce_sum(predictions_slice, keepdims=True, axis=-1)
      else:
        predictions_sum = predictions_slice

      # replace model
      self.model = tf.keras.Model(inputs=sequence, outputs=predictions_sum)


  def downcast(self, dtype=tf.float16, head_i=None):
    """ Downcast model output type. """
    # choose model
    if self.ensemble is not None:
      model = self.ensemble
    elif head_i is not None:
      model = self.models[head_i]
    else:
      model = self.model

    # sequence input
    sequence = tf.keras.Input(shape=(self.seq_length, 4), name='sequence')

    # predict and downcast
    preds = model(sequence)
    preds = tf.cast(preds, dtype)
    model_down = tf.keras.Model(inputs=sequence, outputs=preds)

    # replace model
    if self.ensemble is not None:
      self.ensemble = model_down
    elif head_i is not None:
      self.models[head_i] = model_down
    else:
      self.model = model_down


  def evaluate(self, seq_data, head_i=None, loss_label='poisson', loss_fn=None):
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

    if loss_fn is None:
      loss_fn = loss_label

    if loss_label == 'bce':
      model.compile(optimizer=tf.keras.optimizers.SGD(),
                    loss=loss_fn,
                    metrics=[metrics.SeqAUC(curve='ROC', summarize=False),
                             metrics.SeqAUC(curve='PR', summarize=False)])
    else:      
      model.compile(optimizer=tf.keras.optimizers.SGD(),
                    loss=loss_fn,
                    metrics=[metrics.PearsonR(num_targets, summarize=False),
                             metrics.R2(num_targets, summarize=False)])

    # evaluate
    return model.evaluate(seq_data.dataset)


  def get_bn_layer(self, bn_layer_i=0):
    """ Return specified batch normalization layer. """
    bn_layers = [layer for layer in self.model.layers if layer.name.startswith('batch_normalization')]
    return bn_layers[bn_layer_i]


  def get_conv_layer(self, conv_layer_i=0):
    """ Return specified convolution layer. """
    conv_layers = [layer for layer in self.model.layers if layer.name.startswith('conv')]
    return conv_layers[conv_layer_i]


  def get_dense_layer(self, layer_i=0):
    """ Return specified convolution layer. """
    dense_layers = [layer for layer in self.model.layers if layer.name.startswith('dense')]
    return dense_layers[layer_i]


  def get_conv_weights(self, conv_layer_i=0):
    """ Return kernel weights for specified convolution layer. """
    conv_layer = self.get_conv_layer(conv_layer_i)
    weights = conv_layer.weights[0].numpy()
    weights = np.transpose(weights, [2,1,0])
    return weights


  def gradients(self, seq_1hot, head_i=None, pos_slice=None, batch_size=4, dtype='float16'):
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

    # batching parameters
    num_targets = model.output_shape[-1]
    num_batches = int(np.ceil(num_targets / batch_size))

    ti_start = 0
    grads = []
    for bi in range(num_batches):
      # sequence input
      sequence = tf.keras.Input(shape=(self.seq_length, 4), name='sequence')

      # predict
      predictions = model(sequence)

      # slice
      ti_end = min(num_targets, ti_start + batch_size)
      target_slice = np.arange(ti_start, ti_end)
      predictions_slice = tf.gather(predictions, target_slice, axis=-1)

      # replace model
      model_batch = tf.keras.Model(inputs=sequence, outputs=predictions_slice)

      with tf.GradientTape() as tape:
        tape.watch(seq_1hot)

        # predict
        preds = model_batch(seq_1hot, training=False)

        if pos_slice is not None:
          # slice specified positions
          preds = tf.gather(preds, pos_slice, axis=-2)
        
        # sum across positions
        preds = tf.reduce_sum(preds, axis=-2)

      # compute jacboian
      grads_batch = tape.jacobian(preds, seq_1hot)
      grads_batch = tf.squeeze(grads_batch)
      grads_batch = tf.transpose(grads_batch, [1,2,0])

      # zero mean each position
      grads_batch = grads_batch - tf.reduce_mean(grads_batch, axis=-2, keepdims=True)

      # convert numpy dtype
      grads_batch = grads_batch.numpy().astype(dtype)
      grads.append(grads_batch)

      # next batch
      ti_start += batch_size

    # concat target batches
    grads = np.concatenate(grads, axis=-1)

    return grads


  def num_targets(self, head_i=None):
    if head_i is None:
      return self.model.output_shape[-1]
    else:
      return self.models[head_i].output_shape[-1]


  def predict(self, seq_data, head_i=None, generator=False, stream=False, dtype='float32', **kwargs):
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
      return model.predict_generator(dataset, **kwargs).astype(dtype)
    elif stream:
      preds = []
      for x, y in seq_data.dataset:
        yh = model.predict(x, **kwargs)
        preds.append(yh.astype(dtype))
      return np.concatenate(preds, axis=0, dtype=dtype)
    else:
      return model.predict(dataset, **kwargs).astype(dtype)


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

  def step(self, step=2, head_i=None):
    """ Step positions across sequence. """
    # choose model
    if self.ensemble is not None:
      model = self.ensemble
    elif head_i is not None:
      model = self.models[head_i]
    else:
      model = self.model

    # sequence input
    sequence = tf.keras.Input(shape=(self.seq_length, 4), name='sequence')

    # predict and step across positions
    preds = model(sequence)
    step_positions = np.arange(preds.shape[1], step=step)
    preds_step = tf.gather(preds, step_positions, axis=-2)
    model_step = tf.keras.Model(inputs=sequence, outputs=preds_step)

    # replace model
    if self.ensemble is not None:
      self.ensemble = model_step
    elif head_i is not None:
      self.models[head_i] = model_step
    else:
      self.model = model_step
