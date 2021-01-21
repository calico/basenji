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
import pdb
import sys

import numpy as np
import tensorflow as tf

# from tensor2tensor.layers.common_attention import attention_bias_proximal
# from tensor2tensor.layers.common_attention import _generate_relative_positions_embeddings
# from tensor2tensor.layers.common_attention import _relative_attention_inner

############################################################
# Basic
############################################################

class Clip(tf.keras.layers.Layer):
  def __init__(self, min_value, max_value):
    super(Clip, self).__init__()
    self.min_value = min_value
    self.max_value = max_value
  def call(self, x):
    return tf.clip_by_value(x, self.min_value, self.max_value)
  def get_config(self):
    config = super().get_config().copy()
    config.update({
      'min_value': self.min_value,
      'max_value': self.max_value
    })
    return config

class Exp(tf.keras.layers.Layer):
  def __init__(self, base=None, minus=None):
    super(Exp, self).__init__()
    if base is None:
      self.base = None
    else:
      self.base = tf.constant(base, dtype=tf.float32)
    if minus is None:
      self.minus = None
    else:
      self.minus = tf.constant(minus, dtype=tf.float32)

  def call(self, x):
    if self.base is None:
      y = tf.keras.activations.exponential(x)
    else:
      y = tf.math.pow(self.base, x)

    if self.minus is not None:
      y -= self.minus

    return y
  def get_config(self):
    config = super().get_config().copy()
    config['base'] = self.base
    config['minus'] = self.minus
    return config

class PolyReLU(tf.keras.layers.Layer):
  def __init__(self, shift=0):
    super(PolyReLU, self).__init__()

  def call(self, x):
    x3 = tf.math.pow((x-2), 3)
    y = tf.keras.activations.relu(x3)
    return y

class GELU(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(GELU, self).__init__(**kwargs)
  def call(self, x):
    # return tf.keras.activations.sigmoid(1.702 * x) * x
    return tf.keras.activations.sigmoid(tf.constant(1.702) * x) * x

class Softplus(tf.keras.layers.Layer):
  def __init__(self, exp_max=10000):
    super(Softplus, self).__init__()
    self.exp_max = exp_max
  def call(self, x):
    x = tf.clip_by_value(x, -self.exp_max, self.exp_max)
    return tf.keras.activations.softplus(x)
  def get_config(self):
    config = super().get_config().copy()
    config['exp_max'] = self.exp_max
    return config

############################################################
# Center ops
############################################################

class CenterSlice(tf.keras.layers.Layer):
  def __init__(self, center):
    super(CenterSlice, self).__init__()
    self.center = center
  def call(self, x):
    seq_len = x.shape[1]
    center_start = (seq_len - self.center) // 2
    center_end = center_start + self.center
    return x[:, center_start:center_end, :]
  def get_config(self):
    config = super().get_config().copy()
    config.update({
      'center': self.center
    })
    return config

class CenterAverage(tf.keras.layers.Layer):
  def __init__(self, center):
    super(CenterAverage, self).__init__()
    self.center = center
    self.slice = CenterSlice(self.center)
  def call(self, x):
    return tf.keras.backend.mean(self.slice(x), axis=1, keepdims=True) 
  def get_config(self):
    config = super().get_config().copy()
    config.update({
      'center': self.center
    })
    return config

############################################################
# Attention
############################################################
class Attention(tf.keras.layers.Layer):
  def __init__(self, max_relative_position, dropout=0):
    super(Attention, self).__init__()
    self.max_relative_position = max_relative_position
    self.dropout = dropout

  def build(self, input_shape):
    # extract shapes
    qs, vs, ks = input_shape
    seq_length = qs[-2]
    depth_kq = qs[-1]
    depth_q = ks[-1]
    assert(depth_kq == depth_q)
    depth_v = vs[-1]

    # initialize bias
    self.attn_bias = attention_bias_proximal(seq_length)

    # initialize relative positions
    self.relations_keys = _generate_relative_positions_embeddings(
      seq_length, seq_length, depth_kq, self.max_relative_position,
      "relative_positions_keys")
    self.relations_values = _generate_relative_positions_embeddings(
      seq_length, seq_length, depth_v, self.max_relative_position,
      "relative_positions_values")
    # tf.contrib.summary.histogram('relations_keys', self.relations_keys)
    # tf.contrib.summary.histogram('relations_values', self.relations_values)

  def call(self, qvk):
    query, value, key = qvk

    # expand to fake multi-head
    key = tf.expand_dims(key, axis=1)
    query = tf.expand_dims(query, axis=1)
    value = tf.expand_dims(value, axis=1)

    # Compute self attention considering the relative position embeddings.
    logits = _relative_attention_inner(query, key, self.relations_keys, True)
    logits += self.attn_bias

    weights = tf.nn.softmax(logits, name="attention_weights")
    weights = tf.nn.dropout(weights, rate=self.dropout)

    z = _relative_attention_inner(weights, value, self.relations_values, False)

    # slice single head
    z = z[:,0,:,:]

    return z

  def get_config(self):
    config = super().get_config().copy()
    config.update({
      'max_relative_position': self.max_relative_position,
      'dropout': self.dropout
    })
    return config


class WheezeExcite(tf.keras.layers.Layer):
  def __init__(self, pool_size):
    super(WheezeExcite, self).__init__()
    self.pool_size = pool_size
    assert(self.pool_size % 2 == 1)
    self.paddings = [[0,0], [self.pool_size//2, self.pool_size//2], [0,0]]

  def build(self, input_shape):
    self.num_channels = input_shape[-1]

    self.wheeze = tf.keras.layers.AveragePooling1D(self.pool_size,
        strides=1, padding='valid')

    self.excite1 = tf.keras.layers.Dense(
      units=self.num_channels//4,
      activation='relu')
    self.excite2 = tf.keras.layers.Dense(
      units=self.num_channels,
      activation='relu')

  def call(self, x):
    # pad
    x_pad = tf.pad(x, self.paddings, 'SYMMETRIC')

    # squeeze
    x_squeeze = self.wheeze(x_pad)

    # excite
    x_excite = self.excite1(x_squeeze)
    x_excite = self.excite2(x_excite)
    x_excite = tf.keras.activations.sigmoid(x_excite)

    # scale
    xs = x * x_excite

    return xs

  def get_config(self):
    config = super().get_config().copy()
    config.update({
      'pool_size': self.pool_size
    })
    return config


class SqueezeExcite(tf.keras.layers.Layer):
  def __init__(self, activation='relu', additive=False, bottleneck_ratio=8,
    batch_norm=False, bn_momentum=0.9):
    super(SqueezeExcite, self).__init__()
    self.activation = activation
    self.additive = additive
    self.batch_norm = batch_norm
    self.bn_momentum = bn_momentum
    self.bottleneck_ratio = bottleneck_ratio

  def build(self, input_shape):
    self.num_channels = input_shape[-1]

    if len(input_shape) == 3:
      self.one_or_two = 'one'
      self.gap = tf.keras.layers.GlobalAveragePooling1D()
    elif len(input_shape) == 4:
      self.one_or_two = 'two'
      self.gap = tf.keras.layers.GlobalAveragePooling2D()
    else:
      print('SqueezeExcite: input dim %d unexpected' % len(input_shape), file=sys.stderr)
      exit(1)

    self.dense1 = tf.keras.layers.Dense(
      units=self.num_channels//self.bottleneck_ratio,
      activation='relu')
    self.dense2 = tf.keras.layers.Dense(
      units=self.num_channels,
      activation=None)
    if self.batch_norm:
      self.bn = tf.keras.layers.BatchNormalization(
        momentum=self.bn_momentum,
        gamma_initializer='zeros')

  def call(self, x):
    # activate
    x = activate(x, self.activation)

    # squeeze
    squeeze = self.gap(x)

    # excite
    excite = self.dense1(squeeze)
    excite = self.dense2(excite)
    if self.batch_norm:
      excite = self.bn(excite)

    # scale
    if self.one_or_two == 'one':
      excite = tf.reshape(excite, [-1,1,self.num_channels])
    else:
      excite = tf.reshape(excite, [-1,1,1,self.num_channels])

    if self.additive:
      xs = x + excite
    else:
      excite = tf.keras.activations.sigmoid(excite)
      xs = x * excite

    return xs

  def get_config(self):
    config = super().get_config().copy()
    config.update({
      'activation': self.activation,
      'additive': self.additive,
      'batch_norm': self.batch_norm,
      'bn_momentum': self.bn_momentum,
      'bottleneck_ratio': self.bottleneck_ratio
    })
    return config

class GlobalContext(tf.keras.layers.Layer):
  def __init__(self):
    super(GlobalContext, self).__init__()

  def build(self, input_shape):
    self.num_channels = input_shape[-1]

    self.context_key = tf.keras.layers.Dense(units=1, activation=None)

    self.dense1 = tf.keras.layers.Dense(units=self.num_channels//4)
    self.ln = tf.keras.layers.LayerNormalization()
    self.dense2 = tf.keras.layers.Dense(units=self.num_channels)

  def call(self, x):
    # context attention
    keys = self.context_key(x) # [batch x length x 1]
    attention = tf.keras.activations.softmax(keys, axis=-2) # [batch x length x 1]

    # context summary 
    context = x * attention # [batch x length x channels]
    context = tf.keras.backend.sum(context, axis=-2, keepdims=True) # [batch x 1 x channels]

    # transform
    transform = self.dense1(context) # [batch x 1 x channels/4]
    transform = tf.keras.activations.relu(self.ln(transform)) # [batch x 1 x channels/4]
    transform = self.dense2(transform) # [batch x 1 x channels]
    # transform = tf.reshape(transform, [-1,1,self.num_channels])

    # fusion
    xs = x + transform # [batch x length x channels]

    return xs

############################################################
# Position
############################################################
class ConcatPosition(tf.keras.layers.Layer):
  ''' Concatenate position to 1d feature vectors.'''

  def __init__(self, transform=None, power=1):
    super(ConcatPosition, self).__init__()
    self.transform = transform
    self.power = power

  def call(self, inputs):
    input_shape = tf.shape(inputs)
    batch_size, seq_len = input_shape[0], input_shape[1]

    pos_range = tf.range(-seq_len//2, seq_len//2)
    if self.transform is None:
      pos_feature = pos_range
    elif self.transform == 'abs':
      pos_feature = tf.math.abs(pos_range)
    elif self.transform == 'reversed':
      pos_feature = pos_range[::-1]
    else:
      raise ValueError('Unknown ConcatPosition transform.')

    if self.power != 1:
      pos_feature = tf.pow(pos_feature, self.power)
    pos_feature = tf.expand_dims(pos_feature, axis=0)
    pos_feature = tf.expand_dims(pos_feature, axis=-1)
    pos_feature = tf.tile(pos_feature, [batch_size, 1, 1])
    pos_feature = tf.dtypes.cast(pos_feature, dtype=tf.float32)

    return tf.concat([pos_feature, inputs], axis=-1)

  def get_config(self):
    config = super().get_config().copy()
    config.update({
      'transform': self.transform,
      'power': self.power
    })
    return config


############################################################
# 2D
############################################################
class OneToTwo(tf.keras.layers.Layer):
  ''' Transform 1d to 2d with i,j vectors operated on.'''
  def __init__(self, operation='mean'):
    super(OneToTwo, self).__init__()
    self.operation = operation.lower()
    valid_operations = ['concat','mean','max','multipy','multiply1']
    assert self.operation in valid_operations

  def call(self, oned):
    _, seq_len, features = oned.shape

    twod1 = tf.tile(oned, [1, seq_len, 1])
    twod1 = tf.reshape(twod1, [-1, seq_len, seq_len, features])
    twod2 = tf.transpose(twod1, [0,2,1,3])

    if self.operation == 'concat':
      twod  = tf.concat([twod1, twod2], axis=-1)

    elif self.operation == 'multiply':
      twod  = tf.multiply(twod1, twod2)

    elif self.operation == 'multiply1':
      twod = tf.multiply(twod1+1, twod2+1) - 1

    else:
      twod1 = tf.expand_dims(twod1, axis=-1)
      twod2 = tf.expand_dims(twod2, axis=-1)
      twod  = tf.concat([twod1, twod2], axis=-1)

      if self.operation == 'mean':
        twod = tf.reduce_mean(twod, axis=-1)

      elif self.operation == 'max':
        twod = tf.reduce_max(twod, axis=-1)

    return twod

  def get_config(self):
    config = super().get_config().copy()
    config['operation'] = self.operation
    return config

# depracated: use OneToTwo
class AverageTo2D(tf.keras.layers.Layer):
  ''' Transform 1d to 2d with i,j vectors averaged.'''
  def __init__(self):
    super(AverageTo2D, self).__init__()

  def call(self,inputs):
    input_shape = tf.shape(inputs)
    assert len(inputs.shape)==3
    batch_size, seq_len, output_dim = inputs.shape

    matrix_repr1 = tf.tile(inputs, [1, seq_len, 1])
    matrix_repr1 = tf.reshape(matrix_repr1, [-1, seq_len, seq_len, output_dim])
    matrix_repr2 = tf.transpose(matrix_repr1, [0,2,1,3])

    matrix_repr1 = tf.expand_dims(matrix_repr1, axis=-1)
    matrix_repr2 = tf.expand_dims(matrix_repr2, axis=-1)
    current  = tf.concat([matrix_repr1, matrix_repr2], axis=-1)
    current = tf.reduce_mean(current, axis=-1)

    return current

# depracated: use OneToTwo
class MaxTo2D(tf.keras.layers.Layer):
  ''' Transform 1d to 2d with i,j vectors maxed.'''
  def __init__(self):
    super(MaxTo2D, self).__init__()

  def call(self,inputs):
    input_shape = tf.shape(inputs)
    assert len(inputs.shape)==3
    batch_size, seq_len, output_dim = inputs.shape

    matrix_repr1 = tf.tile(inputs, [1, seq_len, 1])
    matrix_repr1 = tf.reshape(matrix_repr1, [-1, seq_len, seq_len, output_dim])
    matrix_repr2 = tf.transpose(matrix_repr1, [0,2,1,3])

    matrix_repr1 = tf.expand_dims(matrix_repr1, axis=-1)
    matrix_repr2 = tf.expand_dims(matrix_repr2, axis=-1)
    current  = tf.concat([matrix_repr1, matrix_repr2], axis=-1)
    current = tf.reduce_max(current, axis=-1)

    return current

# depracated: use OneToTwo
class DotTo2D(tf.keras.layers.Layer):
  ''' Transform 1d to 2d with i,j vectors maxed.'''
  def __init__(self):
    super(DotTo2D, self).__init__()

  def call(self,inputs):
    input_shape = tf.shape(inputs)
    assert len(inputs.shape)==3
    batch_size, seq_len, output_dim = inputs.shape

    matrix_repr1 = tf.tile(inputs, [1, seq_len, 1])
    matrix_repr1 = tf.reshape(matrix_repr1, [-1, seq_len, seq_len, output_dim])
    matrix_repr2 = tf.transpose(matrix_repr1, [0,2,1,3])

    current  = tf.multiply(matrix_repr1, matrix_repr2)

    return current

# depracated: use OneToTwo
class GeoDotTo2D(tf.keras.layers.Layer):
  ''' Transform 1d to 2d with i,j vectors maxed.'''
  def __init__(self):
    super(GeoDotTo2D, self).__init__()

  def call(self,inputs):
    input_shape = tf.shape(inputs)
    assert len(inputs.shape)==3
    batch_size, seq_len, output_dim = inputs.shape

    matrix_repr1 = tf.tile(inputs, [1, seq_len, 1])
    matrix_repr1 = tf.reshape(matrix_repr1, [-1, seq_len, seq_len, output_dim])
    matrix_repr2 = tf.transpose(matrix_repr1, [0,2,1,3])

    current = tf.multiply(matrix_repr1+1, matrix_repr2+1)
    current = tf.sqrt(current)-1

    return current

# depracated: use OneToTwo
class ConcatTo2D(tf.keras.layers.Layer):
  ''' Transform 1d to 2d with i,j vectors concatenated.'''
  def __init__(self):
    super(ConcatTo2D, self).__init__()

  def call(self,inputs):
    input_shape = tf.shape(inputs)
    assert len(inputs.shape)==3
    batch_size, seq_len, output_dim = inputs.shape

    matrix_repr1 = tf.tile(inputs, [1, seq_len, 1])
    matrix_repr1 = tf.reshape(matrix_repr1, [-1, seq_len, seq_len, output_dim])
    matrix_repr2 = tf.transpose(matrix_repr1, [0,2,1,3])
    current  = tf.concat([matrix_repr1, matrix_repr2], axis=-1)

    return current

class ConcatDist2D(tf.keras.layers.Layer):
  ''' Concatenate the pairwise distance to 2d feature matrix.'''
  def __init__(self):
    super(ConcatDist2D, self).__init__()

  def call(self,inputs):
    input_shape = tf.shape(inputs)
    batch_size, seq_len = input_shape[0], input_shape[1]

    ## concat 2D distance ##
    pos = tf.expand_dims(tf.range(0, seq_len), axis=-1)
    matrix_repr1 = tf.tile(pos, [1,seq_len])
    matrix_repr2 = tf.transpose(matrix_repr1, [1,0])
    dist  = tf.math.abs( tf.math.subtract(matrix_repr1, matrix_repr2) )
    dist = tf.dtypes.cast(dist, tf.float32)
    dist = tf.expand_dims(dist, axis=-1)
    dist = tf.expand_dims(dist, axis=0)
    dist = tf.tile(dist, [batch_size, 1, 1, 1])
    return tf.concat([inputs, dist], axis=-1)

class UpperTri(tf.keras.layers.Layer):
  ''' Unroll matrix to its upper triangular portion.'''
  def __init__(self, diagonal_offset=2):
    super(UpperTri, self).__init__()
    self.diagonal_offset = diagonal_offset

  def call(self, inputs):
    seq_len = inputs.shape[1]
    output_dim = inputs.shape[-1]

    if type(seq_len) == tf.compat.v1.Dimension:
      seq_len = seq_len.value
      output_dim = output_dim.value

    triu_tup = np.triu_indices(seq_len, self.diagonal_offset)
    triu_index = list(triu_tup[0]+ seq_len*triu_tup[1])
    unroll_repr = tf.reshape(inputs, [-1, seq_len**2, output_dim])
    return tf.gather(unroll_repr, triu_index, axis=1)

  def get_config(self):
    config = super().get_config().copy()
    config['diagonal_offset'] = self.diagonal_offset
    return config

class Symmetrize2D(tf.keras.layers.Layer):
  '''Take the average of a matrix and its transpose to enforce symmetry.'''
  def __init__(self):
    super(Symmetrize2D, self).__init__()
  def call(self, x):
    x_t = tf.transpose(x,[0,2,1,3])
    x_sym = (x+x_t)/2
    return x_sym

############################################################
# Augmentation
############################################################

class EnsembleReverseComplement(tf.keras.layers.Layer):
  """Expand tensor to include reverse complement of one hot encoded DNA sequence."""
  def __init__(self):
    super(EnsembleReverseComplement, self).__init__()
  def call(self, seqs_1hot):
    if not isinstance(seqs_1hot, list):
      seqs_1hot = [seqs_1hot]

    ens_seqs_1hot = []
    for seq_1hot in seqs_1hot:
      rc_seq_1hot = tf.gather(seq_1hot, [3, 2, 1, 0], axis=-1)
      rc_seq_1hot = tf.reverse(rc_seq_1hot, axis=[1])
      ens_seqs_1hot += [(seq_1hot, tf.constant(False)), (rc_seq_1hot, tf.constant(True))]

    return ens_seqs_1hot

class StochasticReverseComplement(tf.keras.layers.Layer):
  """Stochastically reverse complement a one hot encoded DNA sequence."""
  def __init__(self):
    super(StochasticReverseComplement, self).__init__()
  def call(self, seq_1hot, training=None):
    if training:
      rc_seq_1hot = tf.gather(seq_1hot, [3, 2, 1, 0], axis=-1)
      rc_seq_1hot = tf.reverse(rc_seq_1hot, axis=[1])
      reverse_bool = tf.random.uniform(shape=[]) > 0.5
      src_seq_1hot = tf.cond(reverse_bool, lambda: rc_seq_1hot, lambda: seq_1hot)
      return src_seq_1hot, reverse_bool
    else:
      return seq_1hot, tf.constant(False)

class SwitchReverse(tf.keras.layers.Layer):
  """Reverse predictions if the inputs were reverse complemented."""
  def __init__(self):
    super(SwitchReverse, self).__init__()
  def call(self, x_reverse):
    x = x_reverse[0]
    reverse = x_reverse[1]

    xd = len(x.shape)
    if xd == 3:
      rev_axes = [1]
    elif xd == 4:
      rev_axes = [1,2]
    else:
      raise ValueError('Cannot recognize SwitchReverse input dimensions %d.' % xd)
    
    return tf.keras.backend.switch(reverse,
                                   tf.reverse(x, axis=rev_axes),
                                   x)

class SwitchReverseTriu(tf.keras.layers.Layer):
  def __init__(self, diagonal_offset):
    super(SwitchReverseTriu, self).__init__()
    self.diagonal_offset = diagonal_offset

  def call(self, x_reverse):
    x_ut = x_reverse[0]
    reverse = x_reverse[1]

    # infer original sequence length
    ut_len = x_ut.shape[1]
    if type(ut_len) == tf.compat.v1.Dimension:
      ut_len = ut_len.value
    seq_len = int(np.sqrt(2*ut_len + 0.25) - 0.5)
    seq_len += self.diagonal_offset

    # get triu indexes
    ut_indexes = np.triu_indices(seq_len, self.diagonal_offset)
    assert(len(ut_indexes[0]) == ut_len)

    # construct a ut matrix of ut indexes
    mat_ut_indexes = np.zeros(shape=(seq_len,seq_len), dtype='int')
    mat_ut_indexes[ut_indexes] = np.arange(ut_len)

    # make lower diag mask
    mask_ut = np.zeros(shape=(seq_len,seq_len), dtype='bool')
    mask_ut[ut_indexes] = True
    mask_ld = ~mask_ut

    # construct a matrix of symmetric ut indexes
    mat_indexes = mat_ut_indexes + np.multiply(mask_ld, mat_ut_indexes.T)

    # reverse complement
    mat_rc_indexes = mat_indexes[::-1,::-1]

    # extract ut order
    rc_ut_order = mat_rc_indexes[ut_indexes]

    return tf.keras.backend.switch(reverse,
                                   tf.gather(x_ut, rc_ut_order, axis=1),
                                   x_ut)
  def get_config(self):
    config = super().get_config().copy()
    config['diagonal_offset'] = self.diagonal_offset
    return config
    
class EnsembleShift(tf.keras.layers.Layer):
  """Expand tensor to include shifts of one hot encoded DNA sequence."""
  def __init__(self, shifts=[0], pad='uniform'):
    super(EnsembleShift, self).__init__()
    self.shifts = shifts
    self.pad = pad

  def call(self, seqs_1hot):
    if not isinstance(seqs_1hot, list):
      seqs_1hot = [seqs_1hot]

    ens_seqs_1hot = []
    for seq_1hot in seqs_1hot:
      for shift in self.shifts:
        ens_seqs_1hot.append(shift_sequence(seq_1hot, shift))

    return ens_seqs_1hot

  def get_config(self):
    config = super().get_config().copy()
    config.update({
      'shifts': self.shifts,
      'pad': self.pad
    })
    return config

class StochasticShift(tf.keras.layers.Layer):
  """Stochastically shift a one hot encoded DNA sequence."""
  def __init__(self, shift_max=0, pad='uniform'):
    super(StochasticShift, self).__init__()
    self.shift_max = shift_max
    self.augment_shifts = tf.range(-self.shift_max, self.shift_max+1)
    self.pad = pad

  def call(self, seq_1hot, training=None):
    if training:
      shift_i = tf.random.uniform(shape=[], minval=0, dtype=tf.int64,
                                  maxval=len(self.augment_shifts))
      shift = tf.gather(self.augment_shifts, shift_i)
      sseq_1hot = tf.cond(tf.not_equal(shift, 0),
                          lambda: shift_sequence(seq_1hot, shift),
                          lambda: seq_1hot)
      return sseq_1hot
    else:
      return seq_1hot

  def get_config(self):
    config = super().get_config().copy()
    config.update({
      'shift_max': self.shift_max,
      'pad': self.pad
    })
    return config

def shift_sequence(seq, shift, pad_value=0.25):
  """Shift a sequence left or right by shift_amount.

  Args:
  seq: [batch_size, seq_length, seq_depth] sequence
  shift: signed shift value (tf.int32 or int)
  pad_value: value to fill the padding (primitive or scalar tf.Tensor)
  """
  if seq.shape.ndims != 3:
      raise ValueError('input sequence should be rank 3')
  input_shape = seq.shape

  pad = pad_value * tf.ones_like(seq[:, 0:tf.abs(shift), :])

  def _shift_right(_seq):
    # shift is positive
    sliced_seq = _seq[:, :-shift:, :]
    return tf.concat([pad, sliced_seq], axis=1)

  def _shift_left(_seq):
    # shift is negative
    sliced_seq = _seq[:, -shift:, :]
    return tf.concat([sliced_seq, pad], axis=1)

  sseq = tf.cond(tf.greater(shift, 0),
                 lambda: _shift_right(seq),
                 lambda: _shift_left(seq))
  sseq.set_shape(input_shape)

  return sseq

############################################################
# Factorization
############################################################

class FactorInverse(tf.keras.layers.Layer):
  """Inverse a target matrix factorization."""
  def __init__(self, components_npy):
    super(FactorInverse, self).__init__()
    self.components_npy = components_npy
    self.components = tf.constant(np.load(components_npy), dtype=tf.float32)

  def call(self, W):
    return tf.keras.backend.dot(W, self.components)

  def get_config(self):
    config = super().get_config().copy()
    config.update({
      'components_npy': self.components_npy
    })
    return config

############################################################
# helpers
############################################################

def activate(current, activation, verbose=False):
  if verbose: print('activate:',activation)
  if activation == 'relu':
    current = tf.keras.layers.ReLU()(current)
  elif activation == 'polyrelu':
    current = PolyReLU()(current)
  elif activation == 'gelu':
    current = GELU()(current)
  elif activation == 'sigmoid':
    current = tf.keras.layers.Activation('sigmoid')(current)
  elif activation == 'tanh':
    current = tf.keras.layers.Activation('tanh')(current)
  elif activation == 'exp':
    current = Exp()(current)
  elif activation == 'softplus':
    current = Softplus()(current)
  else:
    print('Unrecognized activation "%s"' % activation, file=sys.stderr)
    exit(1)

  return current
