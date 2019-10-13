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

class Exp(tf.keras.layers.Layer):
    def __init__(self):
        super(Exp, self).__init__()
    def call(self, x):
      return tf.keras.activations.exponential(x)

class GELU(tf.keras.layers.Layer):
    def __init__(self):
        super(GELU, self).__init__()
    def call(self, x):
        return tf.keras.activations.sigmoid(1.702 * x) * x

class SliceCenter(tf.keras.layers.Layer):
  def __init__(self, left, right=None):
    super(SliceCenter, self).__init__()
    self.left = left
    self.right = right if right is not None else -self.left
  def call(self, x):
    return x[:, self.left:self.right, :]

class Softplus(tf.keras.layers.Layer):
  def __init__(self, exp_max):
    super(Softplus, self).__init__()
    self.exp_max = exp_max
  def call(self, x):
    x = tf.clip_by_value(x, -self.exp_max, 10000)
    return tf.keras.activations.softplus(x)


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
    # seq_len = seq_len.value
    # batch_size  = batch_size.value
    # output_dim = output_dim.value

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
    seq_len = inputs.shape[1].value
    output_dim = inputs.shape[-1]

    triu_tup = np.triu_indices(seq_len, self.diagonal_offset)
    triu_index = list(triu_tup[0]+ seq_len*triu_tup[1])
    unroll_repr = tf.reshape(inputs, [-1, seq_len**2, output_dim])
    return tf.gather(unroll_repr, triu_index, axis=1)

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
  def call(self, seq_1hot, training):
    def stoch_rc():
      rc_seq_1hot = tf.gather(seq_1hot, [3, 2, 1, 0], axis=-1)
      rc_seq_1hot = tf.reverse(rc_seq_1hot, axis=[1])
      reverse_bool = tf.random.uniform(shape=[]) > 0.5
      src_seq_1hot = tf.cond(reverse_bool, lambda: rc_seq_1hot, lambda: seq_1hot)
      return src_seq_1hot, reverse_bool

    return tf.cond(training,
                   stoch_rc,
                   lambda: (seq_1hot, tf.constant(False)))

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
    ut_len = x_ut.shape[1].value
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

class StochasticShift(tf.keras.layers.Layer):
  """Stochastically shift a one hot encoded DNA sequence."""
  def __init__(self, shift_max=0, pad='uniform'):
    super(StochasticShift, self).__init__()
    self.augment_shifts = tf.range(-shift_max, shift_max+1)
    self.pad = pad
  def call(self, seq_1hot, training):
    def stoch_shift():
      shift_i = tf.random.uniform(shape=[], minval=0,
        maxval=len(self.augment_shifts), dtype=tf.int64)
      shift = tf.gather(self.augment_shifts, shift_i)

      sseq_1hot = tf.cond(tf.not_equal(shift, 0),
                          lambda: shift_sequence(seq_1hot, shift),
                          lambda: seq_1hot)
      return sseq_1hot

    return tf.cond(training,
                   stoch_shift,
                   lambda: seq_1hot)

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
# helpers
############################################################

def activate(current, activation, verbose=False):
  if verbose: print('activate:',activation)
  if activation == 'relu':
    current = tf.keras.layers.ReLU()(current)
  elif activation == 'gelu':
    current = GELU()(current)
  elif activation == 'sigmoid':
    current = tf.keras.layers.Activation('sigmoid')(current)
  elif activation == 'tanh':
    current = tf.keras.layers.Activation('tanh')(current)
  elif activation == 'exp':
    current = Exp()(current)
  else:
    print('Unrecognized activation "%s"' % activation, file=sys.stderr)
    exit(1)

  return current
