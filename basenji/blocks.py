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
import numpy as np
import tensorflow as tf

from basenji import layers

############################################################
# Convolution
############################################################
def conv_block(inputs, filters=None, kernel_size=1, activation='relu', activation_end=None,
    strides=1, dilation_rate=1, l2_scale=0, dropout=0, conv_type='standard', residual=False,
    pool_size=1, batch_norm=False, bn_momentum=0.99, bn_gamma=None, bn_type='standard',
    kernel_initializer='he_normal', padding='same'):
  """Construct a single convolution block.

  Args:
    inputs:        [batch_size, seq_length, features] input sequence
    filters:       Conv1D filters
    kernel_size:   Conv1D kernel_size
    activation:    relu/gelu/etc
    strides:       Conv1D strides
    dilation_rate: Conv1D dilation rate
    l2_scale:      L2 regularization weight.
    dropout:       Dropout rate probability
    conv_type:     Conv1D layer type
    residual:      Residual connection boolean
    pool_size:     Max pool width
    batch_norm:    Apply batch normalization
    bn_momentum:   BatchNorm momentum
    bn_gamma:      BatchNorm gamma (defaults according to residual)

  Returns:
    [batch_size, seq_length, features] output sequence
  """

  # flow through variable current
  current = inputs

  # choose convolution type
  if conv_type == 'separable':
    conv_layer = tf.keras.layers.SeparableConv1D
  else:
    conv_layer = tf.keras.layers.Conv1D

  if filters is None:
    filters = inputs.shape[-1]

  # activation
  current = layers.activate(current, activation)

  # convolution
  current = conv_layer(
    filters=filters,
    kernel_size=kernel_size,
    strides=strides,
    padding='same',
    use_bias=False,
    dilation_rate=dilation_rate,
    kernel_initializer=kernel_initializer,
    kernel_regularizer=tf.keras.regularizers.l2(l2_scale))(current)

  # batch norm
  if batch_norm:
    if bn_gamma is None:
      bn_gamma = 'zeros' if residual else 'ones'
    if bn_type == 'sync':
      bn_layer = tf.keras.layers.experimental.SyncBatchNormalization
    else:
      bn_layer = tf.keras.layers.BatchNormalization
    current = bn_layer(
      momentum=bn_momentum,
      gamma_initializer=bn_gamma)(current)

  # dropout
  if dropout > 0:
    current = tf.keras.layers.Dropout(rate=dropout)(current)

  # residual add
  if residual:
    current = tf.keras.layers.Add()([inputs,current])

  # end activation
  if activation_end is not None:
    current = layers.activate(current, activation_end)
    
  # Pool
  if pool_size > 1:
    current = tf.keras.layers.MaxPool1D(
      pool_size=pool_size,
      padding=padding)(current)

  return current


def conv_block_2d(inputs, filters=128, activation='relu', conv_type='standard', 
    kernel_size=1, strides=1, dilation_rate=1, l2_scale=0, dropout=0, pool_size=1,
    batch_norm=False, bn_momentum=0.99, bn_gamma='ones', bn_type='standard', symmetric=False):
  """Construct a single 2D convolution block.   """

  # flow through variable current
  current = inputs

  # activation
  current = layers.activate(current, activation)

  # choose convolution type
  if conv_type == 'separable':
    conv_layer = tf.keras.layers.SeparableConv2D
  else:
    conv_layer = tf.keras.layers.Conv2D

  # convolution
  current = conv_layer(
    filters=filters,
    kernel_size=kernel_size,
    strides=strides,
    padding='same',
    use_bias=False,
    dilation_rate=dilation_rate,
    kernel_initializer='he_normal',
    kernel_regularizer=tf.keras.regularizers.l2(l2_scale))(current)

  # batch norm
  if batch_norm:
    if bn_type == 'sync':
      bn_layer = tf.keras.layers.experimental.SyncBatchNormalization
    else:
      bn_layer = tf.keras.layers.BatchNormalization
    current = bn_layer(
      momentum=bn_momentum,
      gamma_initializer=bn_gamma)(current)

  # dropout
  if dropout > 0:
    current = tf.keras.layers.Dropout(rate=dropout)(current)

  # pool
  if pool_size > 1:
    current = tf.keras.layers.MaxPool2D(
      pool_size=pool_size,
      padding='same')(current)

  # symmetric
  if symmetric:
    current = layers.Symmetrize2D()(current)

  return current


def xception_block(inputs, filters=None, kernel_size=1,
  dropout=0, pool_size=2, **kwargs):
  """Construct a single convolution block.

  Args:
    inputs:        [batch_size, seq_length, features] input sequence
    filters:       Conv1D filters
    kernel_size:   Conv1D kernel_size
    dropout:       Dropout rate probability
    pool_size:     Pool/stride width

  Returns:
    [batch_size, seq_length, features] output sequence
  """

  # flow through variable current
  current = inputs

  if filters is None:
    filters = inputs.shape[-1]

  # strided convolution
  current_stride = conv_block(current,
    filters=filters,
    kernel_size=pool_size,
    strides=pool_size,
    dropout=0,
    kernel_initializer='ones',
    **kwargs)

  # pooled convolution
  current_pool = current
  for ci in range(2):
    current_pool = conv_block(current_pool,
      filters=filters,
      kernel_size=kernel_size,
      conv_type='separable',
      dropout=dropout,
      **kwargs)

  # should the last conv_block be set to bn_gamma='zeros'?
  # I don't think so since we really need that new information

  # max pool
  current_pool = tf.keras.layers.MaxPool1D(
    pool_size=int(1.5*pool_size),
    strides=pool_size,
    padding='same')(current_pool)

  # residual add
  current = tf.keras.layers.Add()([current_stride,current_pool])

  return current


############################################################
# Towers
############################################################
def conv_tower(inputs, filters_init, filters_mult=1, repeat=1, **kwargs):
  """Construct a reducing convolution block.

  Args:
    inputs:        [batch_size, seq_length, features] input sequence
    filters_init:  Initial Conv1D filters
    filters_mult:  Multiplier for Conv1D filters
    repeat:        Conv block repetitions

  Returns:
    [batch_size, seq_length, features] output sequence
  """

  # flow through variable current
  current = inputs

  # initialize filters
  rep_filters = filters_init

  for ri in range(repeat):
    # convolution
    current = conv_block(current,
      filters=int(np.round(rep_filters)),
      **kwargs)

    # update filters
    rep_filters *= filters_mult

  return current


def res_tower(inputs, filters_init, filters_mult=1, dropout=0,
              pool_size=2, repeat=1, num_convs=2, **kwargs):
  """Construct a reducing convolution block.

  Args:
    inputs:        [batch_size, seq_length, features] input sequence
    filters_init:  Initial Conv1D filters
    filters_mult:  Multiplier for Conv1D filters
    dropout:       Dropout on subsequent convolution blocks.
    repeat:        Residual block repetitions
    num_convs:     Conv blocks per residual layer

  Returns:
    [batch_size, seq_length, features] output sequence
  """

  # flow through variable current
  current = inputs

  # initialize filters
  rep_filters = filters_init

  for ri in range(repeat):
    rep_filters_int = int(np.round(rep_filters))

    # initial
    current = conv_block(current,
      filters=rep_filters_int,
      dropout=0,
      bn_gamma='ones',
      **kwargs)
    current0 = current

    # subsequent
    for ci in range(1,num_convs):
      bg = 'ones' if ci < num_convs-1 else 'zeros'
      current = conv_block(current,
                           filters=rep_filters_int,
                           dropout=dropout,
                           bn_gamma=bg,
                           **kwargs)

    # residual add
    current = tf.keras.layers.Add()([current0,current])

    # pool
    if pool_size > 1:
      current = tf.keras.layers.MaxPool1D(
        pool_size=pool_size,
        padding='same')(current)

    # update filters
    rep_filters *= filters_mult

  return current


def xception_tower(inputs, filters_init, filters_mult=1, repeat=1, **kwargs):
  """Construct a reducing convolution block.

  Args:
    inputs:        [batch_size, seq_length, features] input sequence
    filters_init:  Initial Conv1D filters
    filters_mult:  Multiplier for Conv1D filters
    repeat:        Conv block repetitions

  Returns:
    [batch_size, seq_length, features] output sequence
  """

  # flow through variable current
  current = inputs

  # initialize filters
  rep_filters = filters_init

  for ri in range(repeat):
    # convolution
    current = xception_block(current,
      filters=int(np.round(rep_filters)),
      **kwargs)

    # update filters
    rep_filters *= filters_mult

  return current


############################################################
# Attention
############################################################
def multihead_attention(inputs, key_size=None, heads=1, out_size=None,
    num_position_features=None, activation='relu', bn_momentum=0.9,
    attention_dropout=0, position_dropout=0, dropout=0, dense_expansion=0, **kwargs):
  if out_size is None:
    out_size = inputs.shape[-1]
    value_size = out_size // heads

  # activation 
  current = layers.activate(inputs, activation)
    
  # layer norm
  current = tf.keras.layers.LayerNormalization()(current)

  # multi-head attention
  current = layers.MultiheadAttention(value_size=value_size,
    key_size=key_size,
    heads=heads,
    num_position_features=num_position_features,
    attention_dropout_rate=attention_dropout,
    positional_dropout_rate=position_dropout,
    zero_initialize=False)(current)

  # batch norm
  current = tf.keras.layers.BatchNormalization(
    momentum=bn_momentum,
    gamma_initializer='zeros')(current)

  # dropout
  if dropout > 0:
    current = tf.keras.layers.Dropout(dropout)(current)

  # residual
  current = tf.keras.layers.Add()([inputs,current])

  if dense_expansion == 0:
    final = current
  else:
    current_mha = current

    # layer norm
    current = tf.keras.layers.LayerNormalization()(current)

    # dense
    expansion_filters = int(dense_expansion*out_size)
    current = tf.keras.layers.Dense(expansion_filters)(current)

    # dropout
    if dropout > 0:
      current = tf.keras.layers.Dropout(dropout)(current)

    # activation 
    current = layers.activate(current, activation)

    # dense
    current = tf.keras.layers.Dense(out_size)(current)

    # dropout
    if dropout > 0:
      current = tf.keras.layers.Dropout(dropout)(current)

    # residual
    final = tf.keras.layers.Add()([current_mha,current])

  return current


def attention(inputs, kq_depth=None, max_relative_position=64,
    batch_norm=False, bn_momentum=0.99, bn_type='standard', **kwargs):
  """Construct a residual attention block.

  Args:
    inputs:                 [batch_size, seq_length, features] input sequence
    kq_depth:               Key-query feature depth
    max_relative_position:  Max relative position to differentiate w/ its own parameter

  Returns:
    output sequence
  """

  # flow through variable current
  current = inputs

  current_depth = current.shape[-1]
  if kq_depth is None:
    kq_depth = current_depth

  # key - who am I?
  key = tf.keras.layers.Conv1D(
    filters=kq_depth,
    kernel_size=1,
    padding='same',
    kernel_initializer='he_normal'
  )(current)

  # query - what am I looking for?
  query = tf.keras.layers.Conv1D(
    filters=kq_depth,
    kernel_size=1,
    padding='same',
    kernel_initializer='he_normal'
  )(current)

  # value - what do I have to say?
  value = tf.keras.layers.Conv1D(
    filters=current_depth,
    kernel_size=1,
    padding='same',
    kernel_initializer='he_normal',
  )(current)

  # apply layer
  z = layers.Attention(max_relative_position=max_relative_position)([query,value,key])

  # batch norm
  if batch_norm:
    if bn_type == 'sync':
      bn_layer = tf.keras.layers.experimental.SyncBatchNormalization
    else:
      bn_layer = tf.keras.layers.BatchNormalization
    z = bn_layer(
      momentum=bn_momentum,
      gamma_initializer='zeros')(z)

  # residual add
  current = tf.keras.layers.Add()([current,z])

  return current

def position_encoding(current, min_rate=.0001):
  """Add original Transformer positional encodings,

  Args:
    current:  [batch_size, seq_length, features] sequence
    min_rate:

  Returns:
    sequence w/ positional encodings concatenated.
  """
  seq_length = current.shape[1].value
  features = current.shape[2].value

  assert(features % 2 == 0)

  # compute angle rates
  angle_rate_exponents = np.linspace(0, 1, features//2)
  angle_rates = min_rate**angle_rate_exponents

  # compute angle radians
  positions = np.range(seq_length)
  angle_rads = positions[:, np.newaxis] * angle_rates[np.newaxis, :]

  # sines and cosines
  sines = np.sin(angle_rads)
  cosines = np.cos(angle_rads)
  pos_encode = np.concatenate([sines, cosines], axis=-1)

  return current

def squeeze_excite(inputs, activation='relu', bottleneck_ratio=8,
     additive=False, batch_norm=False, bn_momentum=0.9, **kwargs):
  return layers.SqueezeExcite(activation, additive, bottleneck_ratio,
    batch_norm, bn_momentum)(inputs)

def wheeze_excite(inputs, pool_size, **kwargs):
  return layers.WheezeExcite(pool_size)(inputs)

def global_context(inputs, **kwargs):
  return layers.GlobalContext()(inputs)

############################################################
# Dilated Towers
############################################################

def dilated_dense(inputs, filters, kernel_size=3, rate_mult=2,
    conv_type='standard', dropout=0, repeat=1, **kwargs):
  """Construct a residual dilated dense block.

  Args:

  Returns:
  """

  # flow through variable current
  current = inputs

  # initialize dilation rate
  dilation_rate = 1.0

  for ri in range(repeat):
    rep_input = current

    # dilate
    current = conv_block(current,
      filters=filters,
      kernel_size=kernel_size,
      dilation_rate=int(np.round(dilation_rate)),
      conv_type=conv_type,
      **kwargs)

    # dense concat
    current = tf.keras.layers.Concatenate()([rep_input,current])

    # update dilation rate
    dilation_rate *= rate_mult

  return current


def dilated_residual(inputs, filters, kernel_size=3, rate_mult=2, 
    conv_type='standard', dropout=0, repeat=1, round=False, **kwargs):
  """Construct a residual dilated convolution block.

  Args:

  Returns:
  """

  # flow through variable current
  current = inputs

  # initialize dilation rate
  dilation_rate = 1.0

  for ri in range(repeat):
    rep_input = current

    # dilate
    current = conv_block(current,
      filters=filters,
      kernel_size=kernel_size,
      dilation_rate=int(np.round(dilation_rate)),
      conv_type=conv_type,
      bn_gamma='ones',
      **kwargs)

    # return
    current = conv_block(current,
      filters=rep_input.shape[-1],
      dropout=dropout,
      bn_gamma='zeros',
      **kwargs)

    # residual add
    current = tf.keras.layers.Add()([rep_input,current])

    # update dilation rate
    dilation_rate *= rate_mult
    if round:
      dilation_rate = np.round(dilation_rate)

  return current

def dilated_residual_2d(inputs, filters, kernel_size=3, rate_mult=2,
    dropout=0, repeat=1, symmetric=True, **kwargs):
  """Construct a residual dilated convolution block.
  """

  # flow through variable current
  current = inputs

  # initialize dilation rate
  dilation_rate = 1.0

  for ri in range(repeat):
    rep_input = current

    # dilate
    current = conv_block_2d(current,
      filters=filters,
      kernel_size=kernel_size,
      dilation_rate=int(np.round(dilation_rate)),
      bn_gamma='ones',
      **kwargs)

    # return
    current = conv_block_2d(current,
      filters=rep_input.shape[-1],
      dropout=dropout,
      bn_gamma='zeros',
      **kwargs)

    # residual add
    current = tf.keras.layers.Add()([rep_input,current])

    # enforce symmetry
    if symmetric:
      current = layers.Symmetrize2D()(current)

    # update dilation rate
    dilation_rate *= rate_mult

  return current

############################################################
# Activations
############################################################

def exp(inputs, base=None, minus=None, **kwargs):
  current = layers.Exp(base, minus)(inputs)
  return current

############################################################
# Center ops
############################################################

def center_average(inputs, center, **kwargs):
  current = layers.CenterAverage(center)(inputs)
  return current

def center_slice(inputs, center, **kwargs):
  current = layers.CenterSlice(center)(inputs)
  return current

############################################################
# 2D
############################################################

def concat_dist_2d(inputs, **kwargs):
  current = layers.ConcatDist2D()(inputs)
  return current

def concat_position(inputs, transform='abs', power=1, **kwargs):
  current = layers.ConcatPosition(transform, power)(inputs)
  return current

def cropping_2d(inputs, cropping, **kwargs):
  current = tf.keras.layers.Cropping2D(cropping)(inputs)
  return current

def one_to_two(inputs, operation='mean', **kwargs):
  current = layers.OneToTwo(operation)(inputs)
  return current

def symmetrize_2d(inputs, **kwargs):
  return layers.Symmetrize2D()(inputs)

def upper_tri(inputs, diagonal_offset=2, **kwargs):
  current = layers.UpperTri(diagonal_offset)(inputs)
  return current

############################################################
# Factorization
############################################################

def factor_inverse(inputs, components_file, **kwargs):
  current = layers.FactorInverse(components_file)(inputs)
  return current

############################################################
# Dense
############################################################
def dense_block(inputs, units=None, activation='relu', activation_end=None,
    flatten=False, dropout=0, l2_scale=0, l1_scale=0, residual=False,
    batch_norm=False, bn_momentum=0.99, bn_gamma=None, bn_type='standard',
    kernel_initializer='he_normal', **kwargs):
  """Construct a single convolution block.

  Args:
    inputs:         [batch_size, seq_length, features] input sequence
    units:          Conv1D filters
    activation:     relu/gelu/etc
    activation_end: Compute activation after the other operations
    flatten:        Flatten across positional axis
    dropout:        Dropout rate probability
    l2_scale:       L2 regularization weight.
    l1_scale:       L1 regularization weight.
    residual:       Residual connection boolean
    batch_norm:     Apply batch normalization
    bn_momentum:    BatchNorm momentum
    bn_gamma:       BatchNorm gamma (defaults according to residual)

  Returns:
    [batch_size, seq_length(?), features] output sequence
  """
  current = inputs

  if units is None:
    units = inputs.shape[-1]

  # activation
  current = layers.activate(current, activation)

  # flatten
  if flatten:
    _, seq_len, seq_depth = current.shape
    current = tf.keras.layers.Reshape((1,seq_len*seq_depth,))(current)

  # dense
  current = tf.keras.layers.Dense(
    units=units,
    use_bias=(not batch_norm),
    kernel_initializer=kernel_initializer,
    kernel_regularizer=tf.keras.regularizers.l1_l2(l1_scale, l2_scale)
    )(current)

  # batch norm
  if batch_norm:
    if bn_gamma is None:
      bn_gamma = 'zeros' if residual else 'ones'
    if bn_type == 'sync':
      bn_layer = tf.keras.layers.experimental.SyncBatchNormalization
    else:
      bn_layer = tf.keras.layers.BatchNormalization
    current = bn_layer(
      momentum=bn_momentum,
      gamma_initializer=bn_gamma)(current)

  # dropout
  if dropout > 0:
    current = tf.keras.layers.Dropout(rate=dropout)(current)

  # residual add
  if residual:
    current = tf.keras.layers.Add()([inputs,current])

  # end activation
  if activation_end is not None:
    current = layers.activate(current, activation_end)

  return current


def final(inputs, units, activation='linear', flatten=False,
          kernel_initializer='he_normal', l2_scale=0, l1_scale=0, **kwargs):
  """Final simple transformation before comparison to targets.

  Args:
    inputs:         [batch_size, seq_length, features] input sequence
    units:          Dense units
    activation:     relu/gelu/etc
    flatten:        Flatten positional axis.
    l2_scale:       L2 regularization weight.
    l1_scale:       L1 regularization weight.

  Returns:
    [batch_size, seq_length(?), units] output sequence

  """
  current = inputs

  # flatten
  if flatten:
    _, seq_len, seq_depth = current.shape
    current = tf.keras.layers.Reshape((1,seq_len*seq_depth,))(current)

  # dense
  current = tf.keras.layers.Dense(
    units=units,
    use_bias=True,
    activation=activation,
    kernel_initializer=kernel_initializer,
    kernel_regularizer=tf.keras.regularizers.l1_l2(l1_scale, l2_scale)
    )(current)

  return current

# depracated, poorly named
def dense(inputs, units, activation='linear', kernel_initializer='he_normal',
          l2_scale=0, l1_scale=0, **kwargs):

  # apply dense layer
  current = tf.keras.layers.Dense(
    units=units,
    use_bias=True,
    activation=activation,
    kernel_initializer=kernel_initializer,
    kernel_regularizer=tf.keras.regularizers.l1_l2(l1_scale, l2_scale)
    )(inputs)

  return current


############################################################
# Depracated
############################################################

# depracated: use AveragePooling1D
def average_pooling(inputs, pool_size=2, **kwargs):
  current = tf.keras.layers.AveragePooling1D(
    pool_size=pool_size,
    padding='same')(inputs)
  return current

# depracated: use one_to_two
def average_to_2d(inputs, **kwargs):
  current = layers.AverageTo2D()(inputs)
  return current

# depracated: use one_to_two
def max_to_2d(inputs, **kwargs):
  current = layers.MaxTo2D()(inputs)
  return current

# depracated: use one_to_two
def dot_to_2d(inputs, **kwargs):
  current = layers.DotTo2D()(inputs)
  return current

# depracated: use one_to_two
def geodot_to_2d(inputs, **kwargs):
  current = layers.GeoDotTo2D()(inputs)
  return current

# depracated: use one_to_two
def concat_to_2d(inputs, **kwargs):
  current = layers.ConcatTo2D()(inputs)
  return current

# depracated: use cropping
# def slice_center(inputs, center=1, **kwargs):
#   crop_len = inputs.shape[1] - center
#   crop_start = crop_len // 2
#   crop_end = crop_len - crop_start
#   current = inputs
#   current = tf.keras.layers.Cropping1D((crop_start,crop_end))(current)
#   return current


############################################################
# Dictionary
############################################################
name_func = {
  'attention': attention,
  'center_slice': center_slice,
  'center_average': center_average,
  'concat_dist_2d': concat_dist_2d,
  'concat_position': concat_position,
  'concat_to_2d': concat_to_2d,
  'conv_block': conv_block,  
  'conv_block_2d': conv_block_2d,
  'conv_tower': conv_tower,
  'cropping_2d': cropping_2d,
  'dense': dense,
  'dense_block': dense_block,
  'dilated_residual': dilated_residual,
  'dilated_residual_2d': dilated_residual_2d,
  'dilated_dense': dilated_dense,
  'exp': exp,
  'factor_inverse': factor_inverse,
  'final': final,
  'global_context': global_context,
  'multihead_attention': multihead_attention,
  'one_to_two': one_to_two,
  'symmetrize_2d':symmetrize_2d,
  'squeeze_excite': squeeze_excite,
  'res_tower': res_tower,
  'upper_tri': upper_tri,
  'wheeze_excite': wheeze_excite,
  'xception_block': xception_block,
  'xception_tower': xception_tower,
}

keras_func = {
  'Conv1D': tf.keras.layers.Conv1D,
  'Cropping1D': tf.keras.layers.Cropping1D,
  'Cropping2D': tf.keras.layers.Cropping2D,
  'Dense': tf.keras.layers.Dense,
  'Flatten': tf.keras.layers.Flatten
}
