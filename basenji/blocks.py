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
def conv_block(inputs, filters=None, kernel_size=1, activation='relu', strides=1,
    dilation_rate=1, l2_scale=0, dropout=0, conv_type='standard', residual=False,
    pool_size=1, batch_norm=False, bn_momentum=0.99, bn_gamma=None,
    kernel_initializer='he_normal'):
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
    current = tf.keras.layers.BatchNormalization(
      momentum=bn_momentum,
      gamma_initializer=bn_gamma,
      fused=True)(current)

  # dropout
  if dropout > 0:
    current = tf.keras.layers.Dropout(rate=dropout)(current)

  # residual add
  if residual:
    current = tf.keras.layers.Add()([inputs,current])
    
  # Pool
  if pool_size > 1:
    current = tf.keras.layers.MaxPool1D(
      pool_size=pool_size,
      padding='same')(current)

  return current


def conv_block_2d(inputs, filters=128, activation='relu', conv_type='standard', 
    kernel_size=1, strides=1, dilation_rate=1, l2_scale=0, dropout=0, pool_size=1,
    batch_norm=False, bn_momentum=0.99, bn_gamma='ones', symmetric=False):
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
    current = tf.keras.layers.BatchNormalization(
      momentum=bn_momentum,
      gamma_initializer=bn_gamma,
      fused=True)(current)

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
def attention(inputs, kq_depth=None, max_relative_position=64,
    batch_norm=False, bn_momentum=0.99, **kwargs):
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
    z = tf.keras.layers.BatchNormalization(
      momentum=bn_momentum,
      gamma_initializer='zeros',
      fused=True)(z)

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

def squeeze_excite(inputs, **kwargs):
  return layers.SqueezeExcite()(inputs)

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
# Keras defaults
############################################################
def dense(inputs, units, activation='softplus', kernel_initializer='he_normal',
    l2_scale=0, l1_scale=0, **kwargs):
  current = tf.keras.layers.Dense(
    units=units,
    activation=activation,
    use_bias=True,
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

# not sure
def slice_center(inputs, center=1, **kwargs):
  crop_len = inputs.shape[1].value - center
  crop_start = crop_len // 2
  crop_end = crop_len - crop_start
  current = inputs
  current = tf.keras.layers.Cropping1D((crop_start,crop_end))(current)
  return current


############################################################
# Dictionary
############################################################
name_func = {
  'attention': attention,
  'conv_block': conv_block,  
  'conv_tower': conv_tower,
  'res_tower': res_tower,
  'xception_block': xception_block,
  'xception_tower': xception_tower,
  'cropping_2d': cropping_2d,
  'dense': dense,
  'dilated_residual': dilated_residual,
  'dilated_dense': dilated_dense,
  'average_pooling': average_pooling,
  'one_to_two': one_to_two,
  'concat_position': concat_position,
  'concat_to_2d': concat_to_2d,
  'average_to_2d': average_to_2d,
  'max_to_2d': max_to_2d,
  'dot_to_2d': dot_to_2d,
  'geodot_to_2d': geodot_to_2d,
  'concat_dist_2d': concat_dist_2d,
  'upper_tri': upper_tri,
  'conv_block_2d': conv_block_2d,
  'dilated_residual_2d': dilated_residual_2d,
  'symmetrize_2d':symmetrize_2d,
  'slice_center': slice_center,
  'squeeze_excite': squeeze_excite,
  'wheeze_excite': wheeze_excite,
  'global_context': global_context
}

keras_func = {
  'Conv1D': tf.keras.layers.Conv1D,
  'Cropping1D': tf.keras.layers.Cropping1D,
  'Cropping2D': tf.keras.layers.Cropping2D,
  'Dense': tf.keras.layers.Dense
}
