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
def conv_block(inputs, filters=128, kernel_size=1, activation='relu', 
    conv_type='standard', strides=1, dilation_rate=1, l2_scale=0, dropout=0, 
    pool_size=1, batch_norm=False, bn_momentum=0.99, bn_gamma='ones'):
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
    pool_size:     Max pool width

  Returns:
    output sequence
  """

  # flow through variable current
  current = inputs

  # activation
  current = layers.activate(current, activation)

  # choose convolution type
  if conv_type == 'separable':
    conv_layer = tf.keras.layers.SeparableConv1D
  else:
    conv_layer = tf.keras.layers.Conv1D

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
  # current = tf.keras.layers.LayerNormalization(
  #   gamma_initializer=bn_gamma)(current)

  # dropout
  if dropout > 0:
    current = tf.keras.layers.Dropout(rate=dropout)(current)

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


############################################################
# Towers
############################################################
def conv_tower(inputs, filters_init, filters_mult=1,
    conv_type='standard', repeat=1, **kwargs):
  """Construct a reducing convolution block.

  Args:

  Returns:
    output sequence
  """

  # flow through variable current
  current = inputs

  # initialize filters
  rep_filters = filters_init

  for ri in range(repeat):
    # convolution
    current = conv_block(current,
      filters=int(np.round(rep_filters)),
      conv_type=conv_type,
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
    conv_type='standard', dropout=0, repeat=1, **kwargs):
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
  'slice_center': slice_center
}

keras_func = {
  'Conv1D': tf.keras.layers.Conv1D,
  'Cropping1D': tf.keras.layers.Cropping1D,
  'Cropping2D': tf.keras.layers.Cropping2D,
  'Dense': tf.keras.layers.Dense
}
