"""SeqNN blocks"""

import numpy as np
import tensorflow as tf

from basenji import layers

def conv_block(inputs, filters=128, activation='relu', kernel_size=1, strides=1, dilation_rate=1, l2_scale=0, dropout=0, pool_size=1, batch_norm=False, bn_momentum=0.99, bn_gamma='ones'):
  """Construct a single convolution block.

  Args:
    inputs:        [batchsize, length, num_channels] input sequence
    filters:       Conv1D filters
    kernel_size:   Conv1D kernel_size
    activation:    relu/gelu/etc
    strides:       Conv1D strides

  Returns:
    output sequence
  """

  # flow through variable current
  current = inputs

  # activation
  current = layers.activate(current, activation)

  # convolution
  current = tf.keras.layers.Conv1D(
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

  # Pool
  if pool_size > 1:
    current = tf.keras.layers.MaxPool1D(
      pool_size=pool_size,
      padding='same')(current)

  return current

def conv_tower(inputs, filters_init, filters_mult=1, repeat=1, **kwargs):
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
      **kwargs)

    # update filters
    rep_filters *= filters_mult

  return current


def dense(inputs, units, activation='softplus', l2_scale=0, l1_scale=0, **kwargs):
  print('dense activation:',activation)
  current = tf.keras.layers.Dense(
    units=units,
    activation=activation,
    use_bias=True,
    kernel_initializer='he_normal',
    kernel_regularizer=tf.keras.regularizers.l1_l2(l1_scale, l2_scale)
    )(inputs)
  return current


def dilated_dense(inputs, filters, kernel_size=3, rate_mult=2, dropout=0, repeat=1, **kwargs):
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
      **kwargs)

    # dense concat
    current = tf.keras.layers.Concatenate()([rep_input,current])

    # update dilation rate
    dilation_rate *= rate_mult

  return current


def dilated_residual(inputs, filters, kernel_size=3, rate_mult=2, dropout=0, repeat=1, **kwargs):
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





class ConcatPosition(tf.keras.layers.Layer):
  ''' concatenate the distance to the center to a (batch size, sequence length, features) tensor
    via a (batch_size, sequence length, 1) tensor
   '''
  def __init__(self, transform='abs', power=1):
    super(ConcatPosition, self).__init__()
    self.transform = transform
    self.power = power
  def call(self,inputs):
    input_shape = tf.shape(inputs)
    batch_size, seq_len, output_dim = input_shape[0], input_shape[1], 1
    seq_len_float = tf.dtypes.cast(seq_len,dtype=tf.float32)

    print('pos_enc with: ', self.transform)
    import time
    time.sleep(.5)
    if self.transform == 'abs':
      positional_input = tf.math.abs( tf.range(-seq_len_float/2+.5,seq_len_float/2) )
    elif self.transform == 'none':
      positional_input = tf.range(-seq_len_float/2+.5,seq_len_float/2) 
    elif self.transform == 'reversed':
      positional_input = tf.range(-seq_len_float/2+.5,seq_len_float/2) [::-1]
    else:
      raise ValueError('unknown transform')

    print('power with: ', self.power)
    time.sleep(.5)
    if self.power != 1:
      positional_input = tf.pow(positional_input, self.power)
    positional_input = tf.expand_dims(positional_input,axis=0)
    positional_input = tf.expand_dims(positional_input,axis=-1)
    positional_input = tf.tile(positional_input, [batch_size,1, 1]   )
    positional_input = tf.dtypes.cast(positional_input,dtype=tf.float32)
    return tf.concat([positional_input ,  inputs], axis=-1)#-1 ) 

def positional_encoding(inputs, transform='abs', power=1,  **kwargs):
  current = ConcatPosition(transform, power)(inputs)
  return current

def average_pooling(inputs, pool_size=2,**kwargs):
    current = tf.keras.layers.AveragePooling1D(
      pool_size=pool_size,
      padding='same')(inputs)
    return current

name_func = {
  'conv_block': conv_block,
  'conv_tower': conv_tower,
  'dense': dense,
  'dilated_residual': dilated_residual,
  'dilated_dense': dilated_dense,
  'positional_encoding': positional_encoding
  'average_pooling': positional_encoding
}

keras_func = {
  'Conv1D': tf.keras.layers.Conv1D,
  'Dense': tf.keras.layers.Dense
}
