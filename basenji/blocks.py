"""SeqNN blocks"""

import numpy as np
import tensorflow as tf

from basenji import layers

def attention(inputs, kq_depth=None, max_relative_position=64, batch_norm=False, bn_momentum=0.99, **kwargs):
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


def conv_block(inputs, filters=128, kernel_size=1, activation='relu', strides=1, dilation_rate=1, l2_scale=0, dropout=0, pool_size=1, batch_norm=False, bn_momentum=0.99, bn_gamma='ones'):
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


def dense(inputs, units, activation='softplus', kernel_initializer='he_normal', l2_scale=0, l1_scale=0, **kwargs):

  print('dense, activation:',activation)
  print('l1_l2',l1_scale,l2_scale)
  print('kernel_initializer:',kernel_initializer)
  current = tf.keras.layers.Dense(
    units=units,
    activation=activation,
    use_bias=True,
    kernel_initializer=kernel_initializer,
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
    return tf.concat([positional_input ,  inputs], axis=-1) # the real thing 
    

def positional_encoding(inputs, transform='abs', power=1,  **kwargs):
  current = ConcatPosition(transform, power)(inputs)
  return current

def average_pooling(inputs, pool_size=2,**kwargs):
    current = tf.keras.layers.AveragePooling1D(
      pool_size=pool_size,
      padding='same')(inputs)
    return current


class ConcatTo2D(tf.keras.layers.Layer):
  def __init__(self):
    super(ConcatTo2D, self).__init__()
  def call(self,inputs):
    input_shape = tf.shape(inputs)
    assert len(inputs.shape)==3
    batch_size, seq_len, output_dim = inputs.shape # input_shape[0], input_shape[1], input_shape[2]
    seq_len = seq_len.value
    batch_size  = batch_size.value
    output_dim = output_dim.value
    matrix_repr1 = tf.tile(inputs, [1, seq_len,1])
    matrix_repr1 = tf.reshape(matrix_repr1, [-1, seq_len, seq_len, output_dim])
    matrix_repr2 = tf.transpose(matrix_repr1, [0,2,1,3])
    current  = tf.concat([matrix_repr1, matrix_repr2], axis= -1)
    return current 

def concat_2D(inputs, **kwargs):
  current = ConcatTo2D()(inputs)
  return current


class ConcatDist2D(tf.keras.layers.Layer):
  ''' concatenate the pairwise distance to a (batch size, sequence length, sequence length, features) tensor
  '''
  def __init__(self):
    super(ConcatDist2D, self).__init__()
  def call(self,inputs):
    input_shape = tf.shape(inputs)
    batch_size, seq_len = input_shape[0], input_shape[1] #.value
    print('making 2D pos_enc of distances')

    ## concat 2D distance ##
    pos = tf.expand_dims(tf.range(0, seq_len), axis=-1)
    matrix_repr1 = tf.tile(pos, [1,seq_len])
    matrix_repr2 = tf.transpose(matrix_repr1, [1,0])
    dist  = tf.math.abs( tf.math.subtract(matrix_repr1, matrix_repr2) )
    dist = tf.dtypes.cast(dist,tf.float32)
    dist = tf.expand_dims(dist,axis=-1)
    dist = tf.expand_dims(dist,axis=0)
    dist = tf.tile(dist, [ batch_size , 1 , 1 , 1])
    return tf.concat([inputs, dist],axis=-1)

def positional_encoding_2D(inputs,   **kwargs):
  current = ConcatDist2D()(inputs)
  return current

class upperTriu2D(tf.keras.layers.Layer):
  ''' squish to upper triangular 
  '''
  def __init__(self):
    super(upperTriu2D, self).__init__()
  def call(self,inputs):
    batch_size, seq_len, output_dim = inputs.shape[0].value, inputs.shape[1].value, inputs.shape[-1].value 
    triu_tup = np.triu_indices(seq_len ,2)
    triu_index = list(triu_tup[0]+ seq_len*triu_tup[1])
    unroll_repr = tf.reshape(inputs, [-1, seq_len**2, output_dim])
    return tf.gather(unroll_repr, triu_index, axis=1)

def upper_triu_2D(inputs,   **kwargs):
  current = upperTriu2D()(inputs)
  return current


def conv_block_2D(inputs, filters=128, activation='relu', kernel_size=1, strides=1, dilation_rate=1, l2_scale=0, dropout=0, pool_size=1, batch_norm=False, bn_momentum=0.99, bn_gamma='ones'):
  """Construct a single 2D convolution block.   """
  print('conv2D: l2_scale',l2_scale)
  # flow through variable current
  current = inputs

  # activation
  current = layers.activate(current, activation)

  # convolution
  current = tf.keras.layers.Conv2D(
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
    current = tf.keras.layers.MaxPool2D(
      pool_size=pool_size,
      padding='same')(current)

  return current


class symmetrize2D(tf.keras.layers.Layer):
  ''' symmetrize 
  '''
  def __init__(self):
    super(symmetrize2D, self).__init__()
  def call(self,inputs):
    return (inputs + tf.transpose(inputs,[0,2,1,3])) / 2

def symmetrize_2D(inputs,**kwargs):
  return symmetrize2D()(inputs)

def symmetric_dilated_residual_2D(inputs, filters, kernel_size=3, rate_mult=2, dropout=0, repeat=1, **kwargs):
  """Construct a residual dilated convolution block.
  """

  # flow through variable current
  current = inputs

  # initialize dilation rate
  dilation_rate = 1.0

  for ri in range(repeat):
    rep_input = current

    # dilate
    current = conv_block_2D(current,
      filters=filters,
      kernel_size=kernel_size,
      dilation_rate=int(np.round(dilation_rate)),
      bn_gamma='ones',
      **kwargs)

    # return
    current = conv_block_2D(current,
      filters=rep_input.shape[-1],
      dropout=dropout,
      bn_gamma='zeros',
      **kwargs)

    # residual add
    current = tf.keras.layers.Add()([rep_input,current])

    # enforce symmetry
    current = symmetrize2D()(current)

    # update dilation rate
    dilation_rate *= rate_mult

  return current



############### experimental zone ############3

def bidirectional_LSTM(inputs, units, useGPU=True, **kwargs):
  if useGPU:
    current = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(units, return_sequences = True))(inputs)
  else:
    current = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences = True))(inputs)
  return current



def separable_conv_block_2D(inputs, filters=128, activation='relu', kernel_size=1, strides=1, dilation_rate=1, depth_multiplier=1,
        l2_scale=0, dropout=0, pool_size=1, batch_norm=False, bn_momentum=0.99, bn_gamma='ones'):
  """Construct a single 2D convolution block.   """
  print('conv2D: l2_scale',l2_scale)
  # flow through variable current
  current = inputs

  # activation
  current = layers.activate(current, activation)

  # convolution
  current = tf.keras.layers.SeparableConv2D(
    filters=filters,
    kernel_size=kernel_size,
    strides=strides,
    padding='same',
    use_bias=False,
    dilation_rate=dilation_rate,
    depth_multiplier = depth_multiplier,
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
    current = tf.keras.layers.MaxPool2D(
      pool_size=pool_size,
      padding='same')(current)

  return current


def symmetric_separable_dilated_residual_2D(inputs, filters, kernel_size=3, rate_mult=2, dropout=0, repeat=1, **kwargs):
  """Construct a residual dilated convolution block.
  """

  # flow through variable current
  current = inputs

  # initialize dilation rate
  dilation_rate = 1.0

  for ri in range(repeat):
    rep_input = current

    # dilate
    current = separable_conv_block_2D(current,
      filters=filters,
      kernel_size=kernel_size,
      dilation_rate=int(np.round(dilation_rate)),
      bn_gamma='ones',
      **kwargs)

    # return
    current = conv_block_2D(current,
      filters=rep_input.shape[-1],
      dropout=dropout,
      bn_gamma='zeros',
      **kwargs)

    # residual add
    current = tf.keras.layers.Add()([rep_input,current])

    # enforce symmetry
    current = symmetrize2D()(current)

    # update dilation rate
    dilation_rate *= rate_mult

  return current



def separable_conv_block(inputs, filters=128, activation='relu', kernel_size=1, strides=1, dilation_rate=1, depth_multiplier=1,
      l2_scale=0, dropout=0, pool_size=1, batch_norm=False, bn_momentum=0.99, bn_gamma='ones'):

  # flow through variable current
  current = inputs

  # activation
  current = layers.activate(current, activation)

  # convolution
  current = tf.keras.layers.SeparableConv1D(
    filters=filters,
    kernel_size=kernel_size,
    strides=strides,
    depth_multiplier=depth_multiplier,
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


def separable_conv_tower(inputs, filters_init, filters_mult=1, repeat=1, **kwargs):
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
    current = separable_conv_block(current,
      filters=int(np.round(rep_filters)),
      **kwargs)

    # update filters
    rep_filters *= filters_mult

  return current


def separable_dilated_residual(inputs, filters, kernel_size=3, rate_mult=2, depth_multiplier=1, dropout=0, repeat=1, **kwargs):
  # flow through variable current
  current = inputs

  # initialize dilation rate
  dilation_rate = 1.0

  for ri in range(repeat):
    rep_input = current

    # dilate
    current = separable_conv_block(current,
      filters=filters,
      kernel_size=kernel_size,
      dilation_rate=int(np.round(dilation_rate)),
      depth_multiplier=1,
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

=======
def position_encoding(current, min_rate=.0001):
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

    # activation
    current = layers.activate(current, activation)

    return current


def slice_center(inputs, center=1, **kwargs):
  crop_len = inputs.shape[1].value - center
  crop_start = crop_len // 2
  crop_end = crop_len - crop_start
  current = inputs
  current = tf.keras.layers.Cropping1D((crop_start,crop_end))(current)
  return current


name_func = {
  'attention': attention,
  'conv_block': conv_block,
  'conv_tower': conv_tower,
  'dense': dense,
  'dilated_residual': dilated_residual,
  'dilated_dense': dilated_dense,
  'positional_encoding': positional_encoding,
  'average_pooling': average_pooling,
  'concat_2D':concat_2D,
  'positional_encoding_2D':positional_encoding_2D,
  'upper_triu_2D': upper_triu_2D,
  'conv_block_2D':conv_block_2D,
  'symmetric_dilated_residual_2D':symmetric_dilated_residual_2D,
  'bidirectional_LSTM':bidirectional_LSTM,
  'symmetrize_2D':symmetrize_2D,
  'separable_dilated_residual':separable_dilated_residual,
  'separable_conv_tower':separable_conv_tower,
  'separable_conv_block':separable_conv_block,
  'symmetric_separable_dilated_residual_2D':symmetric_separable_dilated_residual_2D
  'slice_center': slice_center
}

keras_func = {
  'Conv1D': tf.keras.layers.Conv1D,
  'Dense': tf.keras.layers.Dense
}
