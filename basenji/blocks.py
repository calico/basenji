"""SeqNN blocks"""

import tensorflow as tf

from basenji import layers

def conv_pool(inputs, filters=128, activation='relu', kernel_size=1,
              strides=1, dilation_rate=1, l2_weight=0, momentum=0.99,
              renorm=False, renorm_momentum=0.99, dropout=0, pool_size=1,
              skip_inputs=None, concat=False, is_training=None):
  """Construct a single (dilated) CNN block.

  Args:
    inputs:        [batchsize, length, num_channels] input sequence
    filters:       Conv1D filters
    kernel_size:   Conv1D kernel_size
    activation:    relu/gelu/etc
    strides:       Conv1D strides
    dilation_rate: Conv1D dilation_rate
    l2_weight:
    momentum:
    renorm_momentum:
    skip_inputs:
    is_training:  boolean variable for train/test

  Returns:
    output sequence
  """

  # flow through variable current
  current = inputs

  # activation
  if activation == 'relu':
    current = tf.keras.layers.ReLU()(current)
  elif activation == 'gelu':
    current = layers.GELU()(current)
  else:
    print('Unrecognized activation "%s"' % activation, file=sys.stderr)
    exit(1)

  # convolution
  current = tf.keras.layers.Conv1D(
    filters=filters,
    kernel_size=kernel_size,
    strides=strides,
    padding='same',
    dilation_rate=dilation_rate,
    use_bias=False,
    kernel_initializer='he_normal',
    kernel_regularizer=tf.keras.regularizers.l2(l2_weight))(current)

  # batch norm
  current = tf.keras.layers.BatchNormalization(
    momentum=momentum,
    gamma_initializer=('ones' if skip_inputs is None else 'zeros'),
    renorm=renorm,
    renorm_clipping={'rmin': 1./4, 'rmax':4., 'dmax':6.},
    renorm_momentum=renorm_momentum,
    fused=True)(current, training=is_training)

  # dropout
  if dropout > 0:
    current = tf.keras.layers.Dropout(rate=dropout)(current, training=is_training)

   # skip
  if skip_inputs is not None:
    current = tf.keras.layers.Add()([skip_inputs,current])

  # concat
  elif concat:
    current = tf.keras.layers.Concatenate()([inputs,current])

  # Pool
  if pool_size > 1:
    current = tf.keras.layers.MaxPool1D(
      pool_size=pool_size,
      padding='same')(current)

  return current
