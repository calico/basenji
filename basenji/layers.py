"""Wrapper code for using commonly-used layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from basenji import ops


def conv_block(seqs_repr, conv_params, is_training,
               batch_norm, batch_norm_momentum,
               batch_renorm, batch_renorm_momentum,
               nonlinearity, l2_scale, layer_reprs, name=''):
  """Construct a single (dilated) CNN block.

  Args:
    seqs_repr:    [batchsize, length, num_channels] input sequence
    conv_params:  convolution parameters
    is_training:  whether is a training graph or not
    batch_norm:   whether to use batchnorm
    bn_momentum:  batch norm momentum
    batch_renorm: whether to use batch renormalization in batchnorm
    nonlinearity: relu/gelu/etc
    l2_scale:     L2 weight regularization scale
    name:         optional name for the block

  Returns:
    updated representation for the sequence
  """
  # nonlinearity
  if nonlinearity == 'relu':
      seqs_repr_next = tf.nn.relu(seqs_repr)
      tf.logging.info('ReLU')
  elif nonlinearity == 'gelu':
      seqs_repr_next = tf.nn.sigmoid(1.702 * seqs_repr) * seqs_repr
      tf.logging.info('GELU')
  else:
      print('Unrecognized nonlinearity "%s"' % nonlinearity, file=sys.stderr)
      exit(1)

  # Convolution
  seqs_repr_next = tf.layers.conv1d(
      seqs_repr_next,
      filters=conv_params.filters,
      kernel_size=[conv_params.filter_size],
      strides=conv_params.stride,
      padding='same',
      dilation_rate=[conv_params.dilation],
      use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_in'),
      kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_scale))
  tf.logging.info('Convolution w/ %d %dx%d filters strided %d, dilated %d' %
                  (conv_params.filters, seqs_repr.shape[2],
                   conv_params.filter_size, conv_params.stride,
                   conv_params.dilation))

  # Batch norm
  if batch_norm:
    if conv_params.skip_layers > 0:
      gamma_init = tf.zeros_initializer()
    else:
      gamma_init = tf.ones_initializer()

    seqs_repr_next = tf.layers.batch_normalization(
        seqs_repr_next,
        momentum=batch_norm_momentum,
        training=is_training,
        gamma_initializer=gamma_init,
        renorm=batch_renorm,
        renorm_clipping={'rmin': 1./4, 'rmax':4., 'dmax':6.},
        renorm_momentum=batch_renorm_momentum,
        fused=True)
    tf.logging.info('Batch normalization')

  # Dropout
  if conv_params.dropout > 0:
    seqs_repr_next = tf.layers.dropout(
        inputs=seqs_repr_next,
        rate=conv_params.dropout,
        training=is_training)
    tf.logging.info('Dropout w/ probability %.3f' % conv_params.dropout)

  # Skip
  if conv_params.skip_layers > 0:
    if conv_params.skip_layers > len(layer_reprs):
      raise ValueError('Skip connection reaches back too far.')

    # Add
    seqs_repr_next += layer_reprs[-conv_params.skip_layers]

  # Dense
  elif conv_params.dense:
    seqs_repr_next = tf.concat(values=[seqs_repr, seqs_repr_next], axis=2)

  # Pool
  if conv_params.pool > 1:
    seqs_repr_next = tf.layers.max_pooling1d(
        inputs=seqs_repr_next,
        pool_size=conv_params.pool,
        strides=conv_params.pool,
        padding='same')
    tf.logging.info('Max pool %d' % conv_params.pool)

  return seqs_repr_next


def conv2_block(matrix_repr, conv_params, is_training,
                batch_norm, batch_norm_momentum,
                batch_renorm, batch_renorm_momentum,
                nonlinearity, l2_scale, layer_reprs, name=''):
  """Construct a single (dilated) CNN block.

  Args:
    matrix_repr:  [batchsize, length, length, num_channels] input matrix
    conv_params:  convolution parameters
    is_training:  whether is a training graph or not
    batch_norm:   whether to use batchnorm
    bn_momentum:  batch norm momentum
    batch_renorm: whether to use batch renormalization in batchnorm
    nonlinearity: relu/gelu/etc
    l2_scale:     L2 weight regularization scale
    name:         optional name for the block

  Returns:
    updated representation for the sequence
  """
  # nonlinearity
  if nonlinearity == 'relu':
      matrix_repr_next = tf.nn.relu(matrix_repr)
      tf.logging.info('ReLU')
  elif nonlinearity == 'gelu':
      matrix_repr_next = tf.nn.sigmoid(1.702 * matrix_repr) * matrix_repr
      tf.logging.info('GELU')
  else:
      print('Unrecognized nonlinearity "%s"' % nonlinearity, file=sys.stderr)
      exit(1)

  # Convolution
  matrix_repr_next = tf.layers.conv2d(
      matrix_repr_next,
      filters=conv_params.filters,
      kernel_size=[conv_params.filter_size, conv_params.filter_size],
      strides=(conv_params.stride, conv_params.stride),
      padding='same',
      dilation_rate=(conv_params.dilation, conv_params.dilation),
      use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_in'),
      kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_scale))
  tf.logging.info('Convolution w/ %d %dx%d filters, dilated %d' %
                  (conv_params.filters, matrix_repr_next.shape[-1],
                   conv_params.filter_size, conv_params.dilation))

  # Batch norm
  if batch_norm:
    if conv_params.skip_layers > 0:
      gamma_init = tf.zeros_initializer()
    else:
      gamma_init = tf.ones_initializer()

    matrix_repr_next = tf.layers.batch_normalization(
        matrix_repr_next,
        momentum=batch_norm_momentum,
        training=is_training,
        gamma_initializer=gamma_init,
        renorm=batch_renorm,
        renorm_clipping={'rmin': 1./4, 'rmax':4., 'dmax':6.},
        renorm_momentum=batch_renorm_momentum,
        fused=True)
    tf.logging.info('Batch normalization')

  # Dropout
  if conv_params.dropout > 0:
    matrix_repr_next = tf.layers.dropout(
        inputs=matrix_repr_next,
        rate=conv_params.dropout,
        training=is_training)
    tf.logging.info('Dropout w/ probability %.3f' % conv_params.dropout)

  # Skip
  if conv_params.skip_layers > 0:
    if conv_params.skip_layers > len(layer_reprs):
      raise ValueError('Skip connection reaches back too far.')

    # Add
    matrix_repr_next += layer_reprs[-conv_params.skip_layers]

  # # Dense
  # elif conv_params.dense:
  #   seqs_repr_next = tf.concat(values=[seqs_repr, seqs_repr_next], axis=2)

  # Pool
  if conv_params.pool > 1:
    matrix_repr_next = tf.layers.max_pooling2d(
        inputs=matrix_repr_next,
        pool_size=(conv_params.pool, conv_params.pool),
        strides=(conv_params.pool, conv_params.pool),
        padding='same')
    tf.logging.info('Max pool %d' % conv_params.pool)

  return matrix_repr_next
