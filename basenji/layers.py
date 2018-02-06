"""Wrapper code for using commonly-used layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from basenji import ops


def renorm_clipping():
  RMAX_START_STEP = 6000
  RMAX_END_STEP = 60000
  RMAX_START_VALUE = 1
  RMAX_END_VALUE = 3
  DMAX_START_STEP = 6000
  DMAX_END_STEP = 60000
  DMAX_START_VALUE = 0
  DMAX_END_VALUE = 5

  RMAX_decay = ops.adjust_max(RMAX_START_STEP, RMAX_END_STEP, RMAX_START_VALUE, RMAX_END_VALUE, name='RMAXDECAY')
  DMAX_decay = ops.adjust_max(DMAX_START_STEP, DMAX_END_STEP, DMAX_START_VALUE, DMAX_END_VALUE, name='DMAXDECAY')
  renorm_clipping = {
      'rmin': 1. / RMAX_decay,
      'rmax': RMAX_decay,
      'dmax': DMAX_decay
  }
  return renorm_clipping


def cnn_block(seqs_repr, cnn_filters, cnn_filter_sizes, cnn_dilation,
              cnn_strides, is_training, batch_norm, bn_momentum, batch_renorm,
              renorm_clipping, cnn_pool, cnn_dropout_value, cnn_dropout_op,
              cnn_dense,name=''):
  """Construct a single (dilated) CNN block.

  Args:
    seqs_repr: [batchsize, length, num_channels] input sequence
    cnn_filters: num filters
    cnn_filter_sizes: size of kernel
    cnn_dilation: dilation factor
    cnn_strides: strides
    is_training: whether is a training graph or not
    batch_norm: whether to use batchnorm
    bn_momentum: batch norm momentum
    batch_renorm: whether to use batch renormalization in batchnorm
    renorm_clipping: clipping used by batch_renorm
    cnn_pool: max pooling factor
    cnn_droput_value: scalar dropout rate
    cnn_dropout_op: dropout Tensor (useful if setting dropout by placeholder)
    cnn_dense: if True, concat outputs to inputs
    name: optional name for the block

  Returns:
    updated representation for the sequence
  """
  seqs_repr_next = tf.layers.conv1d(
      seqs_repr,
      filters=cnn_filters,
      kernel_size=[cnn_filter_sizes],
      strides=cnn_strides,
      padding='same',
      dilation_rate=[cnn_dilation],
      use_bias=False,
      kernel_initializer=tf.contrib.layers.xavier_initializer(),
      kernel_regularizer=None)
  tf.logging.info('Convolution w/ %d %dx%d filters strided %d, dilated %d' %
        (cnn_filters, seqs_repr.shape[2], cnn_filter_sizes, cnn_strides,
         cnn_dilation))

  if batch_norm:
    seqs_repr_next = tf.layers.batch_normalization(
        seqs_repr_next,
        momentum=bn_momentum,
        training=is_training,
        renorm=batch_renorm,
        renorm_clipping=renorm_clipping,
        renorm_momentum=bn_momentum,
        fused=True)
    tf.logging.info('Batch normalization')

  # ReLU
  seqs_repr_next = tf.nn.relu(seqs_repr_next)
  tf.logging.info('ReLU')

  # pooling
  if cnn_pool > 1:
    seqs_repr_next = tf.layers.max_pooling1d(
        seqs_repr_next, pool_size=cnn_pool, strides=cnn_pool, padding='same')
    tf.logging.info('Max pool %d' % cnn_pool)

  # dropout
  if cnn_dropout_value > 0:
    # seqs_repr_next = tf.nn.dropout(seqs_repr_next, 1.0 - cnn_dropout_op)
    seqs_repr_next = tf.layers.dropout(seqs_repr_next, rate=cnn_dropout_value, training=is_training)
    tf.logging.info('Dropout w/ probability %.3f' % cnn_dropout_value)

  if cnn_dense:
    # concat layer repr
    seqs_repr = tf.concat(values=[seqs_repr, seqs_repr_next], axis=2)
  else:
    # update layer repr
    seqs_repr = seqs_repr_next

  return seqs_repr
