"""SeqNN regression metrics."""

import numpy as np
import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops, math_ops
from tensorflow.python.keras import backend as K


class PearsonR(tf.keras.metrics.Metric):
  def __init__(self, num_targets, name='pearsonr', **kwargs):
    super(PearsonR, self).__init__(name=name, **kwargs)
    self._shape = (num_targets,)
    self._count = self.add_weight(name='count', shape=self._shape, initializer='zeros')

    self._product = self.add_weight(name='product', shape=self._shape, initializer='zeros')
    self._true_sum = self.add_weight(name='true_sum', shape=self._shape, initializer='zeros')
    self._true_sumsq = self.add_weight(name='true_sumsq', shape=self._shape, initializer='zeros')
    self._pred_sum = self.add_weight(name='pred_sum', shape=self._shape, initializer='zeros')
    self._pred_sumsq = self.add_weight(name='pred_sumsq', shape=self._shape, initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.cast(y_true, 'float32')
    y_pred = tf.cast(y_pred, 'float32')

    product = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=[0,1])
    self._product.assign_add(product)

    true_sum = tf.reduce_sum(y_true, axis=[0,1])
    self._true_sum.assign_add(true_sum)

    true_sumsq = tf.reduce_sum(tf.math.square(y_true), axis=[0,1])
    self._true_sumsq.assign_add(true_sumsq)

    pred_sum = tf.reduce_sum(y_pred, axis=[0,1])
    self._pred_sum.assign_add(pred_sum)

    pred_sumsq = tf.reduce_sum(tf.math.square(y_pred), axis=[0,1])
    self._pred_sumsq.assign_add(pred_sumsq)

    count = array_ops.ones_like(y_true)
    count = tf.reduce_sum(count, axis=[0,1])
    self._count.assign_add(count)

  def result(self):
    true_mean = tf.divide(self._true_sum, self._count)
    true_mean2 = tf.math.square(true_mean)
    pred_mean = tf.divide(self._pred_sum, self._count)
    pred_mean2 = tf.math.square(pred_mean)

    term1 = self._product
    term2 = -tf.multiply(true_mean, self._pred_sum)
    term3 = -tf.multiply(pred_mean, self._true_sum)
    term4 = tf.multiply(self._count, tf.multiply(true_mean, pred_mean))
    covariance = term1 + term2 + term3 + term4

    true_var = self._true_sumsq - tf.multiply(self._count, true_mean2)
    pred_var = self._pred_sumsq - tf.multiply(self._count, pred_mean2)
    tp_var = tf.multiply(tf.math.sqrt(true_var), tf.math.sqrt(pred_var))
    correlation = tf.divide(covariance, tp_var)

    return tf.reduce_mean(correlation)

  def reset_states(self):
      K.batch_set_value([(v, np.zeros(self._shape)) for v in self.variables])


class R2(tf.keras.metrics.Metric):
  def __init__(self, num_targets, name='r2', **kwargs):
    super(R2, self).__init__(name=name, **kwargs)
    self._shape = (num_targets,)
    self._count = self.add_weight(name='count', shape=self._shape, initializer='zeros')

    self._true_sum = self.add_weight(name='true_sum', shape=self._shape, initializer='zeros')
    self._true_sumsq = self.add_weight(name='true_sumsq', shape=self._shape, initializer='zeros')

    self._product = self.add_weight(name='product', shape=self._shape, initializer='zeros')
    self._pred_sumsq = self.add_weight(name='pred_sumsq', shape=self._shape, initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.cast(y_true, 'float32')
    y_pred = tf.cast(y_pred, 'float32')

    true_sum = tf.reduce_sum(y_true, axis=[0,1])
    self._true_sum.assign_add(true_sum)

    true_sumsq = tf.reduce_sum(tf.math.square(y_true), axis=[0,1])
    self._true_sumsq.assign_add(true_sumsq)

    product = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=[0,1])
    self._product.assign_add(product)

    pred_sumsq = tf.reduce_sum(tf.math.square(y_pred), axis=[0,1])
    self._pred_sumsq.assign_add(pred_sumsq)

    count = array_ops.ones_like(y_true)
    count = tf.reduce_sum(count, axis=[0,1])
    self._count.assign_add(count)

  def result(self):
    true_mean = tf.divide(self._true_sum, self._count)
    true_mean2 = tf.math.square(true_mean)

    total = self._true_sumsq - tf.multiply(self._count, true_mean2)

    resid1 = self._pred_sumsq
    resid2 = -2*self._product
    resid3 = self._true_sumsq
    resid = resid1 + resid2 + resid3

    r2 = tf.ones_like(self._shape, dtype=tf.float32) - tf.divide(resid, total)

    return tf.reduce_mean(r2)

  def reset_states(self):
    K.batch_set_value([(v, np.zeros(self._shape)) for v in self.variables])

############################################################
# Augmentation
############################################################

class StochasticReverseComplement(tf.keras.layers.Layer):
  def __init__(self):
    super(StochasticReverseComplement, self).__init__()
  def call(self, seq_1hot):
    """Stochastically reverse complement a one hot encoded DNA sequence."""
    rc_seq_1hot = tf.gather(seq_1hot, [3, 2, 1, 0], axis=-1)
    rc_seq_1hot = tf.reverse(rc_seq_1hot, axis=[1])
    reverse_bool = tf.random_uniform(shape=[]) > 0.5
    src_seq_1hot = tf.cond(reverse_bool, lambda: rc_seq_1hot, lambda: seq_1hot)
    return src_seq_1hot, reverse_bool

class SwitchReverse(tf.keras.layers.Layer):
  def __init__(self):
    super(SwitchReverse, self).__init__()
  def call(self, x_reverse):
    x = x_reverse[0]
    reverse = x_reverse[1]
    return tf.keras.backend.switch(reverse,
                                   tf.reverse(x, axis=[1]),
                                   x)

class StochasticShift(tf.keras.layers.Layer):
  def __init__(self, shift_max=0, pad='uniform'):
    super(StochasticShift, self).__init__()
    self.augment_shifts = tf.range(-shift_max, shift_max+1)
    self.pad = pad

  def call(self, seq_1hot):
    """Stochastically shift a one hot encoded DNA sequence."""
    shift_i = tf.random_uniform(shape=[], minval=0,
      maxval=len(self.augment_shifts), dtype=tf.int64)
    shift = tf.gather(self.augment_shifts, shift_i)

    sseq_1hot = tf.cond(tf.not_equal(shift, 0),
                        lambda: shift_sequence(seq_1hot, shift),
                        lambda: seq_1hot)

    return sseq_1hot

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

def activate(current, activation):
  if activation == 'relu':
    current = tf.keras.layers.ReLU()(current)
  elif activation == 'gelu':
    current = layers.GELU()(current)
  else:
    print('Unrecognized activation "%s"' % activation, file=sys.stderr)
    exit(1)

  return current
