"""Wrapper code for using commonly-used layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from basenji import ops


class Clip(tf.keras.layers.Layer):
  def __init__(self, min_value, max_value):
    super(Clip, self).__init__()
    self.min_value = min_value
    self.max_value = max_value
  def call(self, x):
    return tf.clip_by_value(x, self.min_value, self.max_value)

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
