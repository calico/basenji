"""Wrapper code for using commonly-used layers."""
import sys
import tensorflow as tf

from tensor2tensor.layers.common_attention import attention_bias_proximal
from tensor2tensor.layers.common_attention import _generate_relative_positions_embeddings, _relative_attention_inner

from basenji import ops

############################################################
# Keras Layers
############################################################

class Clip(tf.keras.layers.Layer):
  def __init__(self, min_value, max_value):
    super(Clip, self).__init__()
    self.min_value = min_value
    self.max_value = max_value
  def call(self, x):
    return tf.clip_by_value(x, self.min_value, self.max_value)

class Exp(tf.keras.layers.Layer):
    def __init__(self):
        super(Exp, self).__init__()
    def call(self, x):
      return tf.keras.activations.exponential(x)

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


class Attention(tf.keras.layers.Layer):
  def __init__(self, max_relative_position, dropout=0):
    super(Attention, self).__init__()
    self.max_relative_position = max_relative_position
    self.dropout = dropout

  def build(self, input_shape):
    # extract shapes
    qs, vs, ks = input_shape
    seq_length = qs[-2]
    depth_kq = qs[-1]
    depth_q = ks[-1]
    assert(depth_kq == depth_q)
    depth_v = vs[-1]

    # initialize bias
    self.attn_bias = attention_bias_proximal(seq_length)

    # initialize relative positions
    self.relations_keys = _generate_relative_positions_embeddings(
      seq_length, seq_length, depth_kq, self.max_relative_position,
      "relative_positions_keys")
    self.relations_values = _generate_relative_positions_embeddings(
      seq_length, seq_length, depth_v, self.max_relative_position,
      "relative_positions_values")
    # tf.contrib.summary.histogram('relations_keys', self.relations_keys)
    # tf.contrib.summary.histogram('relations_values', self.relations_values)

  def call(self, qvk):
    query, value, key = qvk

    # expand to fake multi-head
    key = tf.expand_dims(key, axis=1)
    query = tf.expand_dims(query, axis=1)
    value = tf.expand_dims(value, axis=1)

    # Compute self attention considering the relative position embeddings.
    logits = _relative_attention_inner(query, key, self.relations_keys, True)
    logits += self.attn_bias

    weights = tf.nn.softmax(logits, name="attention_weights")
    weights = tf.nn.dropout(weights, rate=self.dropout)

    z = _relative_attention_inner(weights, value, self.relations_values, False)

    # slice single head
    z = z[:,0,:,:]

    return z

############################################################
# Augmentation
############################################################

class EnsembleReverseComplement(tf.keras.layers.Layer):
  def __init__(self):
    super(EnsembleReverseComplement, self).__init__()
  def call(self, seqs_1hot):
    """Expand tensor to include reverse complement of one hot encoded DNA sequence."""
    if not isinstance(seqs_1hot, list):
      seqs_1hot = [seqs_1hot]

    ens_seqs_1hot = []
    for seq_1hot in seqs_1hot:
      rc_seq_1hot = tf.gather(seq_1hot, [3, 2, 1, 0], axis=-1)
      rc_seq_1hot = tf.reverse(rc_seq_1hot, axis=[1])
      ens_seqs_1hot += [(seq_1hot, tf.constant(False)), (rc_seq_1hot, tf.constant(True))]

    return ens_seqs_1hot

class StochasticReverseComplement(tf.keras.layers.Layer):
  def __init__(self):
    super(StochasticReverseComplement, self).__init__()
  def call(self, seq_1hot, training):
    """Stochastically reverse complement a one hot encoded DNA sequence."""
    if training:
      rc_seq_1hot = tf.gather(seq_1hot, [3, 2, 1, 0], axis=-1)
      rc_seq_1hot = tf.reverse(rc_seq_1hot, axis=[1])
      reverse_bool = tf.random.uniform(shape=[]) > 0.5
      src_seq_1hot = tf.cond(reverse_bool, lambda: rc_seq_1hot, lambda: seq_1hot)
      return src_seq_1hot, reverse_bool
    else:
      return seq_1hot, tf.constant(False)


class SwitchReverse(tf.keras.layers.Layer):
  def __init__(self):
    super(SwitchReverse, self).__init__()
  def call(self, x_reverse):
    x = x_reverse[0]
    reverse = x_reverse[1]
    return tf.keras.backend.switch(reverse,
                                   tf.reverse(x, axis=[1]),
                                   x)

class EnsembleShift(tf.keras.layers.Layer):
  def __init__(self, shifts=[0], pad='uniform'):
    super(EnsembleShift, self).__init__()
    self.shifts = shifts
    self.pad = pad
  def call(self, seqs_1hot):
    """Expand tensor to include shifts of one hot encoded DNA sequence."""
    if not isinstance(seqs_1hot, list):
      seqs_1hot = [seqs_1hot]

    ens_seqs_1hot = []
    for seq_1hot in seqs_1hot:
      for shift in self.shifts:
        ens_seqs_1hot.append(shift_sequence(seq_1hot, shift))

    return ens_seqs_1hot

class StochasticShift(tf.keras.layers.Layer):
  def __init__(self, shift_max=0, pad='uniform'):
    super(StochasticShift, self).__init__()
    self.augment_shifts = tf.range(-shift_max, shift_max+1)
    self.pad = pad
  def call(self, seq_1hot, training):
    """Stochastically shift a one hot encoded DNA sequence."""
    if training:
      shift_i = tf.random.uniform(shape=[], minval=0,
        maxval=len(self.augment_shifts), dtype=tf.int64)
      shift = tf.gather(self.augment_shifts, shift_i)

      sseq_1hot = tf.cond(tf.not_equal(shift, 0),
                          lambda: shift_sequence(seq_1hot, shift),
                          lambda: seq_1hot)
    else:
      sseq_1hot = seq_1hot

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


############################################################
# helpers
############################################################

def activate(current, activation, verbose=False):
  if verbose: print('activate:',activation)
  if activation == 'relu':
    current = tf.keras.layers.ReLU()(current)
  elif activation == 'gelu':
    current = GELU()(current)
  elif activation == 'sigmoid':
    current = tf.keras.layers.Activation('sigmoid')(current)
  elif activation == 'tanh':
    current = tf.keras.layers.Activation('tanh')(current)
  elif activation == 'exp':
    current = Exp()(current)
  else:
    print('Unrecognized activation "%s"' % activation, file=sys.stderr)
    exit(1)

  return current
