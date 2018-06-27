# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import pdb
import tensorflow as tf

from basenji import ops

def shift_sequence(seq, shift_amount, pad_value=0.25):
  """Shift a sequence left or right by shift_amount.

  Args:
    seq: a [batch_size, sequence_length, sequence_depth] sequence to shift
    shift_amount: the signed amount to shift (tf.int32 or int)
    pad_value: value to fill the padding (primitive or scalar tf.Tensor)
  """
  if seq.shape.ndims != 3:
    raise ValueError('input sequence should be rank 3')
  input_shape = seq.shape

  pad = pad_value * tf.ones_like(seq[:, 0:tf.abs(shift_amount), :])

  def _shift_right(_seq):
    sliced_seq = _seq[:, :-shift_amount:, :]
    return tf.concat([pad, sliced_seq], axis=1)

  def _shift_left(_seq):
    sliced_seq = _seq[:, -shift_amount:, :]
    return tf.concat([sliced_seq, pad], axis=1)

  output = tf.cond(
      tf.greater(shift_amount, 0), lambda: _shift_right(seq),
      lambda: _shift_left(seq))

  output.set_shape(input_shape)
  return output

def augment_deterministic_set(data_ops, augment_rc=False, augment_shifts=[0]):
  """

  Args:
    data_ops: dict with keys 'sequence,' 'label,' and 'na.'
    augment_rc: Boolean
    augment_shifts: List of ints.
  Returns
    data_ops_list:
  """
  augment_pairs = []
  for ashift in augment_shifts:
    augment_pairs.append((False, ashift))
    if augment_rc:
      augment_pairs.append((True, ashift))

  data_ops_list = []
  for arc, ashift in augment_pairs:
    data_ops_aug = augment_deterministic(data_ops, arc, ashift)
    data_ops_list.append(data_ops_aug)

  return data_ops_list


def augment_deterministic(data_ops, augment_rc=False, augment_shift=0):
  """Apply a deterministic augmentation, specified by the parameters.

  Args:
    data_ops: dict with keys 'sequence,' 'label,' and 'na.'
    augment_rc: Boolean
    augment_shift: Int
  Returns
    data_ops: augmented data, with all existing keys transformed
              and 'reverse_preds' bool added.
  """
  data_ops_aug = {}
  for key in data_ops:
    if key not in ['sequence']:
      data_ops_aug[key] = data_ops[key]

  if augment_shift == 0:
    data_ops_aug['sequence'] = data_ops['sequence']
  else:
    shift_amount = tf.constant(augment_shift, shape=(), dtype=tf.int64)
    data_ops_aug['sequence'] = shift_sequence(data_ops['sequence'], shift_amount)

  if augment_rc:
    data_ops_aug = augment_deterministic_rc(data_ops_aug)
  else:
    data_ops_aug['reverse_preds'] = tf.zeros((), dtype=tf.bool)

  return data_ops_aug


def augment_deterministic_rc(data_ops):
  """Apply a deterministic reverse complement augmentation.

  Args:
    data_ops: dict with keys 'sequence,' 'label,' and 'na.'
  Returns
    data_ops_aug: augmented data ops
  """
  data_ops_aug = ops.reverse_complement_transform(data_ops)
  data_ops_aug['reverse_preds'] = tf.ones((), dtype=tf.bool)
  return data_ops_aug


def augment_stochastic_rc(data_ops):
  """Apply a stochastic reverse complement augmentation.

  Args:
    data_ops: dict with keys 'sequence,' 'label,' and 'na.'
  Returns
    data_ops_aug: augmented data
  """
  reverse_preds = tf.random_uniform(shape=[]) > 0.5
  data_ops_aug = tf.cond(reverse_preds, lambda: ops.reverse_complement_transform(data_ops),
                                        lambda: data_ops.copy())
  data_ops_aug['reverse_preds'] = reverse_preds
  return data_ops_aug


def augment_stochastic_shifts(seq, augment_shifts):
  """Apply a stochastic shift augmentation.

  Args:
    seq: input sequence of size [batch_size, length, depth]
    augment_shifts: list of int offsets to sample from
  Returns:
    shifted and padded sequence of size [batch_size, length, depth]
  """
  shift_index = tf.random_uniform(shape=[], minval=0,
      maxval=len(augment_shifts), dtype=tf.int64)
  shift_value = tf.gather(tf.constant(augment_shifts), shift_index)

  seq = tf.cond(tf.not_equal(shift_value, 0),
                lambda: shift_sequence(seq, shift_value),
                lambda: seq)

  return seq


def augment_stochastic(data_ops, augment_rc=False, augment_shifts=[]):
  """Apply stochastic augmentations,

  Args:
    data_ops: dict with keys 'sequence,' 'label,' and 'na.'
    augment_rc: Boolean for whether to apply reverse complement augmentation.
    augment_shifts: list of int offsets to sample shift augmentations.
  Returns:
    data_ops_aug: augmented data
  """
  if augment_shifts:
    data_ops['sequence'] = augment_stochastic_shifts(data_ops['sequence'],
                                                     augment_shifts)

  if augment_rc:
    data_ops = augment_stochastic_rc(data_ops)
  else:
    data_ops['reverse_preds'] = tf.zeros((), dtype=tf.bool)

  return data_ops
