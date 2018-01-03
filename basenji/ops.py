# Copyright 2017 Calico LLC

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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

################################################################################
# shiyemin code - https://github.com/tensorflow/tensorflow/issues/7476
def adjust_max(start, stop, start_value, stop_value, name=None):
  with tf.name_scope(name, "AdjustMax", [start, stop, name]) as name:
    global_step = tf.train.get_or_create_global_step()
    if global_step is not None:
      start = tf.convert_to_tensor(start, dtype=tf.int64)
      stop = tf.convert_to_tensor(stop, dtype=tf.int64)
      start_value = tf.convert_to_tensor(start_value, dtype=tf.float32)
      stop_value = tf.convert_to_tensor(stop_value, dtype=tf.float32)

      pred_fn_pairs = {}
      pred_fn_pairs[global_step <= start] = lambda: start_value
      pred_fn_pairs[(global_step > start) & (global_step <= stop)] = lambda: tf.train.polynomial_decay(
                                  start_value, global_step-start, stop-start,
                                  end_learning_rate=stop_value, power=1.0, cycle=False)
      default = lambda: stop_value
      return tf.case(pred_fn_pairs, default, exclusive=True)
    else:
      return None


################################################################################
def bidirectional_rnn_tied(cell_fw,
                           cell_bw,
                           inputs,
                           initial_state_fw=None,
                           initial_state_bw=None,
                           dtype=None,
                           sequence_length=None,
                           scope=None):
  name = scope or "BiRNN"
  with tf.variable_scope(name) as fw_scope:
    # Forward direction
    output_fw, output_state_fw = tf.nn.rnn(
        cell_fw,
        inputs,
        initial_state_fw,
        dtype,
        sequence_length,
        scope=fw_scope)

  with tf.variable_scope(name, reuse=True) as bw_scope:
    # Backward direction
    tmp, output_state_bw = tf.nn.rnn(
        cell_bw,
        tf.reverse(inputs, sequence_length),
        initial_state_bw,
        dtype,
        sequence_length,
        scope=bw_scope)

  output_bw = tf.reverse(tmp, sequence_length)

  # Concat each of the forward/backward outputs
  outputs = [tf.concat(1, [fw, bw]) for fw, bw in zip(output_fw, output_bw)]

  return (outputs, output_state_fw, output_state_bw)


################################################################################
def bidirectional_rnn_rc(cell_fw,
                         cell_bw,
                         inputs,
                         initial_state_fw=None,
                         initial_state_bw=None,
                         dtype=None,
                         sequence_length=None,
                         scope=None):
  name = scope or "BiRNN"
  with tf.variable_scope(name) as fw_scope:
    # Forward direction
    output_fw, output_state_fw = tf.nn.rnn(
        cell_fw,
        inputs,
        initial_state_fw,
        dtype,
        sequence_length,
        scope=fw_scope)

  with tf.variable_scope(name, reuse=True) as bw_scope:
    # Backward direction
    tmp, output_state_bw = tf.nn.rnn(
        cell_bw,
        reverse_complement(inputs, sequence_length),
        initial_state_bw,
        dtype,
        sequence_length,
        scope=bw_scope)

  output_bw = tf.reverse(tmp, sequence_length)

  # Concat each of the forward/backward outputs
  outputs = [tf.concat(1, [fw, bw]) for fw, bw in zip(output_fw, output_bw)]

  return (outputs, output_state_fw, output_state_bw)


def reverse_complement_transform(seq, label, na):
  """Reverse complement of batched onehot seq and corresponding label and na."""
  rank = seq.shape.ndims
  if rank != 3:
    raise ValueError("input seq must be rank 3.")

  complement = tf.gather(seq, [3, 2, 1, 0], axis=-1)
  return (tf.reverse(complement, axis=[1]),
          tf.reverse(label, axis=[1]),
          tf.reverse(na, axis=[1]))


def reverse_complement(input_seq, lengths=None):
  # TODO(dbelanger) remove dependencies on this method,
  # as it is easy to mis-use in ways that lead to buggy results.
  """Reverse complement a list of one hot coded nucleotide Tensors.
    Args:
    input_seq: Sequence of seq_len tensors of dimension (batch_size, 4)
    lengths:   A `Tensor` of dimension batch_size, containing lengths for each
               sequence in the batch. If "None" is specified, simply reverse
               complements the list.
    Returns:
    reverse complemented sequence
    """
  if lengths is not None:
    print("Not yet implemented", file=sys.stderr)
    exit(1)
  else:
    nt_rc = tf.constant(
        [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]],
        dtype="float32")
    return [tf.matmul(ris, nt_rc) for ris in reversed(input_seq)]

def variance(data, weights=None):
  """Returns the variance of input tensor t, each entry weighted by the
  corresponding index in weights.

  Follows the tf.metrics API for an idempotent tensor and an update tensor.

  Args:
    data: input tensor of arbitrary shape.
    weights: input tensor of same shape as `t`. When None, use a weight of 1 for
      all inputs.

  Returns:
    variance_value: idempotent tensor containing the variance of `t`, whose
      shape is `[1]`
    update_op: A (non-idempotent) op to update the variance value
  """
  if weights is None:
    weights = tf.ones(shape=data.shape, dtype=tf.float32)

  tsquared_mean, tsquared_update = tf.metrics.mean(tf.square(data), weights)
  mean_t, t_update = tf.metrics.mean(data, weights)
  variance_value = tsquared_mean - mean_t*mean_t
  update_op = tf.group(tsquared_update, t_update)

  return variance_value, update_op

def r2_metric(preds, targets, weights):
  """Returns ops for R2 statistic following the tf.metrics API.
  Args:
    preds: predictions (arbitrary shape)
    targets: targets (same shape as predictions)
    weights: per-instance weights (same shape as predictions)

  Returns:
    r2: idempotent tensor containing the r2 value
    update_op: op for updating the value given new data
  """

  res_ss, res_ss_update = tf.metrics.mean(tf.square(preds - targets), weights)

  tot_ss, tot_ss_update = variance(targets, weights)
  r2 = 1. - res_ss / tot_ss

  update_op = tf.group(res_ss_update, tot_ss_update)
  return r2, update_op

def per_target_r2(preds, targets, weights):
  """Returns ops for per-target R2 statistic following the tf.metrics API.
  Args:
    preds: arbitrary shaped predictions, with final dimension
           indexing distinct targets
    targets: targets (same shape as predictions)
    weights: per-instance weights (same shape as predictions)

  Returns:
    r2: idempotent [preds.shape[-1]] tensor of r2 values for each target.
    update_op: op for updating the value given new data
  """
  preds_split = tf.unstack(preds, axis=-1)
  targets_split = tf.unstack(targets, axis=-1)
  weights_split = tf.unstack(weights, axis=-1)

  r2_metrics = [
      r2_metric(p, t, w)
      for p, t, w in zip(preds_split, targets_split, weights_split)
  ]

  r2_values = [r[0] for r in r2_metrics]
  stacked_r2 = tf.stack(r2_values)
  update_ops = tf.group(*[r[1] for r in r2_metrics])
  return stacked_r2, update_ops



def r2_averaged_over_all_prediction_tasks(preds, targets, weights):
  """Returns ops for multi-task R2 statistic following the tf.metrics API.
  Args:
    preds: predictions, with final dimension indexing distinct targets.
    targets: targets (same shape as predictions)
    weights: per-instance weights (same shape as predictions)

  Returns:
    r2: idempotent tensor containing the mean multi-task r2 value,
      of shape `[1]`
    update_op: op for updating the value given new data
  """
  r2s, update = per_target_r2(preds, targets, weights)
  mean = tf.reduce_mean(r2s)
  return mean, update
