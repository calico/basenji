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

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.rnn import _reverse_seq

################################################################################
# shiyemin code - https://github.com/tensorflow/tensorflow/issues/7476
def adjust_max(start, stop, start_value, stop_value, name=None):
    with ops.name_scope(name, "AdjustMax",
                        [start, stop, name]) as name:
        global_step = tf.train.get_global_step()
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
def bidirectional_rnn_tied(cell_fw, cell_bw, inputs, initial_state_fw=None, initial_state_bw=None, dtype=None, sequence_length=None, scope=None):
    name = scope or "BiRNN"
    with variable_scope.variable_scope(name) as fw_scope:
        # Forward direction
        output_fw, output_state_fw = tf.nn.rnn(cell_fw, inputs, initial_state_fw, dtype, sequence_length, scope=fw_scope)

    with variable_scope.variable_scope(name, reuse=True) as bw_scope:
        # Backward direction
        tmp, output_state_bw = tf.nn.rnn(cell_bw, _reverse_seq(inputs, sequence_length),
                 initial_state_bw, dtype, sequence_length, scope=bw_scope)

    output_bw = _reverse_seq(tmp, sequence_length)

    # Concat each of the forward/backward outputs
    outputs = [array_ops.concat(1, [fw, bw]) for fw, bw in zip(output_fw, output_bw)]

    return (outputs, output_state_fw, output_state_bw)

################################################################################
def bidirectional_rnn_rc(cell_fw, cell_bw, inputs, initial_state_fw=None, initial_state_bw=None, dtype=None, sequence_length=None, scope=None):
    name = scope or "BiRNN"
    with variable_scope.variable_scope(name) as fw_scope:
        # Forward direction
        output_fw, output_state_fw = tf.nn.rnn(cell_fw, inputs, initial_state_fw, dtype, sequence_length, scope=fw_scope)

    with variable_scope.variable_scope(name, reuse=True) as bw_scope:
        # Backward direction
        tmp, output_state_bw = tf.nn.rnn(cell_bw, _reverse_complement(inputs, sequence_length),
                 initial_state_bw, dtype, sequence_length, scope=bw_scope)

    output_bw = _reverse_seq(tmp, sequence_length)

    # Concat each of the forward/backward outputs
    outputs = [array_ops.concat(1, [fw, bw]) for fw, bw in zip(output_fw, output_bw)]

    return (outputs, output_state_fw, output_state_bw)

################################################################################
def _reverse_complement(input_seq, lengths):
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
        print('Not yet implemented', file=sys.stderr)
        exit(1)
    else:
        nt_rc = tf.constant([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]], dtype='float32')
        return [tf.matmul(ris,nt_rc) for ris in reversed(input_seq)]


