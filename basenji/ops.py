# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains convenience wrappers for typical Neural Network TensorFlow layers.

   Additionally it maintains a collection with update_ops that need to be
   updated after the ops have been computed, for exmaple to update moving means
   and moving variances of batch_norm.

   Ops that have different behavior during training or eval have an is_training
   parameter. Additionally Ops that contain variables.variable have a trainable
   parameter, which control if the ops variables are trainable or not.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.rnn import _reverse_seq
from tensorflow.python.training import moving_averages

DATA_FORMAT_NCHW = 'NCHW'
DATA_FORMAT_NHWC = 'NHWC'


def create_global_step(graph=None):
  """Create global step tensor in graph.
  Args:
    graph: The graph in which to create the global step tensor. If missing,
      use default graph.
  Returns:
    Global step tensor.
  Raises:
    ValueError: if global step tensor is already defined.
  """
  graph = graph or ops.get_default_graph()
  if tf.train.get_global_step(graph) is not None:
    raise ValueError('"global_step" already exists.')
  # Create in proper graph and base name_scope.
  with graph.as_default() as g, g.name_scope(None):
    return variable_scope.get_variable(
        ops.GraphKeys.GLOBAL_STEP,
        shape=[],
        dtype=dtypes.int64,
        initializer=init_ops.zeros_initializer,
        trainable=False,
        collections=[ops.GraphKeys.GLOBAL_VARIABLES, ops.GraphKeys.GLOBAL_STEP])


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


@add_arg_scope
def fused_batch_norm(
        inputs,
        renorm=False,
        RMAX=None,
        DMAX=None,
        decay=0.999,
        center=True,
        scale=False,
        epsilon=0.001,
        activation_fn=None,
        param_initializers=None,
        is_training=True,
        reuse=None,
        variables_collections=None,
        outputs_collections=None,
        trainable=True,
        data_format=DATA_FORMAT_NHWC,
        zero_debias_moving_mean=False,
        scope=None):
    """Adds a Batch Normalization layer from http://arxiv.org/abs/1502.03167.

        "Batch Normalization: Accelerating Deep Network Training by Reducing
        Internal Covariate Shift"

        Sergey Ioffe, Christian Szegedy

    Can be used as a normalizer function for conv2d and fully_connected.

    Note: When is_training is True the moving_mean and moving_variance need to be
    updated, by default the update_ops are placed in `tf.GraphKeys.UPDATE_OPS` so
    they need to be added as a dependency to the `train_op`, example:

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
        updates = tf.group(*update_ops)
        total_loss = control_flow_ops.with_dependencies([updates], total_loss)

    Args:
        inputs: a tensor with 2 or more dimensions, where the first dimension has
        `batch_size`. The normalization is over all but the last dimension if
        `data_format` is `NHWC` and the second dimension if `data_format` is
        `NCHW`.
        decay: decay for the moving average. Reasonable values for `decay` are close
        to 1.0, typically in the multiple-nines range: 0.999, 0.99, 0.9, etc.
        Lower `decay` value (recommend trying `decay`=0.9) if model experiences
        reasonably good training performance but poor validation and/or test
        performance.
        center: If True, add offset of `beta` to normalized tensor.  If False,
        `beta` is ignored.
        scale: If True, multiply by `gamma`. If False, `gamma` is
        not used. When the next layer is linear (also e.g. `nn.relu`), this can be
        disabled since the scaling can be done by the next layer.
        epsilon: small float added to variance to avoid dividing by zero.
        activation_fn: activation function, default set to None to skip it and
        maintain a linear activation.
        param_initializers: optional initializers for beta, gamma, moving mean and
        moving variance.
        updates_collections: collections to collect the update ops for computation.
        The updates_ops need to be executed with the train_op.
        If None, a control dependency would be added to make sure the updates are
        computed in place.
        is_training: whether or not the layer is in training mode. In training mode
        it would accumulate the statistics of the moments into `moving_mean` and
        `moving_variance` using an exponential moving average with the given
        `decay`. When it is not in training mode then it would use the values of
        the `moving_mean` and the `moving_variance`.
        reuse: whether or not the layer and its variables should be reused. To be
        able to reuse the layer scope must be given.
        variables_collections: optional collections for the variables.
        outputs_collections: collections to add the outputs.
        trainable: If `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
        data_format: A string. `NHWC` (default) and `NCHW` are supported.
        zero_debias_moving_mean: Use zero_debias for moving_mean.
        scope: Optional scope for `variable_scope`.

    Returns:
        A `Tensor` representing the output of the operation.

    Raises:
        ValueError: if `data_format` is neither `NHWC` nor `NCHW`.
        ValueError: if the rank of `inputs` is undefined.
        ValueError: if the rank of `inputs` is neither 2 or 4.
        ValueError: if rank or `C` dimension of `inputs` is undefined.
    """
    if data_format not in (DATA_FORMAT_NCHW, DATA_FORMAT_NHWC):
        raise ValueError('data_format has to be either NCHW or NHWC.')
    with tf.variable_scope(
            scope, 'BatchNorm', [inputs], reuse=reuse) as sc:
        inputs = ops.convert_to_tensor(inputs)
        original_shape = inputs.get_shape()
        original_rank = original_shape.ndims
        if original_rank is None:
            raise ValueError('Inputs %s has undefined rank' % inputs.name)
        elif original_rank not in [2, 4]:
            raise ValueError('Inputs %s has unsupported rank.'
                            ' Expected 2 or 4 but got %d' % (
                                inputs.name, original_rank))
        if original_rank == 2:
            channels = inputs.get_shape()[-1].value
            if channels is None:
                raise ValueError('`C` dimension must be known but is None')
            new_shape = [-1, 1, 1, channels]
            if data_format == DATA_FORMAT_NCHW:
                new_shape = [-1, channels, 1, 1]
            inputs = array_ops.reshape(inputs, new_shape)
        inputs_shape = inputs.get_shape()
        dtype = inputs.dtype.base_dtype
        if data_format == DATA_FORMAT_NHWC:
            params_shape = inputs_shape[-1:]
        else:
            params_shape = inputs_shape[1:2]
        if not params_shape.is_fully_defined():
            raise ValueError('Inputs %s has undefined `C` dimension %s.' %
                            (inputs.name, params_shape))

        if not param_initializers:
            param_initializers = {}
        # Allocate parameters for the beta and gamma of the normalization.
        trainable_beta = trainable and center
        if trainable_beta:
            beta_collections = utils.get_variable_collections(variables_collections,
                                                            'beta')
            beta_initializer = param_initializers.get('beta',
                                                    init_ops.zeros_initializer)
            real_beta = variables.model_variable(
                    'beta',
                    shape=params_shape,
                    dtype=dtype,
                    initializer=beta_initializer,
                    collections=beta_collections,
                    trainable=trainable_beta)
            beta = tf.zeros(params_shape, name='fakebeta')
        else:
            real_beta = tf.zeros(params_shape, name='beta')
            beta = tf.zeros(params_shape, name='fakebeta')
        trainable_gamma = trainable and scale
        if trainable_gamma:
            gamma_collections = utils.get_variable_collections(variables_collections,
                                                            'gamma')
            gamma_initializer = param_initializers.get('gamma',
                                                    init_ops.ones_initializer())
            gamma = variables.model_variable(
                    'gamma',
                    shape=params_shape,
                    dtype=dtype,
                    initializer=gamma_initializer,
                    collections=gamma_collections,
                    trainable=trainable_gamma)
        else:
            gamma = tf.ones(params_shape, name='gamma')

        # Create moving_mean and moving_variance variables and add them to the
        # appropiate collections.
        moving_mean_collections = utils.get_variable_collections(
                variables_collections, 'moving_mean')
        moving_mean_initializer = param_initializers.get(
                'moving_mean', init_ops.zeros_initializer)
        moving_mean = variables.model_variable(
                'moving_mean',
                shape=params_shape,
                dtype=dtype,
                initializer=moving_mean_initializer,
                trainable=False,
                collections=moving_mean_collections)
        moving_variance_collections = utils.get_variable_collections(
                variables_collections, 'moving_variance')
        moving_variance_initializer = param_initializers.get(
                'moving_variance', init_ops.ones_initializer())
        moving_variance = variables.model_variable(
                'moving_variance',
                shape=params_shape,
                dtype=dtype,
                initializer=moving_variance_initializer,
                trainable=False,
                collections=moving_variance_collections)

        def _fused_batch_norm_training():
            outputs, mean, variance = nn.fused_batch_norm(
                    inputs, gamma, beta, epsilon=epsilon, data_format=data_format)
            if renorm:
                moving_inv = math_ops.rsqrt(moving_variance + epsilon)
                r = tf.stop_gradient(tf.clip_by_value(tf.sqrt(variance + epsilon) * moving_inv,
                                                        1/RMAX,
                                                        RMAX))
                d = tf.stop_gradient(tf.clip_by_value((mean - moving_mean) * moving_inv,
                                                        -DMAX,
                                                        DMAX))
                outputs = outputs * r + d
            return outputs, mean, variance
        def _fused_batch_norm_inference():
            return nn.fused_batch_norm(
                    inputs,
                    gamma,
                    beta,
                    mean=moving_mean,
                    variance=moving_variance,
                    epsilon=epsilon,
                    is_training=False,
                    data_format=data_format)
        outputs, mean, variance = utils.smart_cond(is_training,
                                                _fused_batch_norm_training,
                                                _fused_batch_norm_inference)
        outputs = tf.nn.bias_add(outputs, real_beta)

        # If `is_training` doesn't have a constant value, because it is a `Tensor`,
        # a `Variable` or `Placeholder` then is_training_value will be None and
        # `need_updates` will be true.
        is_training_value = utils.constant_value(is_training)
        need_updates = is_training_value is None or is_training_value
        if need_updates:
            moving_vars_fn = lambda: (moving_mean, moving_variance)
            def _delay_updates():
                """Internal function that delay updates moving_vars if is_training."""
                update_moving_mean = moving_averages.assign_moving_average(
                        moving_mean, mean, decay, zero_debias=zero_debias_moving_mean)
                update_moving_variance = moving_averages.assign_moving_average(
                        moving_variance, variance, decay, zero_debias=False)
                return update_moving_mean, update_moving_variance
            update_mean, update_variance = utils.smart_cond(is_training,
                                                            _delay_updates,
                                                            moving_vars_fn)
            ops.add_to_collections(ops.GraphKeys.UPDATE_OPS, update_mean)
            ops.add_to_collections(ops.GraphKeys.UPDATE_OPS, update_variance)

        outputs.set_shape(inputs_shape)
        if original_shape.ndims == 2:
            outputs = array_ops.reshape(outputs, original_shape)
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return utils.collect_named_outputs(outputs_collections,
                                        sc.original_name_scope, outputs)


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


