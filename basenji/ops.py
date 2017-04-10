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
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.rnn import _reverse_seq
from tensorflow.python.training import moving_averages

DATA_FORMAT_NCHW = 'NCHW'
DATA_FORMAT_NHWC = 'NHWC'

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
# tensorflow/tensorflow/contrib/layers/python/layers/layers.py

@add_arg_scope
def batch_norm(inputs,
               decay=0.999,
               center=True,
               scale=False,
               epsilon=0.001,
               activation_fn=None,
               param_initializers=None,
               param_regularizers=None,
               updates_collections=ops.GraphKeys.UPDATE_OPS,
               is_training=True,
               reuse=None,
               variables_collections=None,
               outputs_collections=None,
               trainable=True,
               batch_weights=None,
               fused=False,
               data_format=DATA_FORMAT_NHWC,
               zero_debias_moving_mean=False,
               scope=None,
               renorm=False,
               renorm_clipping=None,
               renorm_decay=0.99):
  """Adds a Batch Normalization layer from http://arxiv.org/abs/1502.03167.
    "Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift"
    Sergey Ioffe, Christian Szegedy
  Can be used as a normalizer function for conv2d and fully_connected.
  Note: When is_training is True the moving_mean and moving_variance need to be
  updated, by default the update_ops are placed in `tf.GraphKeys.UPDATE_OPS` so
  they need to be added as a dependency to the `train_op`, example:
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss)
  One can set updates_collections=None to force the updates in place, but that
  can have speed penalty, especially in distributed settings.
  Args:
    inputs: A tensor with 2 or more dimensions, where the first dimension has
      `batch_size`. The normalization is over all but the last dimension if
      `data_format` is `NHWC` and the second dimension if `data_format` is
      `NCHW`.
    decay: Decay for the moving average. Reasonable values for `decay` are close
      to 1.0, typically in the multiple-nines range: 0.999, 0.99, 0.9, etc.
      Lower `decay` value (recommend trying `decay`=0.9) if model experiences
      reasonably good training performance but poor validation and/or test
      performance. Try zero_debias_moving_mean=True for improved stability.
    center: If True, add offset of `beta` to normalized tensor. If False, `beta`
      is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    epsilon: Small float added to variance to avoid dividing by zero.
    activation_fn: Activation function, default set to None to skip it and
      maintain a linear activation.
    param_initializers: Optional initializers for beta, gamma, moving mean and
      moving variance.
    param_regularizers: Optional regularizer for beta and gamma.
    updates_collections: Collections to collect the update ops for computation.
      The updates_ops need to be executed with the train_op.
      If None, a control dependency would be added to make sure the updates are
      computed in place.
    is_training: Whether or not the layer is in training mode. In training mode
      it would accumulate the statistics of the moments into `moving_mean` and
      `moving_variance` using an exponential moving average with the given
      `decay`. When it is not in training mode then it would use the values of
      the `moving_mean` and the `moving_variance`.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional collections for the variables.
    outputs_collections: Collections to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    batch_weights: An optional tensor of shape `[batch_size]`,
      containing a frequency weight for each batch item. If present,
      then the batch normalization uses weighted mean and
      variance. (This can be used to correct for bias in training
      example selection.)
    fused:  Use nn.fused_batch_norm if True, nn.batch_normalization otherwise.
    data_format: A string. `NHWC` (default) and `NCHW` are supported.
    zero_debias_moving_mean: Use zero_debias for moving_mean. It creates a new
      pair of variables 'moving_mean/biased' and 'moving_mean/local_step'.
    scope: Optional scope for `variable_scope`.
    renorm: Whether to use Batch Renormalization
      (https://arxiv.org/abs/1702.03275). This adds extra variables during
      training. The inference is the same for either value of this parameter.
    renorm_clipping: A dictionary that may map keys 'rmax', 'rmin', 'dmax' to
      scalar `Tensors` used to clip the renorm correction. The correction
      `(r, d)` is used as `corrected_value = normalized_value * r + d`, with
      `r` clipped to [rmin, rmax], and `d` to [-dmax, dmax]. Missing rmax, rmin,
      dmax are set to inf, 0, inf, respectively.
    renorm_decay: Momentum used to update the moving means and standard
      deviations with renorm. Unlike `momentum`, this affects training
      and should be neither too small (which would add noise) nor too large
      (which would give stale estimates). Note that `decay` is still applied
      to get the means and variances for inference.
  Returns:
    A `Tensor` representing the output of the operation.
  Raises:
    ValueError: If `batch_weights` is not None and `fused` is True.
    ValueError: If `param_regularizers` is not None and `fused` is True.
    ValueError: If `data_format` is neither `NHWC` nor `NCHW`.
    ValueError: If the rank of `inputs` is undefined.
    ValueError: If rank or channels dimension of `inputs` is undefined.
  """
  if fused:
    if batch_weights is not None:
      raise ValueError('Weighted mean and variance is not currently '
                       'supported for fused batch norm.')
    if param_regularizers is not None:
      raise ValueError('Regularizers are not currently '
                       'supported for fused batch norm.')
    if renorm:
      raise ValueError('Renorm is not supported for fused batch norm.')
    return _fused_batch_norm(
        inputs,
        decay=decay,
        center=center,
        scale=scale,
        epsilon=epsilon,
        activation_fn=activation_fn,
        param_initializers=param_initializers,
        updates_collections=updates_collections,
        is_training=is_training,
        reuse=reuse,
        variables_collections=variables_collections,
        outputs_collections=outputs_collections,
        trainable=trainable,
        data_format=data_format,
        zero_debias_moving_mean=zero_debias_moving_mean,
        scope=scope)

  if data_format not in (DATA_FORMAT_NCHW, DATA_FORMAT_NHWC):
    raise ValueError('data_format has to be either NCHW or NHWC.')

  layer_variable_getter = _build_variable_getter()
  with variable_scope.variable_scope(
      scope, 'BatchNorm', [inputs], reuse=reuse,
      custom_getter=layer_variable_getter) as sc:
    inputs = ops.convert_to_tensor(inputs)

    # Determine whether we can use the core layer class.
    if (batch_weights is None and
        updates_collections is ops.GraphKeys.UPDATE_OPS and
        not zero_debias_moving_mean):
      # Use the core layer class.
      axis = 1 if data_format == DATA_FORMAT_NCHW else -1
      if not param_initializers:
        param_initializers = {}
      beta_initializer = param_initializers.get('beta',
                                                init_ops.zeros_initializer())
      gamma_initializer = param_initializers.get('gamma',
                                                 init_ops.ones_initializer())
      moving_mean_initializer = param_initializers.get(
          'moving_mean', init_ops.zeros_initializer())
      moving_variance_initializer = param_initializers.get(
          'moving_variance', init_ops.ones_initializer())
      if not param_regularizers:
        param_regularizers = {}
      beta_regularizer = param_regularizers.get('beta')
      gamma_regularizer = param_regularizers.get('gamma')
      layer = BatchNormalization(
          axis=axis,
          momentum=decay,
          epsilon=epsilon,
          center=center,
          scale=scale,
          beta_initializer=beta_initializer,
          gamma_initializer=gamma_initializer,
          moving_mean_initializer=moving_mean_initializer,
          moving_variance_initializer=moving_variance_initializer,
          beta_regularizer=beta_regularizer,
          gamma_regularizer=gamma_regularizer,
          trainable=trainable,
          renorm=renorm,
          renorm_clipping=renorm_clipping,
          renorm_momentum=renorm_decay,
          name=sc.name,
          _scope=sc,
          _reuse=reuse)
      outputs = layer.apply(inputs, training=is_training)

      # Add variables to collections.
      _add_variable_to_collections(
          layer.moving_mean, variables_collections, 'moving_mean')
      _add_variable_to_collections(
          layer.moving_variance, variables_collections, 'moving_variance')
      if layer.beta:
        _add_variable_to_collections(layer.beta, variables_collections, 'beta')
      if layer.gamma:
        _add_variable_to_collections(
            layer.gamma, variables_collections, 'gamma')

      if activation_fn is not None:
        outputs = activation_fn(outputs)
      return utils.collect_named_outputs(outputs_collections,
                                         sc.original_name_scope, outputs)

    # Not supported by layer class: batch_weights argument,
    # and custom updates_collections. In that case, use the legacy BN
    # implementation.
    # Custom updates collections are not supported because the update logic
    # is different in this case, in particular w.r.t. "forced updates" and
    # update op reuse.
    if renorm:
      raise ValueError('renorm is not supported with batch_weights, '
                       'updates_collections or zero_debias_moving_mean')
    inputs_shape = inputs.get_shape()
    inputs_rank = inputs_shape.ndims
    if inputs_rank is None:
      raise ValueError('Inputs %s has undefined rank.' % inputs.name)
    dtype = inputs.dtype.base_dtype
    if batch_weights is not None:
      batch_weights = ops.convert_to_tensor(batch_weights)
      inputs_shape[0:1].assert_is_compatible_with(batch_weights.get_shape())
      # Reshape batch weight values so they broadcast across inputs.
      nshape = [-1] + [1 for _ in range(inputs_rank - 1)]
      batch_weights = array_ops.reshape(batch_weights, nshape)

    if data_format == DATA_FORMAT_NCHW:
      moments_axes = [0] + list(range(2, inputs_rank))
      params_shape = inputs_shape[1:2]
      # For NCHW format, rather than relying on implicit broadcasting, we
      # explicitly reshape the params to params_shape_broadcast when computing
      # the moments and the batch normalization.
      params_shape_broadcast = list(
          [1, inputs_shape[1].value] + [1 for _ in range(2, inputs_rank)])
    else:
      moments_axes = list(range(inputs_rank - 1))
      params_shape = inputs_shape[-1:]
      params_shape_broadcast = None
    if not params_shape.is_fully_defined():
      raise ValueError('Inputs %s has undefined channels dimension %s.' % (
          inputs.name, params_shape))

    # Allocate parameters for the beta and gamma of the normalization.
    beta, gamma = None, None
    if not param_initializers:
      param_initializers = {}
    if center:
      beta_collections = utils.get_variable_collections(variables_collections,
                                                        'beta')
      beta_initializer = param_initializers.get('beta',
                                                init_ops.zeros_initializer())
      beta = variables.model_variable('beta',
                                      shape=params_shape,
                                      dtype=dtype,
                                      initializer=beta_initializer,
                                      collections=beta_collections,
                                      trainable=trainable)
    if scale:
      gamma_collections = utils.get_variable_collections(variables_collections,
                                                         'gamma')
      gamma_initializer = param_initializers.get('gamma',
                                                 init_ops.ones_initializer())
      gamma = variables.model_variable('gamma',
                                       shape=params_shape,
                                       dtype=dtype,
                                       initializer=gamma_initializer,
                                       collections=gamma_collections,
                                       trainable=trainable)

    # Create moving_mean and moving_variance variables and add them to the
    # appropiate collections. We disable variable partitioning while creating
    # them, because assign_moving_average is not yet supported for partitioned
    # variables.
    partitioner = variable_scope.get_variable_scope().partitioner
    try:
      variable_scope.get_variable_scope().set_partitioner(None)
      moving_mean_collections = utils.get_variable_collections(
          variables_collections, 'moving_mean')
      moving_mean_initializer = param_initializers.get(
          'moving_mean', init_ops.zeros_initializer())
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
    finally:
      variable_scope.get_variable_scope().set_partitioner(partitioner)

    # If `is_training` doesn't have a constant value, because it is a `Tensor`,
    # a `Variable` or `Placeholder` then is_training_value will be None and
    # `needs_moments` will be true.
    is_training_value = utils.constant_value(is_training)
    need_moments = is_training_value is None or is_training_value
    if need_moments:
      # Calculate the moments based on the individual batch.
      if batch_weights is None:
        if data_format == DATA_FORMAT_NCHW:
          mean, variance = nn.moments(inputs, moments_axes, keep_dims=True)
          mean = array_ops.reshape(mean, [-1])
          variance = array_ops.reshape(variance, [-1])
        else:
          mean, variance = nn.moments(inputs, moments_axes)
      else:
        if data_format == DATA_FORMAT_NCHW:
          mean, variance = nn.weighted_moments(inputs, moments_axes,
                                               batch_weights, keep_dims=True)
          mean = array_ops.reshape(mean, [-1])
          variance = array_ops.reshape(variance, [-1])
        else:
          mean, variance = nn.weighted_moments(inputs, moments_axes,
                                               batch_weights)

      moving_vars_fn = lambda: (moving_mean, moving_variance)
      if updates_collections is None:
        def _force_updates():
          """Internal function forces updates moving_vars if is_training."""
          update_moving_mean = moving_averages.assign_moving_average(
              moving_mean, mean, decay, zero_debias=zero_debias_moving_mean)
          update_moving_variance = moving_averages.assign_moving_average(
              moving_variance, variance, decay, zero_debias=False)
          with ops.control_dependencies([update_moving_mean,
                                         update_moving_variance]):
            return array_ops.identity(mean), array_ops.identity(variance)
        mean, variance = utils.smart_cond(is_training,
                                          _force_updates,
                                          moving_vars_fn)
      else:
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
        ops.add_to_collections(updates_collections, update_mean)
        ops.add_to_collections(updates_collections, update_variance)
        # Use computed moments during training and moving_vars otherwise.
        vars_fn = lambda: (mean, variance)
        mean, variance = utils.smart_cond(is_training, vars_fn, moving_vars_fn)
    else:
      mean, variance = moving_mean, moving_variance
    if data_format == DATA_FORMAT_NCHW:
      mean = array_ops.reshape(mean, params_shape_broadcast)
      variance = array_ops.reshape(variance, params_shape_broadcast)
      beta = array_ops.reshape(beta, params_shape_broadcast)
      if gamma is not None:
        gamma = array_ops.reshape(gamma, params_shape_broadcast)

    # Compute batch_normalization.
    outputs = nn.batch_normalization(inputs, mean, variance, beta, gamma,
                                     epsilon)
    outputs.set_shape(inputs_shape)
    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return utils.collect_named_outputs(outputs_collections,
                                       sc.original_name_scope, outputs)



################################################################################
# tensorflow/tensorflow/python/layers/normalization.py

class BatchNormalization(base._Layer):  # pylint: disable=protected-access
  """Batch Normalization layer from http://arxiv.org/abs/1502.03167.
  "Batch Normalization: Accelerating Deep Network Training by Reducing
  Internal Covariate Shift"
  Sergey Ioffe, Christian Szegedy
  Arguments:
    axis: Integer, the axis that should be normalized (typically the features
      axis). For instance, after a `Conv2D` layer with
      `data_format="channels_first"`, set `axis=1` in `BatchNormalization`.
    momentum: Momentum for the moving average.
    epsilon: Small float added to variance to avoid dividing by zero.
    center: If True, add offset of `beta` to normalized tensor. If False, `beta`
      is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    beta_initializer: Initializer for the beta weight.
    gamma_initializer: Initializer for the gamma weight.
    moving_mean_initializer: Initializer for the moving mean.
    moving_variance_initializer: Initializer for the moving variance.
    beta_regularizer: Optional regularizer for the beta weight.
    gamma_regularizer: Optional regularizer for the gamma weight.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    name: A string, the name of the layer.
    renorm: Whether to use Batch Renormalization
      (https://arxiv.org/abs/1702.03275). This adds extra variables during
      training. The inference is the same for either value of this parameter.
    renorm_clipping: A dictionary that may map keys 'rmax', 'rmin', 'dmax' to
      scalar `Tensors` used to clip the renorm correction. The correction
      `(r, d)` is used as `corrected_value = normalized_value * r + d`, with
      `r` clipped to [rmin, rmax], and `d` to [-dmax, dmax]. Missing rmax, rmin,
      dmax are set to inf, 0, inf, respectively.
    renorm_momentum: Momentum used to update the moving means and standard
      deviations with renorm. Unlike `momentum`, this affects training
      and should be neither too small (which would add noise) nor too large
      (which would give stale estimates). Note that `momentum` is still applied
      to get the means and variances for inference.
  """

  def __init__(self,
               axis=-1,
               momentum=0.99,
               epsilon=1e-3,
               center=True,
               scale=True,
               beta_initializer=init_ops.zeros_initializer(),
               gamma_initializer=init_ops.ones_initializer(),
               moving_mean_initializer=init_ops.zeros_initializer(),
               moving_variance_initializer=init_ops.ones_initializer(),
               beta_regularizer=None,
               gamma_regularizer=None,
               trainable=True,
               name=None,
               renorm=False,
               renorm_clipping=None,
               renorm_momentum=0.99,
               **kwargs):
    super(BatchNormalization, self).__init__(
        name=name, trainable=trainable, **kwargs)
    self.axis = axis
    self.momentum = momentum
    self.epsilon = epsilon
    self.center = center
    self.scale = scale
    self.beta_initializer = beta_initializer
    self.gamma_initializer = gamma_initializer
    self.moving_mean_initializer = moving_mean_initializer
    self.moving_variance_initializer = moving_variance_initializer
    self.beta_regularizer = beta_regularizer
    self.gamma_regularizer = gamma_regularizer
    self.renorm = renorm
    if renorm:
      renorm_clipping = renorm_clipping or {}
      keys = ['rmax', 'rmin', 'dmax']
      if set(renorm_clipping) - set(keys):
        raise ValueError('renorm_clipping %s contains keys not in %s' %
                         (renorm_clipping, keys))
      self.renorm_clipping = renorm_clipping
      self.renorm_momentum = renorm_momentum

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if not input_shape.ndims:
      raise ValueError('Input has undefined rank:', input_shape)
    ndim = len(input_shape)
    if self.axis < 0:
      axis = ndim + self.axis
    else:
      axis = self.axis
    if axis < 0 or axis >= ndim:
      raise ValueError('Value of `axis` argument ' + str(self.axis) +
                       ' is out of range for input with rank ' + str(ndim))
    param_dim = input_shape[axis]
    if not param_dim.value:
      raise ValueError('Input has undefined `axis` dimension. Input shape: ',
                       input_shape)

    if self.center:
      self.beta = variable_scope.get_variable('beta',
                                  shape=(param_dim,),
                                  initializer=self.beta_initializer,
                                  regularizer=self.beta_regularizer,
                                  trainable=True)
    else:
      self.beta = None
    if self.scale:
      self.gamma = variable_scope.get_variable('gamma',
                                   shape=(param_dim,),
                                   initializer=self.gamma_initializer,
                                   regularizer=self.gamma_regularizer,
                                   trainable=True)
    else:
      self.gamma = None

    # Disable variable partitioning when creating the moving mean and variance
    partitioner = variable_scope.get_variable_scope().partitioner
    try:
      variable_scope.get_variable_scope().set_partitioner(None)
      self.moving_mean = variable_scope.get_variable(
          'moving_mean',
          shape=(param_dim,),
          initializer=self.moving_mean_initializer,
          trainable=False)
      self.moving_variance = variable_scope.get_variable(
          'moving_variance',
          shape=(param_dim,),
          initializer=self.moving_variance_initializer,
          trainable=False)
      if self.renorm:
        # Create variables to maintain the moving mean and standard deviation.
        # These are used in training and thus are different from the moving
        # averages above. The renorm variables are colocated with moving_mean
        # and moving_variance.
        # NOTE: below, the outer `with device` block causes the current device
        # stack to be cleared. The nested ones use a `lambda` to set the desired
        # device and ignore any devices that may be set by the custom getter.
        def _renorm_variable(name, shape):
          var = variable_scope.get_variable(name,
                                shape=shape,
                                initializer=init_ops.zeros_initializer(),
                                trainable=False)
          return var
        with ops.device(None):
          with ops.device(lambda _: self.moving_mean.device):
            self.renorm_mean = _renorm_variable('renorm_mean', (param_dim,))
            self.renorm_mean_weight = _renorm_variable('renorm_mean_weight', ())
          # We initialize renorm_stddev to 0, and maintain the (0-initialized)
          # renorm_stddev_weight. This allows us to (1) mix the average
          # stddev with the minibatch stddev early in training, and (2) compute
          # the unbiased average stddev by dividing renorm_stddev by the weight.
          with ops.device(lambda _: self.moving_variance.device):
            self.renorm_stddev = _renorm_variable('renorm_stddev', (param_dim,))
            self.renorm_stddev_weight = _renorm_variable(
                'renorm_stddev_weight', ())
    finally:
      variable_scope.get_variable_scope().set_partitioner(partitioner)

  def _renorm_correction_and_moments(self, mean, variance, training):
    """Returns the correction and update values for renorm."""
    stddev = math_ops.sqrt(variance + self.epsilon)
    # Compute the average mean and standard deviation, as if they were
    # initialized with this batch's moments.
    mixed_renorm_mean = (self.renorm_mean +
                         (1. - self.renorm_mean_weight) * mean)
    mixed_renorm_stddev = (self.renorm_stddev +
                           (1. - self.renorm_stddev_weight) * stddev)
    # Compute the corrections for batch renorm.
    r = stddev / mixed_renorm_stddev
    d = (mean - mixed_renorm_mean) / mixed_renorm_stddev
    # Ensure the corrections use pre-update moving averages.
    with ops.control_dependencies([r, d]):
      mean = array_ops.identity(mean)
      stddev = array_ops.identity(stddev)
    rmin, rmax, dmax = [self.renorm_clipping.get(key)
                        for key in ['rmin', 'rmax', 'dmax']]
    if rmin is not None:
      r = math_ops.maximum(r, rmin)
    if rmax is not None:
      r = math_ops.minimum(r, rmax)
    if dmax is not None:
      d = math_ops.maximum(d, -dmax)
      d = math_ops.minimum(d, dmax)
    # When not training, use r=1, d=0, and decay=1 meaning no updates.
    r = _smart_select(training, lambda: r, lambda: array_ops.ones_like(r))
    d = _smart_select(training, lambda: d, lambda: array_ops.zeros_like(d))
    decay = _smart_select(training, lambda: self.renorm_momentum, lambda: 1.)
    def _update_renorm_variable(var, weight, value):
      """Updates a moving average and weight, returns the unbiased value."""
      # Update the variables without zero debiasing. The debiasing will be
      # accomplished by dividing the exponential moving average by the weight.
      # For example, after a single update, the moving average would be
      # (1-decay) * value. and the weight will be 1-decay, with their ratio
      # giving value.
      new_var = moving_averages.assign_moving_average(
          var, value, decay, zero_debias=False)
      new_weight = moving_averages.assign_moving_average(
          weight, 1., decay, zero_debias=False)
      return new_var / new_weight

    with ops.colocate_with(self.moving_mean):
      new_mean = _update_renorm_variable(self.renorm_mean,
                                         self.renorm_mean_weight,
                                         mean)
    with ops.colocate_with(self.moving_variance):
      new_stddev = _update_renorm_variable(self.renorm_stddev,
                                           self.renorm_stddev_weight,
                                           stddev)
      # Make sqrt(moving_variance + epsilon) = new_stddev.
      new_variance = math_ops.square(new_stddev) - self.epsilon

    return (r, d, new_mean, new_variance)

  def call(self, inputs, training=False):
    # First, compute the axes along which to reduce the mean / variance,
    # as well as the broadcast shape to be used for all parameters.
    input_shape = inputs.get_shape()
    ndim = len(input_shape)
    reduction_axes = list(range(len(input_shape)))
    del reduction_axes[self.axis]
    broadcast_shape = [1] * len(input_shape)
    broadcast_shape[self.axis] = input_shape[self.axis].value

    # Determines whether broadcasting is needed.
    needs_broadcasting = (sorted(reduction_axes) != list(range(ndim))[:-1])

    scale, offset = self.gamma, self.beta

    # Determine a boolean value for `training`: could be True, False, or None.
    training_value = utils.constant_value(training)
    if training_value is not False:
      # Some of the computations here are not necessary when training==False
      # but not a constant. However, this makes the code simpler.
      mean, variance = nn.moments(inputs, reduction_axes)
      if self.renorm:
        r, d, new_mean, new_variance = self._renorm_correction_and_moments(
            mean, variance, training)
        # When training, the normalized values (say, x) will be transformed as
        # x * gamma + beta without renorm, and (x * r + d) * gamma + beta
        # = x * (r * gamma) + (d * gamma + beta) with renorm.
        scale = array_ops.stop_gradient(r, name='renorm_r')
        offset = array_ops.stop_gradient(d, name='renorm_d')
        if self.gamma is not None:
          scale *= self.gamma
          offset *= self.gamma
        if self.beta is not None:
          offset += self.beta
      else:
        new_mean, new_variance = mean, variance

      # Update moving averages when training, and prevent updates otherwise.
      decay = _smart_select(training, lambda: self.momentum, lambda: 1.)
      mean_update = moving_averages.assign_moving_average(
          self.moving_mean, new_mean, decay, zero_debias=False)
      variance_update = moving_averages.assign_moving_average(
          self.moving_variance, new_variance, decay, zero_debias=False)

      if not self.updates:
        # In the future this should be refactored into a self.add_update
        # methods in order to allow for instance-based BN layer sharing
        # across unrelated input streams (e.g. like in Keras).
        self.updates.append(mean_update)
        self.updates.append(variance_update)

      mean = _smart_select(training,
                           lambda: mean,
                           lambda: self.moving_mean)
      variance = _smart_select(training,
                               lambda: variance,
                               lambda: self.moving_variance)

    else:
      mean, variance = self.moving_mean, self.moving_variance

    def _broadcast(v):
      if needs_broadcasting and v is not None:
        # In this case we must explictly broadcast all parameters.
        return array_ops.reshape(v, broadcast_shape)
      return v

    return nn.batch_normalization(inputs,
                                  _broadcast(mean),
                                  _broadcast(variance),
                                  _broadcast(offset),
                                  _broadcast(scale),
                                  self.epsilon)


def batch_normalization(inputs,
                        axis=-1,
                        momentum=0.99,
                        epsilon=1e-3,
                        center=True,
                        scale=True,
                        beta_initializer=init_ops.zeros_initializer(),
                        gamma_initializer=init_ops.ones_initializer(),
                        moving_mean_initializer=init_ops.zeros_initializer(),
                        moving_variance_initializer=init_ops.ones_initializer(),
                        beta_regularizer=None,
                        gamma_regularizer=None,
                        training=False,
                        trainable=True,
                        name=None,
                        reuse=None,
                        renorm=False,
                        renorm_clipping=None,
                        renorm_momentum=0.99):
  """Functional interface for the batch normalization layer.
  Reference: http://arxiv.org/abs/1502.03167
  "Batch Normalization: Accelerating Deep Network Training by Reducing
  Internal Covariate Shift"
  Sergey Ioffe, Christian Szegedy
  Arguments:
    inputs: Tensor input.
    axis: Integer, the axis that should be normalized (typically the features
      axis). For instance, after a `Convolution2D` layer with
      `data_format="channels_first"`, set `axis=1` in `BatchNormalization`.
    momentum: Momentum for the moving average.
    epsilon: Small float added to variance to avoid dividing by zero.
    center: If True, add offset of `beta` to normalized tensor. If False, `beta`
      is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    beta_initializer: Initializer for the beta weight.
    gamma_initializer: Initializer for the gamma weight.
    moving_mean_initializer: Initializer for the moving mean.
    moving_variance_initializer: Initializer for the moving variance.
    beta_regularizer: Optional regularizer for the beta weight.
    gamma_regularizer: Optional regularizer for the gamma weight.
    training: Either a Python boolean, or a TensorFlow boolean scalar tensor
      (e.g. a placeholder). Whether to return the output in training mode
      (normalized with statistics of the current batch) or in inference mode
      (normalized with moving statistics).
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    name: String, the name of the layer.
    reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.
    renorm: Whether to use Batch Renormalization
      (https://arxiv.org/abs/1702.03275). This adds extra variables during
      training. The inference is the same for either value of this parameter.
    renorm_clipping: A dictionary that may map keys 'rmax', 'rmin', 'dmax' to
      scalar `Tensors` used to clip the renorm correction. The correction
      `(r, d)` is used as `corrected_value = normalized_value * r + d`, with
      `r` clipped to [rmin, rmax], and `d` to [-dmax, dmax]. Missing rmax, rmin,
      dmax are set to inf, 0, inf, respectively.
    renorm_momentum: Momentum used to update the moving means and standard
      deviations with renorm. Unlike `momentum`, this affects training
      and should be neither too small (which would add noise) nor too large
      (which would give stale estimates). Note that `momentum` is still applied
      to get the means and variances for inference.
  Returns:
    Output tensor.
  """
  layer = BatchNormalization(
      axis=axis,
      momentum=momentum,
      epsilon=epsilon,
      center=center,
      scale=scale,
      beta_initializer=beta_initializer,
      gamma_initializer=gamma_initializer,
      moving_mean_initializer=moving_mean_initializer,
      moving_variance_initializer=moving_variance_initializer,
      beta_regularizer=beta_regularizer,
      gamma_regularizer=gamma_regularizer,
      trainable=trainable,
      renorm=renorm,
      renorm_clipping=renorm_clipping,
      renorm_momentum=renorm_momentum,
      name=name,
      _reuse=reuse,
      _scope=name)
  return layer.apply(inputs, training=training)


# Aliases

BatchNorm = BatchNormalization
batch_norm = batch_normalization


def _smart_select(pred, fn_then, fn_else):
  """Selects fn_then() or fn_else() based on the value of pred.
  The purpose of this function is the same as `utils.smart_cond`. However, at
  the moment there is a bug (b/36297356) that seems to kick in only when
  `smart_cond` delegates to `tf.cond`, which sometimes results in the training
  hanging when using parameter servers. This function will output the result
  of `fn_then` or `fn_else` if `pred` is known at graph construction time.
  Otherwise, it will use `tf.where` which will result in some redundant work
  (both branches will be computed but only one selected). However, the tensors
  involved will usually be small (means and variances in batchnorm), so the
  cost will be small and will not be incurred at all if `pred` is a constant.
  Args:
    pred: A boolean scalar `Tensor`.
    fn_then: A callable to use when pred==True.
    fn_else: A callable to use when pred==False.
  Returns:
    A `Tensor` whose value is fn_then() or fn_else() based on the value of pred.
  """
  pred_value = utils.constant_value(pred)
  if pred_value:
    return fn_then()
  elif pred_value is False:
    return fn_else()
  t_then = array_ops.expand_dims(fn_then(), 0)
  t_else = array_ops.expand_dims(fn_else(), 0)
  pred = array_ops.reshape(pred, [1])
  result = array_ops.where(pred, t_then, t_else)
  return array_ops.squeeze(result, [0])



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


