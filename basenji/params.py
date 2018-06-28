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

from __future__ import print_function

import collections


import numpy as np
import tensorflow as tf

"""
params.py

Methods to handle model parameters.
"""


def layer_extend(var, default, layers):
  """Process job input to extend for the proper number of layers."""

  # if it's a number
  if not isinstance(var, list):
    # change the default to that number
    default = var

    # make it a list
    var = [var]

  # extend for each layer
  while len(var) < layers:
    var.append(default)

  return var


def read_job_params(job_file):
  """Read job parameters from text table."""

  job = {}

  if job_file is not None:
    for line in open(job_file):
      if line.strip():
        param, val = line.split()

        # require a decimal for floats
        try:
          if val.find('e') != -1:
            val = float(val)
          elif val.find('.') == -1:
            val = int(val)
          else:
            val = float(val)
        except ValueError:
          pass

        if param in job:
          # change to a list
          if not isinstance(job[param], list):
            job[param] = [job[param]]

          # append new value
          job[param].append(val)
        else:
          job[param] = val

    print(job)

  return job


def make_hparams(job, num_worker_replicas=None, num_ps_replicas=None):
  """Convert the parsed job args to an params object.

  Args:
    job: a dictionary of parsed parameters.
      See `basenji.google.params.read_job_params` for more information.
    num_worker_replicas: the number of worker replicas, e.g.
      http://google3/learning/brain/contrib/learn/learn.borg?l=112&rcl=174372550
    num_ps_replicas: the number of ps replicas, e.g.
      http://google3/learning/brain/contrib/learn/learn.borg?l=113&rcl=174372550
  """

  hp = tf.contrib.training.HParams()

  ###################################################
  # data

  hp.add_hparam('seq_depth', job.get('seq_depth', 4))
  hp.add_hparam('num_targets', job['num_targets'])
  hp.add_hparam('target_classes', job.get('target_classes', 1))
  hp.add_hparam('target_pool', job.get('target_pool', 1))

  hp.add_hparam('hic', job.get('hic', False))

  ###################################################
  # batching

  hp.add_hparam('batch_size', job.get('batch_size', 64))
  hp.add_hparam('seq_length', job.get('seq_length', 1024))
  hp.add_hparam('batch_buffer', job.get('batch_buffer', 64))

  hp.add_hparam('batch_norm', bool(job.get('batch_norm', True)))
  hp.add_hparam('batch_renorm', bool(job.get('batch_renorm', False)))
  hp.add_hparam('batch_norm_momentum', 0.9)
  hp.add_hparam('batch_renorm_momentum', 0.9)

  ###################################################
  # training

  optimizer = job.get('optimizer', 'nadam')
  optimizer = job.get('optimization', optimizer)
  hp.add_hparam('optimizer', optimizer.lower())

  hp.add_hparam('learning_rate', job.get('learning_rate', 0.001))
  hp.add_hparam('momentum', job.get('momentum', 0))

  hp.add_hparam('learning_decay_steps', job.get('learning_decay_steps', 200000))
  hp.add_hparam('learning_decay_rate', job.get('learning_decay_rate', 0.2))

  hp.add_hparam('adam_beta1', job.get('adam_beta1', 0.9))
  hp.add_hparam('adam_beta2', job.get('adam_beta2', 0.999))
  hp.add_hparam('adam_eps', job.get('adam_eps', 1e-8))

  hp.add_hparam('grad_clip', job.get('grad_clip', 1.0))

  hp.add_hparam('cnn_l2_scale', job.get('cnn_l2_scale', 0.))
  hp.add_hparam('final_l1_scale', job.get('final_l1_scale', 0.))

  ###################################################
  # loss

  link = job.get('link', 'softplus')
  link = job.get('link_function', link)
  hp.add_hparam('link', link)

  loss = job.get('loss', 'poisson')
  loss = job.get('loss_name', loss)
  hp.add_hparam('loss', loss)

  hp.add_hparam('target_clip', job.get('target_clip', None))
  hp.add_hparam('target_sqrt', bool(job.get('target_sqrt', False)))

  ###################################################
  # architecture

  hp.add_hparam('architecture', job.get('architecture', 'cnn'))

  if hp.architecture in ['dres', 'dilated_residual']:
    add_hparams_dres(hp, job)
  else:
    add_hparams_cnn(hp, job)

  # transform CNN hparams to specific params
  add_cnn_params(hp)

  ###################################################
  # google3

  hp.add_hparam('augment_with_complement',
                job.get('augment_with_complement', False))
  hp.add_hparam('shift_augment_offsets', job.get('shift_augment_offsets', None))

  hp.add_hparam('ensemble_during_training',
                job.get('ensemble_during_training', False))
  hp.add_hparam('ensemble_during_prediction',
                job.get('ensemble_during_prediction', False))

  hp.add_hparam('num_plateau_steps', job.get('num_plateau_steps', 5000))

  hp.add_hparam('plateau_delta', job.get('plateau_delta', 0.05))

  hp.add_hparam('stop_early', job.get('stop_early', False))

  hp.add_hparam('stop_early_num_plateau_steps',
                job.get('stop_early_num_plateau_steps', 10000))

  hp.add_hparam('stop_early_plateau_delta',
                job.get('stop_early_plateau_delta', 0.03))

  # If True, collapse into a single per-sequence feature vector by mean pooling.
  hp.add_hparam('pool_by_averaging', job.get('pool_by_averaging', False))

  # If True, unfold into features of size length * channels.
  hp.add_hparam('pool_by_unfolding', job.get('pool_by_unfolding', False))

  if hp.pool_by_averaging and hp.pool_by_unfolding:
    raise ValueError('It is invalid to specify both pool_by_averaging'
                     ' and pool_by_unfolding')

  tf.logging.info('Parsed params from job argument, and got %s',
                  str(hp.values()))

  hp.add_hparam('num_worker_replicas', num_worker_replicas)
  hp.add_hparam('num_ps_replicas', num_ps_replicas)

  return hp


def add_hparams_cnn(params, job):
  """Add CNN hyper-parameters for a standard verbose CNN definition."""

  # establish layer number using filters
  params.add_hparam('cnn_filters',
                    layer_extend(job.get('cnn_filters', []), 16, 1))
  layers = len(params.cnn_filters)
  params.cnn_layers = layers

  # get remainder, or set to default
  params.add_hparam('cnn_filter_sizes',
                    layer_extend(job.get('cnn_filter_sizes', []), 1, layers))
  params.add_hparam('cnn_stride',
                    layer_extend(job.get('cnn_stride', []), 1, layers))
  params.add_hparam('cnn_pool',
                    layer_extend(job.get('cnn_pool', []), 1, layers))
  params.add_hparam('cnn_dense',
                    layer_extend(job.get('cnn_dense', []), False, layers))
  params.add_hparam('cnn_dilation',
                    layer_extend(job.get('cnn_dilation', []), 1, layers))
  params.add_hparam('cnn_skip',
                    layer_extend(job.get('cnn_skip', []), 0, layers))
  params.add_hparam('cnn_dropout',
                    layer_extend(job.get('cnn_dropout', []), 0., layers))

  # g3 dropout parameterization
  if 'non_dilated_cnn_dropout' in job and 'dilated_cnn_dropout' in job:
    params.cnn_dropout = []
    for ci in range(layers):
      if params.cnn_dilation[ci] > 1:
        params.cnn_dropout.append(job['dilated_cnn_dropout'])
      else:
        params.cnn_dropout.append(job['non_dilated_cnn_dropout'])


def add_hparams_dres(params, job):
  """Add CNN hyper-parameters for a dilated residual network."""

  # DNA
  params.add_hparam('conv_dna_filters', job.get('conv_dna_filters', 256))
  params.add_hparam('conv_dna_filter_size', job.get('conv_dna_filter_size', 13))
  params.add_hparam('conv_dna_stride', job.get('conv_dna_stride', 1))
  params.add_hparam('conv_dna_pool', job.get('conv_dna_pool', 2))
  params.add_hparam('conv_dna_dropout', job.get('conv_dna_dropout', 0.))

  # reduce
  params.add_hparam('conv_reduce_filters_mult', job.get('conv_reduce_filters_mult', 1.25))
  params.add_hparam('conv_reduce_filter_size', job.get('conv_reduce_filter_size', 6))
  params.add_hparam('conv_reduce_stride', job.get('conv_reduce_stride', 1))
  params.add_hparam('conv_reduce_pool', job.get('conv_reduce_pool', 2))
  params.add_hparam('conv_reduce_dropout', job.get('conv_reduce_dropout', 0.))

  params.add_hparam('conv_reduce_width_max', job.get('conv_reduce_width_max', 128))

  # dilated residual
  params.add_hparam('conv_dilate_filters', job.get('conv_dilate_filters', 128))
  params.add_hparam('conv_dilate_rate_mult', job.get('conv_dilate_rate_mult', 2))
  params.add_hparam('conv_dilate_rate_max', job.get('conv_dilate_rate_max', 64))
  params.add_hparam('conv_dilate_dropout', job.get('conv_dilate_dropout', 0.))

  # final
  params.add_hparam('conv_final_filters', job.get('conv_final_filters', 1024))
  params.add_hparam('conv_final_dropout', job.get('conv_final_dropout', 0.))


def add_cnn_params(params):
  """Define CNN params list."""
  if params.architecture in ['dres', 'dilated_residual']:
    add_cnn_params_dres(params)
  else:
    add_cnn_params_cnn(params)


def add_cnn_params_cnn(params):
  """Layer-by-layer CNN parameter mode."""

  params.cnn_params = []
  for ci in range(params.cnn_layers):
    cp = ConvParams(
        filters=params.cnn_filters[ci],
        filter_size=params.cnn_filter_sizes[ci],
        stride=params.cnn_stride[ci],
        pool=params.cnn_pool[ci],
        dropout=params.cnn_dropout[ci],
        dense=params.cnn_dense[ci],
        skip_layers=params.cnn_skip[ci],
        dilation=params.cnn_dilation[ci])
    params.cnn_params.append(cp)


def add_cnn_params_dres(params):
  """Dilated resnet parameter mode.

  Consists of four phases:
   (1) DNA - Initial layer to interact directly with DNA sequence.
   (2) Reduce - Convolution blocks to reduce the sequence length.
   (3) Dilate - Residual dilated convolutions to propagate information
                 across the sequence.
   (4) Final - Final convolution before transforming to predictions.

   Returns:
     [ConvParams]
  """

  params.cnn_params = []

  ###################################
  # DNA

  reduce_width = 1

  cp = ConvParams(
      filters=params.conv_dna_filters,
      filter_size=params.conv_dna_filter_size,
      stride=params.conv_dna_stride,
      pool=params.conv_dna_pool,
      dropout=params.conv_dna_dropout)
  params.cnn_params.append(cp)

  reduce_width *= (cp.pool*cp.stride)
  current_filters = cp.filters

  ###################################
  # reduce

  while reduce_width < params.conv_reduce_width_max:
    current_filters = int(current_filters*params.conv_reduce_filters_mult)

    cp = ConvParams(
        filters=current_filters,
        filter_size=params.conv_reduce_filter_size,
        stride=params.conv_reduce_stride,
        pool=params.conv_reduce_pool,
        dropout=params.conv_reduce_dropout)
    params.cnn_params.append(cp)

    reduce_width *= cp.pool*cp.stride

  ###################################
  # dilated residual

  current_dilation = 1
  while current_dilation <= params.conv_dilate_rate_max:
    # dilate
    cp = ConvParams(
        filters=params.conv_dilate_filters,
        filter_size=3,
        dilation=current_dilation)
    params.cnn_params.append(cp)

    # residual
    cp = ConvParams(
        filters=current_filters,
        filter_size=1,
        skip_layers=2,
        dropout=params.conv_dilate_dropout)
    params.cnn_params.append(cp)

    current_dilation *= params.conv_dilate_rate_mult
    current_dilation = int(np.round(current_dilation))

  ###################################
  # final

  cp = ConvParams(
      filters=params.conv_final_filters,
      dropout=params.conv_final_dropout)
  params.cnn_params.append(cp)

  params.cnn_layers = len(params.cnn_params)


class ConvParams(
    collections.namedtuple('ConvParams',
                           ['filters', 'filter_size', 'stride', 'pool',
                            'dilation', 'dropout', 'skip_layers', 'dense'])):
  """Convolution block parameters.

  Args:
    filters: number of convolution filters.
    filter_size: convolution filter size.
    stride: convolution stride.
    pool: max pool width.
    dilation: convolution dilation rate.
    dropout: dropout rate.
    skip_layers: add block result to preceding layer.
    dense: concat block result to preceding layer.
  """
  def __new__(cls, filters=1, filter_size=1, stride=1, pool=1,
              dilation=1, dropout=0, skip_layers=0, dense=False):
    return super(ConvParams, cls).__new__(cls, filters, filter_size,
                                          stride, pool, dilation,
                                          dropout, skip_layers, dense)
