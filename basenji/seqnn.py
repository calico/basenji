# Copyright 2017 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
from __future__ import print_function

import gc
import pdb
import sys
import time

import numpy as np
import tensorflow as tf

from basenji.dna_io import hot1_augment
from basenji import seqnn_util
from basenji import layers


class SeqNN(seqnn_util.SeqNNModel):

  def __init__(self):
    tf.train.get_or_create_global_step()
    self.params_set = False

  def build(self, job, target_subset=None):
    """Build training ops that depend on placeholders."""

    self.set_params(job)
    self.params_set = True
    data_ops = self.make_placeholders()

    self.build_from_data_ops(job, data_ops, target_subset)

  def build_from_data_ops(self, job, data_ops, target_subset=None):
    """Build training ops from input data ops."""
    if not self.params_set:
      self.set_params(job)
      self.params_set = True
    self.targets = data_ops['label']
    self.inputs = data_ops['sequence']
    self.targets_na = data_ops['na']

    seqs_repr = self.build_representation(data_ops, target_subset)
    self.loss_op, self.loss_adhoc = self.build_loss(seqs_repr, data_ops, target_subset)
    self.build_optimizer(self.loss_op)

  def make_placeholders(self):
    """Allocates placeholders to be used in place of input data ops."""
    # batches
    self.inputs = tf.placeholder(
        tf.float32,
        shape=(self.batch_size, self.seq_length, self.seq_depth),
        name='inputs')
    if self.target_classes == 1:
      self.targets = tf.placeholder(
          tf.float32,
          shape=(self.batch_size, self.seq_length // self.target_pool,
                 self.num_targets),
          name='targets')
    else:
      self.targets = tf.placeholder(
          tf.int32,
          shape=(self.batch_size, self.seq_length // self.target_pool,
                 self.num_targets),
          name='targets')
    self.targets_na = tf.placeholder(
        tf.bool, shape=(self.batch_size, self.seq_length // self.target_pool))

    data = {
        'sequence': self.inputs,
        'label': self.targets,
        'na': self.targets_na
    }
    return data

  def _make_cnn_block_args(self, layer_index):
    """Packages arguments to be used by layers.cnn_block."""
    return {
        'cnn_filters': self.cnn_filters[layer_index],
        'cnn_filter_sizes': self.cnn_filter_sizes[layer_index],
        'cnn_dilation': self.cnn_dilation[layer_index],
        'cnn_strides': self.cnn_strides[layer_index],
        'is_training': self.is_training,
        'batch_norm' : self.batch_norm,
        'batch_norm_momentum' : self.batch_norm_momentum,
        'batch_renorm': self.batch_renorm,
        'batch_renorm_momentum' : self.batch_renorm_momentum,
        'renorm_clipping': self.renorm_clipping,
        'cnn_pool': self.cnn_pool[layer_index],
        'cnn_l2_scale': self.cnn_l2_scale[layer_index],
        'cnn_dropout_value': self.cnn_dropout[layer_index],
        'cnn_dropout_op': self.cnn_dropout_ph[layer_index],
        'cnn_dense': self.cnn_dense[layer_index],
        'name' : 'conv-%d' % layer_index
    }

  def build_representation(self, data_ops, target_subset):
    """Construct per-location real-valued predictions."""
    inputs = data_ops['sequence']
    assert inputs is not None

    print('Targets pooled by %d to length %d' %
          (self.target_pool, self.seq_length // self.target_pool))

    # dropout rates
    self.cnn_dropout_ph = []
    for layer_index in range(self.cnn_layers):
      self.cnn_dropout_ph.append(
          tf.placeholder(tf.float32, name='dropout_%d' % layer_index))

    # training conditional
    self.is_training = tf.placeholder(tf.bool, name='is_training')

    ###################################################
    # convolution layers
    ###################################################
    self.filter_weights = []
    self.layer_reprs = [inputs]

    seqs_repr = inputs
    for layer_index in range(self.cnn_layers):
      with tf.variable_scope('cnn%d' % layer_index) as vs:
        args_for_block = self._make_cnn_block_args(layer_index)
        seqs_repr = layers.cnn_block(seqs_repr=seqs_repr, **args_for_block)

        # save representation
        self.layer_reprs.append(seqs_repr)

    # update batch buffer to reflect pooling
    seq_length = seqs_repr.shape[1].value
    pool_preds = self.seq_length // seq_length
    assert self.batch_buffer % pool_preds == 0, (
        'batch_buffer %d not divisible'
        ' by the CNN pooling %d') % (self.batch_buffer, pool_preds)
    self.batch_buffer_pool = self.batch_buffer // pool_preds


    ###################################################
    # slice out side buffer
    ###################################################

    # predictions
    seq_length = seqs_repr.shape[1]
    seqs_repr = seqs_repr[:, self.batch_buffer_pool:
                          seq_length - self.batch_buffer_pool, :]
    seq_length = seqs_repr.shape[1].value
    self.preds_length = seq_length

    # save penultimate representation
    self.penultimate_op = seqs_repr


    ###################################################
    # final layer
    ###################################################
    with tf.variable_scope('final'):
      final_filters = self.num_targets * self.target_classes
      final_repr = tf.layers.dense(
        inputs=seqs_repr,
        units=final_filters,
        activation=None,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        kernel_regularizer=tf.contrib.layers.l1_regularizer(self.final_l1_scale))
      print('Convolution w/ %d %dx1 filters to final targets' %
            (final_filters, seqs_repr.shape[2]))

      if target_subset is not None:
        # get convolution parameters
        filters_full = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'final/dense/kernel')[0]
        bias_full = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'final/dense/bias')[0]

        # subset to specific targets
        filters_subset = tf.gather(filters_full, target_subset, axis=1)
        bias_subset = tf.gather(bias_full, target_subset, axis=0)

        # substitute a new limited convolution
        final_repr = tf.tensordot(seqs_repr, filters_subset, 1)
        final_repr = tf.nn.bias_add(final_repr, bias_subset)

        # update # targets
        self.num_targets = len(target_subset)

      # expand length back out
      if self.target_classes > 1:
        final_repr = tf.reshape(final_repr, (self.batch_size, -1, self.num_targets,
                                           self.target_classes))

    return final_repr

  def build_optimizer(self, loss_op):
    """Construct optimization op that minimizes loss_op."""
    # define optimization

    if self.optimization == 'adam':
      self.opt = tf.train.AdamOptimizer(
          self.learning_rate,
          beta1=self.adam_beta1,
          beta2=self.adam_beta2,
          epsilon=self.adam_eps)
    elif self.optimization == 'rmsprop':
      self.opt = tf.train.RMSPropOptimizer(
          self.learning_rate, decay=self.decay, momentum=self.momentum)
    elif self.optimization in ['sgd', 'momentum']:
      self.opt = tf.train.MomentumOptimizer(
          self.learning_rate, momentum=self.momentum)
    else:
      print('Cannot recognize optimization algorithm %s' % self.optimization)
      exit(1)

    # compute gradients
    self.gvs = self.opt.compute_gradients(
        loss_op,
        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

    # clip gradients
    if self.grad_clip is not None:
      gradients, variables = zip(*self.gvs)
      gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
      self.gvs = zip(gradients, variables)

    # apply gradients
    self.step_op = self.opt.apply_gradients(
        self.gvs, global_step=tf.train.get_or_create_global_step())

    self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # summary
    self.merged_summary = tf.summary.merge_all()

    # initialize steps
    self.step = 0

  def build_loss(self, seqs_repr, data_ops, target_subset=None):
    """Convert per-location real-valued predictions to a loss."""

    # targets
    tstart = self.batch_buffer // self.target_pool
    tend = (self.seq_length - self.batch_buffer) // self.target_pool
    self.target_length = tend - tstart

    targets = data_ops['label']
    targets = tf.identity(targets[:, tstart:tend, :], name='targets_op')

    if target_subset is not None:
      targets = tf.gather(targets, target_subset, axis=2)

    # work-around for specifying my own predictions
    self.preds_adhoc = tf.placeholder(
        tf.float32, shape=seqs_repr.shape, name='preds-adhoc')

    # choose link
    if self.link in ['identity', 'linear']:
      self.preds_op = tf.identity(seqs_repr, name='preds')

    elif self.link == 'relu':
      self.preds_op = tf.relu(seqs_repr, name='preds')

    elif self.link == 'exp':
      self.preds_op = tf.exp(tf.clip_by_value(seqs_repr, -50, 50), name='preds')

    elif self.link == 'exp_linear':
      self.preds_op = tf.where(
          seqs_repr > 0,
          seqs_repr + 1,
          tf.exp(tf.clip_by_value(seqs_repr, -50, 50)),
          name='preds')

    elif self.link == 'softplus':
      self.preds_op = tf.nn.softplus(tf.clip_by_value(seqs_repr, -50, 50),
                                     name='preds')

    elif self.link == 'softmax':
      # performed in the loss function, but saving probabilities
      self.preds_prob = tf.nn.softmax(seqs_repr, name='preds')

    else:
      print('Unknown link function %s' % self.link, file=sys.stderr)
      exit(1)

    # clip
    if self.target_clip is not None:
      self.preds_op = tf.clip_by_value(self.preds_op, 0, self.target_clip)
      targets = tf.clip_by_value(targets, 0, self.target_clip)

    # sqrt
    if self.target_sqrt:
      self.preds_op = tf.sqrt(self.preds_op)
      targets = tf.sqrt(targets)

    loss_op = None
    loss_adhoc = None
    loss_name = self.loss
    # choose loss
    if loss_name == 'gaussian':
      loss_op = tf.squared_difference(self.preds_op, targets)
      loss_adhoc = tf.squared_difference(self.preds_adhoc, targets)

    elif loss_name == 'poisson':
      loss_op = tf.nn.log_poisson_loss(
          targets, tf.log(self.preds_op), compute_full_loss=True)
      loss_adhoc = tf.nn.log_poisson_loss(
          targets, tf.log(self.preds_adhoc), compute_full_loss=True)

    elif loss_name == 'gamma':
      # jchan document
      loss_op = targets / self.preds_op + tf.log(self.preds_op)
      loss_adhoc = targets / self.preds_adhoc + tf.log(self.preds_adhoc)

    elif loss_name == 'cross_entropy':
      loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=(targets - 1), logits=self.preds_op)
      loss_adhoc = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=(targets - 1), logits=self.preds_adhoc)

    else:
      raise ValueError('Cannot identify loss function %s' % loss_name)

    # reduce lossses by batch and position
    loss_op = tf.reduce_mean(loss_op, axis=[0, 1], name='target_loss')
    loss_op = tf.check_numerics(loss_op, 'Invalid loss', name='loss_check')

    loss_adhoc = tf.reduce_mean(
        loss_adhoc, axis=[0, 1], name='target_loss_adhoc')
    tf.summary.histogram('target_loss', loss_op)
    for ti in np.linspace(0, self.num_targets - 1, 10).astype('int'):
      tf.summary.scalar('loss_t%d' % ti, loss_op[ti])
    self.target_losses = loss_op
    self.target_losses_adhoc = loss_adhoc

    # fully reduce
    loss_op = tf.reduce_mean(loss_op, name='loss')
    loss_adhoc = tf.reduce_mean(loss_adhoc, name='loss_adhoc')

    # add extraneous terms
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_sum = tf.reduce_sum(reg_losses)
    tf.summary.scalar('regularizers', reg_sum)
    loss_op += reg_sum
    loss_adhoc += reg_sum

    # track
    tf.summary.scalar('loss', loss_op)
    self.targets_op = targets
    return loss_op, loss_adhoc

  def drop_rate(self, drop_mult=0.5):
    """ Drop the optimizer learning rate. """
    self.opt._lr *= drop_mult

  def set_mode(self, mode):
    """ Construct a feed dictionary to specify the model's mode. """
    fd = {}

    if mode in ['train', 'training']:
      fd[self.is_training] = True
      for li in range(self.cnn_layers):
        fd[self.cnn_dropout_ph[li]] = self.cnn_dropout[li]

    elif mode in ['test', 'testing', 'evaluate']:
      fd[self.is_training] = False
      for li in range(self.cnn_layers):
        fd[self.cnn_dropout_ph[li]] = 0

    elif mode in [
        'test_mc', 'testing_mc', 'evaluate_mc', 'mc_test', 'mc_testing',
        'mc_evaluate'
    ]:
      fd[self.is_training] = False
      for li in range(self.cnn_layers):
        fd[self.cnn_dropout_ph[li]] = self.cnn_dropout[li]

    else:
      print('Cannot recognize mode %s' % mode)
      exit(1)

    return fd

  def set_params(self, job):
    """ Set model parameters. """

    ###################################################
    # data attributes
    ###################################################
    self.seq_depth = job.get('seq_depth', 4)
    self.num_targets = job['num_targets']
    self.target_classes = job.get('target_classes', 1)
    self.target_pool = job.get('target_pool', 1)

    ###################################################
    # batching
    ###################################################
    self.batch_size = job.get('batch_size', 64)
    self.seq_length = job.get('seq_length', 1024)
    self.batch_buffer = job.get('batch_buffer', 64)

    ###################################################
    # training
    ###################################################
    self.learning_rate = job.get('learning_rate', 0.001)
    self.adam_beta1 = job.get('adam_beta1', 0.9)
    self.adam_beta2 = job.get('adam_beta2', 0.999)
    self.adam_eps = job.get('adam_eps', 1e-8)
    self.momentum = job.get('momentum', 0)
    self.decay = job.get('decay', 0.9)
    self.optimization = job.get('optimization', 'adam').lower()
    self.grad_clip = job.get('grad_clip', 1)

    ###################################################
    # CNN params
    ###################################################
    self.cnn_filters = np.atleast_1d(job.get('cnn_filters', []))
    self.cnn_filter_sizes = np.atleast_1d(job.get('cnn_filter_sizes', []))
    self.cnn_layers = len(self.cnn_filters)

    self.cnn_pool = layer_extend(job.get('cnn_pool', []), 1, self.cnn_layers)
    self.cnn_strides = layer_extend(
        job.get('cnn_strides', []), 1, self.cnn_layers)
    self.cnn_dense = layer_extend(
        job.get('cnn_dense', []), False, self.cnn_layers)
    self.cnn_dilation = layer_extend(
        job.get('cnn_dilation', []), 1, self.cnn_layers)

    ###################################################
    # regularization
    ###################################################
    self.cnn_dropout = layer_extend(
        job.get('cnn_dropout', []), 0, self.cnn_layers)
    self.cnn_l2_scale = layer_extend(job.get('cnn_l2_scale', []), 0., self.cnn_layers)

    self.final_l1_scale = job.get('final_l1_scale', 0.)
    self.batch_norm = bool(job.get('batch_norm', True))
    self.batch_renorm = bool(job.get('batch_renorm', False))
    self.batch_renorm = bool(job.get('renorm', self.batch_renorm))

    self.batch_norm_momentum = job.get('batch_norm_momentum', 0.9)
    self.batch_renorm_momentum = job.get('batch_renorm_momentum', 0.9)

    ###################################################
    # loss
    ###################################################
    self.link = job.get('link', 'exp_linear')
    self.loss = job.get('loss', 'poisson')
    self.target_clip = job.get('target_clip', None)
    self.target_sqrt = bool(job.get('target_sqrt', False))


  def train_epoch(self,
                  sess,
                  batcher,
                  fwdrc=True,
                  shift=0,
                  sum_writer=None,
                  batches_per_epoch=0):
    """Execute one training epoch."""

    # initialize training loss
    train_loss = []

    # setup feed dict
    fd = self.set_mode('train')

    # get first batch
    Xb, Yb, NAb, Nb = batcher.next(fwdrc, shift)

    num_batches = 0
    while Xb is not None and Nb == self.batch_size and (
        batches_per_epoch == 0 or num_batches < batches_per_epoch):

      num_batches += 1
      # update feed dict
      fd[self.inputs] = Xb
      fd[self.targets] = Yb
      fd[self.targets_na] = NAb

      run_returns = sess.run(
          [self.merged_summary, self.loss_op, self.step_op] + self.update_ops,
          feed_dict=fd)
      summary, loss_batch = run_returns[:2]

      # add summary
      if sum_writer is not None:
        sum_writer.add_summary(summary, self.step)

      # accumulate loss
      # avail_sum = np.logical_not(NAb[:Nb,:]).sum()
      # train_loss.append(loss_batch / avail_sum)
      train_loss.append(loss_batch)

      # next batch
      Xb, Yb, NAb, Nb = batcher.next(fwdrc, shift)
      self.step += 1

    # reset training batcher if epoch considered all of the data
    if batches_per_epoch == 0:
      batcher.reset()

    return np.mean(train_loss), self.step

  def train_epoch_from_data_ops(self,
                                sess,
                                sum_writer=None,
                                batches_per_epoch=0):
    """ Execute one training epoch """
    assert batches_per_epoch > 0, ('not implemented for taking a full pass '
                                   'through the data')
    # initialize training loss
    train_loss = []

    # setup feed dict
    fd = self.set_mode('train')

    num_batches = 0
    while num_batches < batches_per_epoch:
      num_batches += 1
      run_returns = sess.run(
          [self.merged_summary, self.loss_op, self.step_op] + self.update_ops,
          feed_dict=fd)
      summary, loss_batch = run_returns[:2]

      # add summary
      if sum_writer is not None:
        sum_writer.add_summary(summary, self.step)

      train_loss.append(loss_batch)

      self.step += 1

    return np.mean(train_loss), self.step


def layer_extend(var, default, layers):
  """ Process job input to extend for the
         proper number of layers. """

  # if it's a number
  if type(var) != list:
    # change the default to that number
    default = var

    # make it a list
    var = [var]

  # extend for each layer
  while len(var) < layers:
    var.append(default)

  return var
