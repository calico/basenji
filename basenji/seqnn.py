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

import pdb
import sys
import time

import numpy as np
import tensorflow as tf

from basenji import layers
from basenji import params
from basenji import seqnn_util
from basenji import tfrecord_batcher

class SeqNN(seqnn_util.SeqNNModel):

  def __init__(self):
    self.global_step = tf.train.get_or_create_global_step()
    self.hparams_set = False

  def build(self, job, target_subset=None):
    """Build training ops that depend on placeholders."""

    self.hp = params.make_hparams(job)
    self.hparams_set = True
    data_ops = self.make_placeholders()

    self.build_from_data_ops(job, data_ops, target_subset=target_subset)

  def build_from_data_ops(self, job, data_ops,
                          augment_rc=False, augment_shifts=[],
                          target_subset=None):
    """Build training ops from input data ops."""
    if not self.hparams_set:
      self.hp = params.make_hparams(job)
      self.hparams_set = True
    self.targets = data_ops['label']
    self.inputs = data_ops['sequence']
    self.targets_na = data_ops['na']

    # training conditional
    self.is_training = tf.placeholder(tf.bool, name='is_training')

    # active only via basenji_train_queues.py for TFRecords
    if augment_rc or len(augment_shifts) > 0:
      # augment data ops
      data_ops_aug, _ = tfrecord_batcher.data_augmentation_from_data_ops(
          data_ops, augment_rc, augment_shifts)

      # condition on training
      data_ops = tf.cond(self.is_training, lambda: data_ops_aug, lambda: data_ops)

    seqs_repr = self.build_representation(data_ops, target_subset)
    self.loss_op, self.loss_adhoc = self.build_loss(seqs_repr, data_ops, target_subset)
    self.build_optimizer(self.loss_op)

  def make_placeholders(self):
    """Allocates placeholders to be used in place of input data ops."""
    # batches
    self.inputs = tf.placeholder(
        tf.float32,
        shape=(self.hp.batch_size, self.hp.seq_length, self.hp.seq_depth),
        name='inputs')
    if self.hp.target_classes == 1:
      self.targets = tf.placeholder(
          tf.float32,
          shape=(self.hp.batch_size, self.hp.seq_length // self.hp.target_pool,
                 self.hp.num_targets),
          name='targets')
    else:
      self.targets = tf.placeholder(
          tf.int32,
          shape=(self.hp.batch_size, self.hp.seq_length // self.hp.target_pool,
                 self.hp.num_targets),
          name='targets')
    self.targets_na = tf.placeholder(
        tf.bool, shape=(self.hp.batch_size, self.hp.seq_length // self.hp.target_pool))

    data = {
        'sequence': self.inputs,
        'label': self.targets,
        'na': self.targets_na
    }
    return data

  def _make_conv_block_args(self, layer_index):
    """Packages arguments to be used by layers.conv_block."""
    return {
        'conv_params': self.hp.cnn_params[layer_index],
        'is_training': self.is_training,
        'batch_norm': self.hp.batch_norm,
        'batch_norm_momentum': self.hp.batch_norm_momentum,
        'batch_renorm': self.hp.batch_renorm,
        'batch_renorm_momentum': self.hp.batch_renorm_momentum,
        'l2_scale': self.hp.cnn_l2_scale,
        'layer_reprs': self.layer_reprs,
        'name': 'conv-%d' % layer_index
    }

  def build_representation(self, data_ops, target_subset):
    """Construct per-location real-valued predictions."""
    inputs = data_ops['sequence']
    assert inputs is not None

    print('Targets pooled by %d to length %d' %
          (self.hp.target_pool, self.hp.seq_length // self.hp.target_pool))

    ###################################################
    # convolution layers
    ###################################################
    self.filter_weights = []
    self.layer_reprs = [inputs]

    seqs_repr = inputs
    for layer_index in range(self.hp.cnn_layers):
      with tf.variable_scope('cnn%d' % layer_index):
        # convolution block
        args_for_block = self._make_conv_block_args(layer_index)
        seqs_repr = layers.conv_block(seqs_repr=seqs_repr, **args_for_block)

        # save representation
        self.layer_reprs.append(seqs_repr)

    # final nonlinearity
    seqs_repr = tf.nn.relu(seqs_repr)

    # update batch buffer to reflect pooling
    seq_length = seqs_repr.shape[1].value
    pool_preds = self.hp.seq_length // seq_length
    assert self.hp.batch_buffer % pool_preds == 0, (
        'batch_buffer %d not divisible'
        ' by the CNN pooling %d') % (self.hp.batch_buffer, pool_preds)
    batch_buffer_pool = self.hp.batch_buffer // pool_preds


    ###################################################
    # slice out side buffer
    ###################################################

    # predictions
    seq_length = seqs_repr.shape[1]
    seqs_repr = seqs_repr[:, batch_buffer_pool:
                          seq_length - batch_buffer_pool, :]
    seq_length = seqs_repr.shape[1].value
    self.preds_length = seq_length

    # save penultimate representation
    self.penultimate_op = seqs_repr


    ###################################################
    # final layer
    ###################################################
    with tf.variable_scope('final'):
      final_filters = self.hp.num_targets * self.hp.target_classes
      final_repr = tf.layers.dense(
          inputs=seqs_repr,
          units=final_filters,
          activation=None,
          kernel_initializer=tf.contrib.layers.xavier_initializer(),
          kernel_regularizer=tf.contrib.layers.l1_regularizer(self.hp.final_l1_scale))
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
        self.hp.num_targets = len(target_subset)

      # expand length back out
      if self.hp.target_classes > 1:
        final_repr = tf.reshape(final_repr,
                                (self.hp.batch_size, -1, self.hp.num_targets,
                                 self.hp.target_classes))

    return final_repr

  def build_optimizer(self, loss_op):
    """Construct optimization op that minimizes loss_op."""

    # adaptive learning rate
    self.learning_rate_adapt = tf.train.exponential_decay(
        learning_rate=self.hp.learning_rate,
        global_step=self.global_step,
        decay_steps=self.hp.learning_decay_steps,
        decay_rate=self.hp.learning_decay_rate,
        staircase=True)
    tf.summary.scalar('learning_rate', self.learning_rate_adapt)

    if self.hp.optimizer == 'adam':
      self.opt = tf.train.AdamOptimizer(
          learning_rate=self.learning_rate_adapt,
          beta1=self.hp.adam_beta1,
          beta2=self.hp.adam_beta2,
          epsilon=self.hp.adam_eps)

    elif self.hp.optimizer == 'nadam':
      self.opt = tf.contrib.opt.NadamOptimizer(
          learning_rate=self.learning_rate_adapt,
          beta1=self.hp.adam_beta1,
          beta2=self.hp.adam_beta2,
          epsilon=self.hp.adam_eps)

    elif self.hp.optimizer in ['sgd', 'momentum']:
      self.opt = tf.train.MomentumOptimizer(
          learning_rate=self.learning_rate_adapt,
          momentum=self.hp.momentum)
    else:
      print('Cannot recognize optimization algorithm %s' % self.hp.optimizer)
      exit(1)

    # compute gradients
    self.gvs = self.opt.compute_gradients(
        loss_op,
        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

    # clip gradients
    if self.hp.grad_clip is not None:
      gradients, variables = zip(*self.gvs)
      gradients, _ = tf.clip_by_global_norm(gradients, self.hp.grad_clip)
      self.gvs = zip(gradients, variables)

    # apply gradients
    self.step_op = self.opt.apply_gradients(
        self.gvs, global_step=self.global_step)

    self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # summary
    self.merged_summary = tf.summary.merge_all()


  def build_loss(self, seqs_repr, data_ops, target_subset=None):
    """Convert per-location real-valued predictions to a loss."""

    # targets
    tstart = self.hp.batch_buffer // self.hp.target_pool
    tend = (self.hp.seq_length - self.hp.batch_buffer) // self.hp.target_pool
    self.target_length = tend - tstart

    targets = data_ops['label']
    targets = tf.identity(targets[:, tstart:tend, :], name='targets_op')

    if target_subset is not None:
      targets = tf.gather(targets, target_subset, axis=2)

    # work-around for specifying my own predictions
    self.preds_adhoc = tf.placeholder(
        tf.float32, shape=seqs_repr.shape, name='preds-adhoc')

    # float 32 exponential clip max
    # exp_max = np.floor(np.log(0.5*tf.float32.max))
    exp_max = 50

    # choose link
    if self.hp.link in ['identity', 'linear']:
      self.preds_op = tf.identity(seqs_repr, name='preds')

    elif self.hp.link == 'relu':
      self.preds_op = tf.relu(seqs_repr, name='preds')

    elif self.hp.link == 'exp':
      seqs_repr_clip = tf.clip_by_value(seqs_repr, -exp_max, exp_max)
      self.preds_op = tf.exp(seqs_repr_clip, name='preds')

    elif self.hp.link == 'exp_linear':
      self.preds_op = tf.where(
          seqs_repr > 0,
          seqs_repr + 1,
          tf.exp(tf.clip_by_value(seqs_repr, -exp_max, exp_max)),
          name='preds')

    elif self.hp.link == 'softplus':
      seqs_repr_clip = tf.clip_by_value(seqs_repr, -exp_max, 10000)
      self.preds_op = tf.nn.softplus(seqs_repr_clip, name='preds')

    elif self.hp.link == 'softmax':
      # performed in the loss function, but saving probabilities
      self.preds_prob = tf.nn.softmax(seqs_repr, name='preds')

    else:
      print('Unknown link function %s' % self.hp.link, file=sys.stderr)
      exit(1)

    # clip
    if self.hp.target_clip is not None:
      self.preds_op = tf.clip_by_value(self.preds_op, 0, self.hp.target_clip)
      targets = tf.clip_by_value(targets, 0, self.hp.target_clip)

    # sqrt
    if self.hp.target_sqrt:
      self.preds_op = tf.sqrt(self.preds_op)
      targets = tf.sqrt(targets)

    loss_op = None
    loss_adhoc = None

    # choose loss
    if self.hp.loss == 'gaussian':
      loss_op = tf.squared_difference(self.preds_op, targets)
      loss_adhoc = tf.squared_difference(self.preds_adhoc, targets)

    elif self.hp.loss == 'poisson':
      loss_op = tf.nn.log_poisson_loss(
          targets, tf.log(self.preds_op), compute_full_loss=True)
      loss_adhoc = tf.nn.log_poisson_loss(
          targets, tf.log(self.preds_adhoc), compute_full_loss=True)

    elif self.hp.loss == 'cross_entropy':
      loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=(targets - 1), logits=self.preds_op)
      loss_adhoc = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=(targets - 1), logits=self.preds_adhoc)

    else:
      raise ValueError('Cannot identify loss function %s' % self.hp.loss)

    # reduce lossses by batch and position
    loss_op = tf.reduce_mean(loss_op, axis=[0, 1], name='target_loss')
    loss_op = tf.check_numerics(loss_op, 'Invalid loss', name='loss_check')

    loss_adhoc = tf.reduce_mean(
        loss_adhoc, axis=[0, 1], name='target_loss_adhoc')
    tf.summary.histogram('target_loss', loss_op)
    for ti in np.linspace(0, self.hp.num_targets - 1, 10).astype('int'):
      tf.summary.scalar('loss_t%d' % ti, loss_op[ti])
    self.target_losses = loss_op
    self.target_losses_adhoc = loss_adhoc

    # fully reduce
    loss_op = tf.reduce_mean(loss_op, name='loss')
    loss_adhoc = tf.reduce_mean(loss_adhoc, name='loss_adhoc')

    # add regularization terms
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_sum = tf.reduce_sum(reg_losses)
    tf.summary.scalar('regularizers', reg_sum)
    loss_op += reg_sum
    loss_adhoc += reg_sum

    # track
    tf.summary.scalar('loss', loss_op)
    self.targets_op = targets
    return loss_op, loss_adhoc


  def set_mode(self, mode):
    """ Construct a feed dictionary to specify the model's mode. """
    fd = {}

    if mode in ['train', 'training']:
      fd[self.is_training] = True

    elif mode in ['test', 'testing', 'evaluate']:
      fd[self.is_training] = False

    elif mode in [
        'test_mc', 'testing_mc', 'evaluate_mc', 'mc_test', 'mc_testing',
        'mc_evaluate'
    ]:
      fd[self.is_training] = False

    else:
      print('Cannot recognize mode %s' % mode)
      exit(1)

    return fd

  def train_epoch(self,
                  sess,
                  batcher,
                  fwdrc=True,
                  shift=0,
                  sum_writer=None,
                  epoch_batches=None,
                  no_steps=False):
    """Execute one training epoch."""

    # initialize training loss
    train_loss = []
    global_step = 0

    # setup feed dict
    fd = self.set_mode('train')

    # get first batch
    Xb, Yb, NAb, Nb = batcher.next(fwdrc, shift)

    batch_num = 0
    while Xb is not None and Nb == self.hp.batch_size and (
        epoch_batches is None or batch_num < epoch_batches):

      # update feed dict
      fd[self.inputs] = Xb
      fd[self.targets] = Yb
      fd[self.targets_na] = NAb

      if no_steps:
        run_returns = sess.run([self.merged_summary, self.loss_op] + \
                                self.update_ops, feed_dict=fd)
        summary, loss_batch = run_returns[:2]
      else:
        run_returns = sess.run(
          [self.merged_summary, self.loss_op, self.global_step, self.step_op] + self.update_ops,
          feed_dict=fd)
        summary, loss_batch, global_step = run_returns[:3]

      # add summary
      if sum_writer is not None:
        sum_writer.add_summary(summary, global_step)

      # accumulate loss
      # avail_sum = np.logical_not(NAb[:Nb,:]).sum()
      # train_loss.append(loss_batch / avail_sum)
      train_loss.append(loss_batch)

      # next batch
      Xb, Yb, NAb, Nb = batcher.next(fwdrc, shift)
      batch_num += 1

    # reset training batcher if epoch considered all of the data
    if epoch_batches is None:
      batcher.reset()

    return np.mean(train_loss), global_step

  def train_epoch_from_data_ops(self,
                                sess,
                                sum_writer=None,
                                epoch_batches=None):
    """ Execute one training epoch """

    # initialize training loss
    train_loss = []
    global_step = 0

    # setup feed dict
    fd = self.set_mode('train')

    data_available = True
    batch_num = 0
    while data_available and (epoch_batches is None or batch_num < epoch_batches):
      try:
        run_returns = sess.run(
            [self.merged_summary, self.loss_op, self.global_step, self.step_op] + self.update_ops,
            feed_dict=fd)
        summary, loss_batch, global_step = run_returns[:3]

        # add summary
        if sum_writer is not None:
          sum_writer.add_summary(summary, global_step)

        # accumulate loss
        train_loss.append(loss_batch)

        # next batch
        batch_num += 1

      except tf.errors.OutOfRangeError:
        data_available = False

    return np.mean(train_loss), global_step
