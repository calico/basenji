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
try:
  import tensorflow_probability as tfp
except ImportError:
  pass

from basenji import augmentation
from basenji import layers
from basenji import params
from basenji import seqnn_util
from basenji import tfrecord_batcher

class SeqNN(seqnn_util.SeqNNModel):

  def __init__(self):
    self.global_step = tf.train.get_or_create_global_step()
    self.hparams_set = False

  def build_feed(self, job, augment_rc=False, augment_shifts=[0],
                 ensemble_rc=False, ensemble_shifts=[0],
                 embed_penultimate=False, target_subset=None):
    """Build training ops that depend on placeholders."""

    self.hp = params.make_hparams(job)
    self.hparams_set = True
    data_ops = self.make_placeholders()

    self.build_from_data_ops(job, data_ops,
          augment_rc=augment_rc,
          augment_shifts=augment_shifts,
          ensemble_rc=ensemble_rc,
          ensemble_shifts=ensemble_shifts,
          embed_penultimate=embed_penultimate,
          target_subset=target_subset)

  def build_feed_sad(self, job,
                     ensemble_rc=False, ensemble_shifts=[0],
                     embed_penultimate=False, target_subset=None):
    """Build SAD predict ops that depend on placeholders."""

    self.hp = params.make_hparams(job)
    self.hparams_set = True
    data_ops = self.make_placeholders()

    self.build_sad(job, data_ops,
                   ensemble_rc=ensemble_rc,
                   ensemble_shifts=ensemble_shifts,
                   embed_penultimate=embed_penultimate,
                   target_subset=target_subset)

  def build_from_data_ops(self, job, data_ops,
                          augment_rc=False, augment_shifts=[0],
                          ensemble_rc=False, ensemble_shifts=[0],
                          embed_penultimate=False, target_subset=None):
    """Build training ops from input data ops."""
    if not self.hparams_set:
      self.hp = params.make_hparams(job)
      self.hparams_set = True

    # training conditional
    self.is_training = tf.placeholder(tf.bool, name='is_training')

    ##################################################
    # training

    # training data_ops w/ stochastic augmentation
    data_ops_train = augmentation.augment_stochastic(
        data_ops, augment_rc, augment_shifts)

    # compute train representation
    self.preds_train = self.build_predict(data_ops_train['sequence'],
                                          None, embed_penultimate, target_subset,
                                          save_reprs=True)
    self.target_length = self.preds_train.shape[1].value

    # training losses
    if not embed_penultimate:
      loss_returns = self.build_loss(self.preds_train, data_ops_train['label'], target_subset)
      self.loss_train, self.loss_train_targets, self.targets_train = loss_returns

      # optimizer
      self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      self.build_optimizer(self.loss_train)

      # allegedly correct, but outperformed by skipping
      # with tf.control_dependencies(self.update_ops):
      #   self.build_optimizer(self.loss_train)


    ##################################################
    # eval

    # eval data ops w/ deterministic augmentation
    data_ops_eval = augmentation.augment_deterministic_set(
        data_ops, ensemble_rc, ensemble_shifts)
    data_seq_eval = tf.stack([do['sequence'] for do in data_ops_eval])
    data_rev_eval = tf.stack([do['reverse_preds'] for do in data_ops_eval])

    # compute eval representation
    map_elems_eval = (data_seq_eval, data_rev_eval)
    build_rep = lambda do: self.build_predict(do[0], do[1], embed_penultimate, target_subset)
    self.preds_ensemble = tf.map_fn(build_rep, map_elems_eval, dtype=tf.float32, back_prop=False)
    self.preds_eval = tf.reduce_mean(self.preds_ensemble, axis=0)

    # eval loss
    if not embed_penultimate:
      loss_returns = self.build_loss(self.preds_eval, data_ops['label'], target_subset)
      self.loss_eval, self.loss_eval_targets, self.targets_eval = loss_returns

    # update # targets
    if target_subset is not None:
      self.hp.num_targets = len(target_subset)

    # helper variables
    self.preds_length = self.preds_train.shape[1]

  def build_sad(self, job, data_ops,
                ensemble_rc=False, ensemble_shifts=[0],
                embed_penultimate=False, target_subset=None):
    """Build SAD predict ops."""
    if not self.hparams_set:
      self.hp = params.make_hparams(job)
      self.hparams_set = True

    # training conditional
    self.is_training = tf.placeholder(tf.bool, name='is_training')

    # eval data ops w/ deterministic augmentation
    data_ops_eval = augmentation.augment_deterministic_set(
        data_ops, ensemble_rc, ensemble_shifts)
    data_seq_eval = tf.stack([do['sequence'] for do in data_ops_eval])
    data_rev_eval = tf.stack([do['reverse_preds'] for do in data_ops_eval])

    # compute eval representation
    map_elems_eval = (data_seq_eval, data_rev_eval)
    build_rep = lambda do: self.build_predict(do[0], do[1], embed_penultimate, target_subset)
    self.preds_ensemble = tf.map_fn(build_rep, map_elems_eval, dtype=tf.float32, back_prop=False)
    self.preds_eval = tf.reduce_mean(self.preds_ensemble, axis=0)

    # update # targets
    if target_subset is not None:
      self.hp.num_targets = len(target_subset)

    # helper variables
    self.preds_length = self.preds_eval.shape[1]

  def make_placeholders(self):
    """Allocates placeholders to be used in place of input data ops."""
    # batches
    self.inputs_ph = tf.placeholder(
        tf.float32,
        shape=(None, self.hp.seq_length, self.hp.seq_depth),
        name='inputs')

    if self.hp.target_classes == 1:
      self.targets_ph = tf.placeholder(
          tf.float32,
          shape=(None, self.hp.seq_length // self.hp.target_pool,
                 self.hp.num_targets),
          name='targets')
    else:
      self.targets_ph = tf.placeholder(
          tf.int32,
          shape=(None, self.hp.seq_length // self.hp.target_pool,
                 self.hp.num_targets),
          name='targets')

    self.targets_na_ph = tf.placeholder(tf.bool,
        shape=(None, self.hp.seq_length // self.hp.target_pool),
        name='targets_na')

    data = {
        'sequence': self.inputs_ph,
        'label': self.targets_ph,
        'na': self.targets_na_ph
    }
    return data

  def _make_conv_block_args(self, layer_index, layer_reprs):
    """Packages arguments to be used by layers.conv_block."""
    return {
        'conv_params': self.hp.cnn_params[layer_index],
        'is_training': self.is_training,
        'nonlinearity': self.hp.nonlinearity,
        'batch_norm': self.hp.batch_norm,
        'batch_norm_momentum': self.hp.batch_norm_momentum,
        'batch_renorm': self.hp.batch_renorm,
        'batch_renorm_momentum': self.hp.batch_renorm_momentum,
        'l2_scale': self.hp.cnn_l2_scale,
        'layer_reprs': layer_reprs,
        'name': 'conv-%d' % layer_index
    }

  def build_predict(self, inputs, reverse_preds=None, embed_penultimate=False, target_subset=None, save_reprs=False):
    """Construct per-location real-valued predictions."""
    assert inputs is not None
    print('Targets pooled by %d to length %d' %
          (self.hp.target_pool, self.hp.seq_length // self.hp.target_pool))

    if self.hp.augment_mutation > 0:
      # sample mutation binary mask across sequences
      mut_mask_probs = self.hp.augment_mutation*np.ones((self.hp.seq_length,1))
      mut_mask_dist = tfp.distributions.Bernoulli(probs=mut_mask_probs, dtype=tf.float32)
      mut_mask = mut_mask_dist.sample(tf.shape(inputs)[0])

      # sample random nucleotide for mutations
      mut_1hot_probs = 0.25*np.ones((self.hp.seq_length,4))
      mut_1hot_dist = tfp.distributions.OneHotCategorical(probs=mut_1hot_probs, dtype=tf.float32)
      mut_1hot = mut_1hot_dist.sample(tf.shape(inputs)[0])

      # modify sequence
      inputs_mut = inputs - mut_mask*inputs + mut_mask*mut_1hot
      inputs = tf.cond(self.is_training, lambda: inputs_mut, lambda: inputs)

    ###################################################
    # convolution layers
    ###################################################
    filter_weights = []
    layer_reprs = [inputs]

    seqs_repr = inputs
    for layer_index in range(self.hp.cnn_layers):
      with tf.variable_scope('cnn%d' % layer_index, reuse=tf.AUTO_REUSE):
        # convolution block
        args_for_block = self._make_conv_block_args(layer_index, layer_reprs)
        seqs_repr = layers.conv_block(seqs_repr=seqs_repr, **args_for_block)

        # save representation
        layer_reprs.append(seqs_repr)

    if save_reprs:
      self.layer_reprs = layer_reprs

    # final nonlinearity
    if self.hp.nonlinearity == 'relu':
      seqs_repr = tf.nn.relu(seqs_repr)
    elif self.hp.nonlinearity == 'gelu':
      seqs_repr = tf.nn.sigmoid(1.702 * seqs_repr) * seqs_repr
    else:
      print('Unrecognized nonlinearity "%s"' % self.hp.nonlinearity, file=sys.stderr)
      exit(1)

    ###################################################
    # slice out side buffer
    ###################################################

    # update batch buffer to reflect pooling
    seq_length = seqs_repr.shape[1].value
    pool_preds = self.hp.seq_length // seq_length
    assert self.hp.batch_buffer % pool_preds == 0, (
        'batch_buffer %d not divisible'
        ' by the CNN pooling %d') % (self.hp.batch_buffer, pool_preds)
    batch_buffer_pool = self.hp.batch_buffer // pool_preds

    # slice out buffer
    seq_length = seqs_repr.shape[1]
    seqs_repr = seqs_repr[:, batch_buffer_pool:
                          seq_length - batch_buffer_pool, :]
    seq_length = seqs_repr.shape[1]

    ###################################################
    # final layer
    ###################################################
    if embed_penultimate:
      final_repr = seqs_repr
    else:
      with tf.variable_scope('final', reuse=tf.AUTO_REUSE):
        final_filters = self.hp.num_targets * self.hp.target_classes
        final_repr = tf.layers.dense(
            inputs=seqs_repr,
            units=final_filters,
            activation=None,
            kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_in'),
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

        # expand length back out
        if self.hp.target_classes > 1:
          final_repr = tf.reshape(final_repr,
                                  (-1, seq_length, self.hp.num_targets,
                                   self.hp.target_classes))

    # transform for reverse complement
    if reverse_preds is not None:
      final_repr = tf.cond(reverse_preds,
                           lambda: tf.reverse(final_repr, axis=[1]),
                           lambda: final_repr)

    ###################################################
    # link function
    ###################################################
    if embed_penultimate:
      predictions = final_repr
    else:
      # work-around for specifying my own predictions
      # self.preds_adhoc = tf.placeholder(
      #     tf.float32, shape=final_repr.shape, name='preds-adhoc')

      # float 32 exponential clip max
      exp_max = 50

      # choose link
      if self.hp.link in ['identity', 'linear']:
        predictions = tf.identity(final_repr, name='preds')

      elif self.hp.link == 'relu':
        predictions = tf.relu(final_repr, name='preds')

      elif self.hp.link == 'exp':
        final_repr_clip = tf.clip_by_value(final_repr, -exp_max, exp_max)
        predictions = tf.exp(final_repr_clip, name='preds')

      elif self.hp.link == 'exp_linear':
        predictions = tf.where(
            final_repr > 0,
            final_repr + 1,
            tf.exp(tf.clip_by_value(final_repr, -exp_max, exp_max)),
            name='preds')

      elif self.hp.link == 'softplus':
        final_repr_clip = tf.clip_by_value(final_repr, -exp_max, 10000)
        predictions = tf.nn.softplus(final_repr_clip, name='preds')

      else:
        print('Unknown link function %s' % self.hp.link, file=sys.stderr)
        exit(1)

      # clip
      if self.hp.target_clip is not None:
        predictions = tf.clip_by_value(predictions, 0, self.hp.target_clip)

      # sqrt
      if self.hp.target_sqrt:
        predictions = tf.sqrt(predictions)

    return predictions

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

    # summary
    self.merged_summary = tf.summary.merge_all()


  def build_loss(self, preds, targets, target_subset=None):
    """Convert per-location real-valued predictions to a loss."""

    # slice buffer
    tstart = self.hp.batch_buffer // self.hp.target_pool
    tend = (self.hp.seq_length - self.hp.batch_buffer) // self.hp.target_pool
    targets = tf.identity(targets[:, tstart:tend, :], name='targets_op')

    if target_subset is not None:
      targets = tf.gather(targets, target_subset, axis=2)

    # clip
    if self.hp.target_clip is not None:
      targets = tf.clip_by_value(targets, 0, self.hp.target_clip)

    # sqrt
    if self.hp.target_sqrt:
      targets = tf.sqrt(targets)

    loss_op = None

    # choose loss
    if self.hp.loss == 'gaussian':
      loss_op = tf.squared_difference(preds, targets)

    elif self.hp.loss == 'poisson':
      loss_op = tf.nn.log_poisson_loss(
          targets, tf.log(preds), compute_full_loss=True)

    elif self.hp.loss == 'cross_entropy':
      loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=(targets - 1), logits=preds)

    else:
      raise ValueError('Cannot identify loss function %s' % self.hp.loss)

    # reduce lossses by batch and position
    loss_op = tf.reduce_mean(loss_op, axis=[0, 1], name='target_loss')
    loss_op = tf.check_numerics(loss_op, 'Invalid loss', name='loss_check')
    target_losses = loss_op

    if target_subset is None:
      tf.summary.histogram('target_loss', loss_op)
      for ti in np.linspace(0, self.hp.num_targets - 1, 10).astype('int'):
        tf.summary.scalar('loss_t%d' % ti, loss_op[ti])

    # fully reduce
    loss_op = tf.reduce_mean(loss_op, name='loss')

    # add regularization terms
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_sum = tf.reduce_sum(reg_losses)
    tf.summary.scalar('regularizers', reg_sum)
    loss_op += reg_sum

    # track
    tf.summary.scalar('loss', loss_op)

    return loss_op, target_losses, targets


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

  def train_epoch_h5_manual(self,
                  sess,
                  batcher,
                  fwdrc=True,
                  shift=0,
                  sum_writer=None,
                  epoch_batches=None,
                  no_steps=False):
    """Execute one training epoch, using HDF5 data
       and manual augmentation."""

    # initialize training loss
    train_loss = []
    batch_sizes = []
    global_step = 0

    # setup feed dict
    fd = self.set_mode('train')

    # get first batch
    Xb, Yb, NAb, Nb = batcher.next(fwdrc, shift)

    batch_num = 0
    while Xb is not None and (epoch_batches is None or batch_num < epoch_batches):
      # update feed dict
      fd[self.inputs_ph] = Xb
      fd[self.targets_ph] = Yb

      if no_steps:
        run_returns = sess.run([self.merged_summary, self.loss_train] + \
                                self.update_ops, feed_dict=fd)
        summary, loss_batch = run_returns[:2]
      else:
        run_returns = sess.run(
          [self.merged_summary, self.loss_train, self.global_step, self.step_op] + self.update_ops,
          feed_dict=fd)
        summary, loss_batch, global_step = run_returns[:3]

      # add summary
      if sum_writer is not None:
        sum_writer.add_summary(summary, global_step)

      # accumulate loss
      train_loss.append(loss_batch)
      batch_sizes.append(Xb.shape[0])

      # next batch
      Xb, Yb, NAb, Nb = batcher.next(fwdrc, shift)
      batch_num += 1

    # reset training batcher if epoch considered all of the data
    if epoch_batches is None:
      batcher.reset()

    avg_loss = np.average(train_loss, weights=batch_sizes)

    return avg_loss, global_step

  def train_epoch_h5(self,
                     sess,
                     batcher,
                     sum_writer=None,
                     epoch_batches=None,
                     no_steps=False):
    """Execute one training epoch using HDF5 data,
       and compute-graph augmentation"""

    # initialize training loss
    train_loss = []
    batch_sizes = []
    global_step = 0

    # setup feed dict
    fd = self.set_mode('train')

    # get first batch
    Xb, Yb, NAb, Nb = batcher.next()

    batch_num = 0
    while Xb is not None and (epoch_batches is None or batch_num < epoch_batches):
      # update feed dict
      fd[self.inputs_ph] = Xb
      fd[self.targets_ph] = Yb

      if no_steps:
        run_returns = sess.run([self.merged_summary, self.loss_train] + \
                                self.update_ops, feed_dict=fd)
        summary, loss_batch = run_returns[:2]
      else:
        run_ops = [self.merged_summary, self.loss_train, self.global_step, self.step_op]
        run_ops += self.update_ops
        summary, loss_batch, global_step = sess.run(run_ops, feed_dict=fd)[:3]

      # add summary
      if sum_writer is not None:
        sum_writer.add_summary(summary, global_step)

      # accumulate loss
      train_loss.append(loss_batch)
      batch_sizes.append(Nb)

      # next batch
      Xb, Yb, NAb, Nb = batcher.next()
      batch_num += 1

    # reset training batcher if epoch considered all of the data
    if epoch_batches is None:
      batcher.reset()

    avg_loss = np.average(train_loss, weights=batch_sizes)

    return avg_loss, global_step


  def train_epoch_tfr(self, sess, sum_writer=None, epoch_batches=None):
    """ Execute one training epoch, using TFRecords data. """

    # initialize training loss
    train_loss = []
    batch_sizes = []
    global_step = 0

    # setup feed dict
    fd = self.set_mode('train')

    data_available = True
    batch_num = 0
    while data_available and (epoch_batches is None or batch_num < epoch_batches):
      try:
        # update_ops won't run
        run_ops = [self.merged_summary, self.loss_train, self.preds_train, self.global_step, self.step_op] + self.update_ops
        run_returns = sess.run(run_ops, feed_dict=fd)
        summary, loss_batch, preds, global_step = run_returns[:4]

        # add summary
        if sum_writer is not None:
          sum_writer.add_summary(summary, global_step)

        # accumulate loss
        train_loss.append(loss_batch)
        batch_sizes.append(preds.shape[0])

        # next batch
        batch_num += 1

      except tf.errors.OutOfRangeError:
        data_available = False

    avg_loss = np.average(train_loss, weights=batch_sizes)

    return avg_loss, global_step
