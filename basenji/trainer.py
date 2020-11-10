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
"""SeqNN trainer"""
import time
from packaging import version
import pdb

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

from basenji import layers
from basenji import metrics

def parse_loss(loss_label, strategy=None, keras_fit=True, spec_weight=1):
  """Parse loss function from label, strategy, and fitting method."""
  if strategy is not None and not keras_fit:
    if loss_label == 'mse':
      loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    elif loss_label == 'bce':
      loss_fn = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    else:
      loss_fn = tf.keras.losses.Poisson(reduction=tf.keras.losses.Reduction.NONE)
  else:
    if loss_label == 'mse':
      loss_fn = tf.keras.losses.MeanSquaredError()
    elif loss_label == 'mse_udot':
      loss_fn = metrics.MeanSquaredErrorUDot(spec_weight)
    elif loss_label == 'bce':
      loss_fn = tf.keras.losses.BinaryCrossentropy()
    else:
      loss_fn = tf.keras.losses.Poisson()

  return loss_fn

class Trainer:
  def __init__(self, params, train_data, eval_data, out_dir,
               strategy=None, num_gpu=1, keras_fit=True):
    self.params = params
    self.train_data = train_data
    if type(self.train_data) is not list:
      self.train_data = [self.train_data]
    self.eval_data = eval_data
    if type(self.eval_data) is not list:
      self.eval_data = [self.eval_data]
    self.out_dir = out_dir
    self.strategy = strategy
    self.num_gpu = num_gpu
    self.batch_size = self.train_data[0].batch_size
    self.compiled = False

    # loss
    self.spec_weight = self.params.get('spec_weight', 1)
    self.loss = self.params.get('loss','poisson').lower()
    self.loss_fn = parse_loss(self.loss, self.strategy, keras_fit, self.spec_weight)

    # optimizer
    self.make_optimizer()

    # early stopping
    self.patience = self.params.get('patience', 20)

    # compute batches/epoch
    self.train_epoch_batches = [td.batches_per_epoch() for td in self.train_data]
    self.eval_epoch_batches = [ed.batches_per_epoch() for ed in self.eval_data]
    self.train_epochs_min = self.params.get('train_epochs_min', 1)
    self.train_epochs_max = self.params.get('train_epochs_max', 10000)

    # dataset
    self.num_datasets = len(self.train_data)
    self.dataset_indexes = []
    for di in range(self.num_datasets):
      self.dataset_indexes += [di]*self.train_epoch_batches[di]
    self.dataset_indexes = np.array(self.dataset_indexes)

  def compile(self, seqnn_model):
    for model in seqnn_model.models:
      if self.loss == 'bce':
        model_metrics = [metrics.SeqAUC(curve='ROC'), metrics.SeqAUC(curve='PR')]
      else:
        num_targets = model.output_shape[-1]
        model_metrics = [metrics.PearsonR(num_targets), metrics.R2(num_targets)]
      
      model.compile(loss=self.loss_fn,
                    optimizer=self.optimizer,
                    metrics=model_metrics)
    self.compiled = True

  def fit_keras(self, seqnn_model):
    if not self.compiled:
      self.compile(seqnn_model)

    if self.loss == 'bce':
      early_stop = EarlyStoppingMin(monitor='val_loss', mode='min', verbose=1,
                       patience=self.patience, min_epoch=self.train_epochs_min)
      save_best = tf.keras.callbacks.ModelCheckpoint('%s/model_best.h5'%self.out_dir,
                                                     save_best_only=True, mode='min',
                                                     monitor='val_loss', verbose=1)
    else:
      early_stop = EarlyStoppingMin(monitor='val_pearsonr', mode='max', verbose=1,
                       patience=self.patience, min_epoch=self.train_epochs_min)
      save_best = tf.keras.callbacks.ModelCheckpoint('%s/model_best.h5'%self.out_dir,
                                                     save_best_only=True, mode='max',
                                                     monitor='val_pearsonr', verbose=1)

    callbacks = [
      early_stop,
      tf.keras.callbacks.TensorBoard(self.out_dir),
      tf.keras.callbacks.ModelCheckpoint('%s/model_check.h5'%self.out_dir),
      save_best]

    seqnn_model.model.fit(
      self.train_data[0].dataset,
      epochs=self.train_epochs_max,
      steps_per_epoch=self.train_epoch_batches[0],
      callbacks=callbacks,
      validation_data=self.eval_data[0].dataset,
      validation_steps=self.eval_epoch_batches[0])


  def fit2(self, seqnn_model):
    if not self.compiled:
      self.compile(seqnn_model)

    assert(len(seqnn_model.models) >= self.num_datasets)

    ################################################################
    # prep

    # metrics
    train_loss, train_r, train_r2 = [], [], []
    valid_loss, valid_r, valid_r2 = [], [], []
    for di in range(self.num_datasets):
      num_targets = seqnn_model.models[di].output_shape[-1]
      train_loss.append(tf.keras.metrics.Mean(name='train%d_loss'%di))
      train_r.append(metrics.PearsonR(num_targets, name='train%d_r'%di))
      train_r2.append(metrics.R2(num_targets, name='train%d_r2'%di))
      valid_loss.append(tf.keras.metrics.Mean(name='valid%d_loss'%di))
      valid_r.append(metrics.PearsonR(num_targets, name='valid%d_r'%di))
      valid_r2.append(metrics.R2(num_targets, name='valid%d_r2'%di))

    # generate decorated train steps
    """
    train_steps = []
    for di in range(self.num_datasets):
      model = seqnn_model.models[di]

      @tf.function
      def train_step(x, y):
        with tf.GradientTape() as tape:
          pred = model(x, training=tf.constant(True))
          loss = self.loss_fn(y, pred) + sum(model.losses)
        train_loss[di](loss)
        train_r[di](y, pred)
        train_r2[di](y, pred)
        gradients = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

      train_steps.append(train_step)
    """
    @tf.function
    def train_step0(x, y):
      with tf.GradientTape() as tape:
        pred = seqnn_model.models[0](x, training=True)
        loss = self.loss_fn(y, pred) + sum(seqnn_model.models[0].losses)
      train_loss[0](loss)
      train_r[0](y, pred)
      train_r2[0](y, pred)
      gradients = tape.gradient(loss, seqnn_model.models[0].trainable_variables)
      self.optimizer.apply_gradients(zip(gradients, seqnn_model.models[0].trainable_variables))

    @tf.function
    def eval_step0(x, y):
      pred = seqnn_model.models[0](x, training=False)
      loss = self.loss_fn(y, pred) + sum(seqnn_model.models[0].losses)
      valid_loss[0](loss)
      valid_r[0](y, pred)
      valid_r2[0](y, pred)

    if self.num_datasets > 1:
      @tf.function
      def train_step1(x, y):
        with tf.GradientTape() as tape:
          pred = seqnn_model.models[1](x, training=True)
          loss = self.loss_fn(y, pred) + sum(seqnn_model.models[1].losses)
        train_loss[1](loss)
        train_r[1](y, pred)
        train_r2[1](y, pred)
        gradients = tape.gradient(loss, seqnn_model.models[1].trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, seqnn_model.models[1].trainable_variables))

      @tf.function
      def eval_step1(x, y):
        pred = seqnn_model.models[1](x, training=False)
        loss = self.loss_fn(y, pred) + sum(seqnn_model.models[1].losses)
        valid_loss[1](loss)
        valid_r[1](y, pred)
        valid_r2[1](y, pred)


    # improvement variables
    valid_best = [-np.inf]*self.num_datasets
    unimproved = [0]*self.num_datasets

    ################################################################
    # training loop

    for ei in range(self.train_epochs_max):
      if ei >= self.train_epochs_min and np.min(unimproved) > self.patience:
        break
      else:
        # shuffle datasets
        np.random.shuffle(self.dataset_indexes)

        # get iterators
        train_data_iters = [iter(td.dataset) for td in self.train_data]

        # train
        t0 = time.time()
        for di in self.dataset_indexes:
          x, y = next(train_data_iters[di])
          if di == 0:
            train_step0(x, y)
          else:
            train_step1(x, y)

        print('Epoch %d - %ds' % (ei, (time.time()-t0)))
        for di in range(self.num_datasets):
          print('  Data %d' % di, end='')
          model = seqnn_model.models[di]          

          # print training accuracy
          print(' - train_loss: %.4f' % train_loss[di].result().numpy(), end='')
          print(' - train_r: %.4f' %  train_r[di].result().numpy(), end='')
          print(' - train_r: %.4f' %  train_r2[di].result().numpy(), end='')

          # evaluate
          for x, y in self.eval_data[di].dataset:
            if di == 0:
              eval_step0(x, y)
            else:
              eval_step1(x, y)

          # print validation accuracy
          print(' - valid_loss: %.4f' % valid_loss[di].result().numpy(), end='')
          print(' - valid_r: %.4f' % valid_r[di].result().numpy(), end='')
          print(' - valid_r2: %.4f' % valid_r2[di].result().numpy(), end='')
          early_stop_stat = valid_r[di].result().numpy()

          # checkpoint
          model.save('%s/model%d_check.h5' % (self.out_dir, di))

          # check best
          if early_stop_stat > valid_best[di]:
            print(' - best!', end='')
            unimproved[di] = 0
            valid_best[di] = early_stop_stat
            model.save('%s/model%d_best.h5' % (self.out_dir, di))
          else:
            unimproved[di] += 1
          print('', flush=True)

          # reset metrics
          train_loss[di].reset_states()
          train_r[di].reset_states()
          train_r2[di].reset_states()
          valid_loss[di].reset_states()
          valid_r[di].reset_states()
          valid_r2[di].reset_states()

        
  def fit_tape(self, seqnn_model):
    if not self.compiled:
      self.compile(seqnn_model)
    model = seqnn_model.model
    
    # metrics
    num_targets = model.output_shape[-1]
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_r = metrics.PearsonR(num_targets, name='train_r')
    train_r2 = metrics.R2(num_targets, name='train_r2')
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_r = metrics.PearsonR(num_targets, name='valid_r')
    valid_r2 = metrics.R2(num_targets, name='valid_r2')
    
    if self.strategy is None:
      @tf.function
      def train_step(x, y):
        with tf.GradientTape() as tape:
          pred = model(x, training=True)
          loss = self.loss_fn(y, pred) + sum(model.losses)
        train_loss(loss)
        train_r(y, pred)
        train_r2(y, pred)
        gradients = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

      @tf.function
      def eval_step(x, y):
        pred = model(x, training=False)
        loss = self.loss_fn(y, pred) + sum(model.losses)
        valid_loss(loss)
        valid_r(y, pred)
        valid_r2(y, pred)

    else:
      def train_step(x, y):
        with tf.GradientTape() as tape:
          pred = model(x, training=True)
          loss_batch_len = self.loss_fn(y, pred)
          loss_batch = tf.reduce_mean(loss_batch_len, axis=-1)
          loss = tf.reduce_sum(loss_batch) / self.batch_size
          loss += sum(model.losses) / self.num_gpu
        train_r(y, pred)
        train_r2(y, pred)
        gradients = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

      @tf.function
      def train_step_distr(xd, yd):
        replica_losses = self.strategy.run(train_step, args=(xd, yd))
        loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM,
                                    replica_losses, axis=None)
        train_loss(loss)


      def eval_step(x, y):
        pred = model(x, training=False)
        loss = self.loss_fn(y, pred) + sum(model.losses)
        valid_loss(loss)
        valid_r(y, pred)
        valid_r2(y, pred)

      @tf.function
      def eval_step_distr(xd, yd):
        return self.strategy.run(eval_step, args=(xd, yd))


    # improvement variables
    valid_best = -np.inf
    unimproved = 0

    # training loop
    for ei in range(self.train_epochs_max):
      if ei >= self.train_epochs_min and unimproved > self.patience:
        break
      else:
        # train
        t0 = time.time()
        train_iter = iter(self.train_data[0].dataset)
        for si in range(self.train_epoch_batches[0]):
          x, y = next(train_iter)
          if self.strategy is not None:
            train_step_distr(x, y)
          else:
            train_step(x, y)

        # evaluate
        # eval_iter = iter(self.eval_data[0].dataset)
        # for si in range(self.eval_epoch_batches[0]):
        #   x, y = next(eval_iter)
        for x, y in self.eval_data[0].dataset:
          if self.strategy is not None:
            eval_step_distr(x, y)
          else:
            eval_step(x, y)

        # print training accuracy
        train_loss_epoch = train_loss.result().numpy()
        train_r_epoch = train_r.result().numpy()
        train_r2_epoch = train_r2.result().numpy()
        print('Epoch %d - %ds - train_loss: %.4f - train_r: %.4f - train_r2: %.4f' % \
          (ei, (time.time()-t0), train_loss_epoch, train_r_epoch, train_r2_epoch), end='')

        # print validation accuracy
        # valid_loss, valid_pr, valid_r2 = model.evaluate(self.eval_data[0].dataset, verbose=0)
        valid_loss_epoch = valid_loss.result().numpy()
        valid_r_epoch = valid_r.result().numpy()
        valid_r2_epoch = valid_r2.result().numpy()
        print(' - valid_loss: %.4f - valid_r: %.4f - valid_r2: %.4f' % \
          (valid_loss_epoch, valid_r_epoch, valid_r2_epoch), end='')

        # checkpoint
        seqnn_model.save('%s/model_check.h5'%self.out_dir)

        # check best
        if valid_r_epoch > valid_best:
          print(' - best!', end='')
          unimproved = 0
          valid_best = valid_r_epoch
          seqnn_model.save('%s/model_best.h5'%self.out_dir)
        else:
          unimproved += 1
        print('', flush=True)

        # reset metrics
        train_loss.reset_states()
        train_r.reset_states()
        train_r2.reset_states()
        valid_loss.reset_states()
        valid_r.reset_states()
        valid_r2.reset_states()


  def make_optimizer(self):
    # schedule (currently OFF)
    initial_learning_rate = self.params.get('learning_rate', 0.01)
    if False:
      lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=self.params.get('decay_steps', 100000),
        decay_rate=self.params.get('decay_rate', 0.96),
        staircase=True)
    else:
      lr_schedule = initial_learning_rate

    if version.parse(tf.__version__) < version.parse('2.2'):
      clip_norm_default = 1000000
    else:
      clip_norm_default = None
    clip_norm = self.params.get('clip_norm', clip_norm_default)

    # optimizer
    optimizer_type = self.params.get('optimizer', 'sgd').lower()
    if optimizer_type == 'adam':
      self.optimizer = tf.keras.optimizers.Adam(
          lr=lr_schedule,
          beta_1=self.params.get('adam_beta1',0.9),
          beta_2=self.params.get('adam_beta2',0.999),
          clipnorm=clip_norm)

    elif optimizer_type in ['sgd', 'momentum']:
      self.optimizer = tf.keras.optimizers.SGD(
          lr=lr_schedule,
          momentum=self.params.get('momentum', 0.99),
          clipnorm=clip_norm)

    else:
      print('Cannot recognize optimization algorithm %s' % optimizer_type)
      exit(1)

class EarlyStoppingMin(tf.keras.callbacks.EarlyStopping):
  """Stop training when a monitored quantity has stopped improving.
  Arguments:
      min_epoch: Minimum number of epochs before considering stopping.
      
  """
  def __init__(self, min_epoch=0, **kwargs):
    super(EarlyStoppingMin, self).__init__(**kwargs)
    self.min_epoch = min_epoch

  def on_epoch_end(self, epoch, logs=None):
    current = self.get_monitor_value(logs)
    if current is None:
      return
    if self.monitor_op(current - self.min_delta, self.best):
      self.best = current
      self.wait = 0
      if self.restore_best_weights:
        self.best_weights = self.model.get_weights()
    else:
      self.wait += 1
      if epoch >= self.min_epoch and self.wait >= self.patience:
        self.stopped_epoch = epoch
        self.model.stop_training = True
        if self.restore_best_weights:
          if self.verbose > 0:
            print('Restoring model weights from the end of the best epoch.')
          self.model.set_weights(self.best_weights)
