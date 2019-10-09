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

import tensorflow as tf

from basenji import layers
from basenji import metrics

class Trainer:
  def __init__(self, params, train_data, eval_data, out_dir):
    self.params = params
    self.train_data = train_data
    self.eval_data = eval_data
    self.out_dir = out_dir
    self.compiled = False

    # optimizer
    self.make_optimizer()

    # early stopping
    self.patience = self.params.get('patience', 20)

    # compute batches/epoch
    self.train_epoch_batches = train_data.batches_per_epoch()
    self.eval_epoch_batches = eval_data.batches_per_epoch()
    self.train_epochs = self.params.get('train_epochs', 1000)

  def compile(self, model):
    num_targets = model.output_shape[-1]
    model.compile(loss=self.params.get('loss','poisson'),
                  optimizer=self.optimizer,
                  metrics=[metrics.PearsonR(num_targets), metrics.R2(num_targets)])
    self.compiled = True

  def fit(self, model):
    if not self.compiled:
      self.compile(model)

    callbacks = [
      tf.keras.callbacks.EarlyStopping(patience=self.patience, monitor='val_loss', verbose=1),
      tf.keras.callbacks.TensorBoard(self.out_dir),
      tf.keras.callbacks.ModelCheckpoint('%s/model_check.h5'%self.out_dir),
      tf.keras.callbacks.ModelCheckpoint('%s/model_best.h5'%self.out_dir, save_best_only=True, monitor='val_loss', verbose=1)]

    model.fit(
      self.train_data.dataset,
      epochs=self.train_epochs,
      steps_per_epoch=self.train_epoch_batches,
      callbacks=callbacks,
      validation_data=self.eval_data.dataset,
      validation_steps=self.eval_epoch_batches)


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

    # optimizer
    optimizer_type = self.params.get('optimizer', 'sgd').lower()
    if optimizer_type == 'adam':
      self.optimizer = tf.keras.optimizers.Adam(
          lr=lr_schedule,
          beta_1=self.params.get('adam_beta1',0.9),
          beta_2=self.params.get('adam_beta2',0.999))

    elif optimizer_type in ['sgd', 'momentum']:
      self.optimizer = tf.keras.optimizers.SGD(
          lr=lr_schedule,
          momentum=self.params.get('momentum', 0.99))

    else:
      print('Cannot recognize optimization algorithm %s' % optimizer_type)
      exit(1)


class EarlyStoppingBest(tf.keras.callbacks.EarlyStopping):
  """Adds printing "best" in verbose mode."""
  def __init__(self, **kwargs):
    super(EarlyStoppingBest, self).__init__(**kwargs)

  def on_epoch_end(self, epoch, logs=None):
    current = self.get_monitor_value(logs)
    if current is None:
      return
    if self.monitor_op(current - self.min_delta, self.best):
      if self.verbose > 0:
        print(' - best!', end='')
      self.best = current
      self.wait = 0
      if self.restore_best_weights:
        self.best_weights = self.model.get_weights()
    else:
      self.wait += 1
      if self.wait >= self.patience:
        self.stopped_epoch = epoch
        self.model.stop_training = True
        if self.restore_best_weights:
          if self.verbose > 0:
            print('Restoring model weights from the end of the best epoch.')
          self.model.set_weights(self.best_weights)
