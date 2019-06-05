"""SeqNN trainer"""

import tensorflow as tf

from basenji import layers

class SeqNNTrainer:
  def __init__(self, params, train_data, eval_data):
    self.params = params
    self.train_data = train_data
    self.eval_data = eval_data

    # optimizer
    self.make_optimizer()

    # early stopping
    self.patience = self.params.get('patience', 20)

    # compute batches/epoch
    self.train_epoch_batches = train_data.batches_per_epoch()
    self.eval_epoch_batches = eval_data.batches_per_epoch()
    self.train_epochs = self.params.get('train_epochs', 1000)


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
