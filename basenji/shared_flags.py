"""Common flags for various basenji training functions."""

import tensorflow as tf

# parameters and data
tf.flags.DEFINE_string('params', '', 'File containing parameter config')
tf.flags.DEFINE_string('data', '', 'hd5 data file')
tf.flags.DEFINE_string('train_data', '', 'train tfrecord file')
tf.flags.DEFINE_string('test_data', '', 'test tfrecord file')


# ensembling/augmentation
tf.flags.DEFINE_boolean('augment_rc', False, 'Augment training with reverse complement.')
tf.flags.DEFINE_boolean('ensemble_rc', False, 'Ensemble prediction with reverse complement.')
tf.flags.DEFINE_string('augment_shifts', '0', 'Augment training with shifted sequences.')
tf.flags.DEFINE_string('ensemble_shifts', '0', 'Ensemble prediction with shifted sequences.')
tf.flags.DEFINE_integer('ensemble_mc', 0, 'Ensemble monte carlo samples.')

# logging
tf.flags.DEFINE_boolean('check_all', False, 'Checkpoint every epoch')
tf.flags.DEFINE_string('logdir', '/tmp/zrl',
                       'directory to keep checkpoints and summaries in')
tf.flags.DEFINE_boolean('log_device_placement', False,
                        'Log device placement (ie, CPU or GPU)')
tf.flags.DEFINE_integer('seed', 1, 'Random seed')


# step counts
tf.flags.DEFINE_integer('train_epochs', None,
                        'Number of training epochs.')
tf.flags.DEFINE_integer('train_epoch_batches', None,
                        'Number of batches per training epoch.')
tf.flags.DEFINE_integer('test_epoch_batches', None,
                        'Number of batches per test epoch.')

# training modes
tf.flags.DEFINE_boolean('no_steps', False, 'Update ops but no step ops')
tf.flags.DEFINE_string('restart', None, 'Restart training the model')
tf.flags.DEFINE_integer('early_stop', 25, 'Stop training if validation loss stagnates.')

# eval options
tf.flags.DEFINE_boolean('acc_thread', False, 'Evaluate validation accuracy in a separate thread.')