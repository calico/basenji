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
tf.flags.DEFINE_integer('train_steps_per_iteration', None,
                        'if > 0, use this many steps for an epoch')
tf.flags.DEFINE_integer('num_test_batches', None,
                        'if > 0, use this many test examples when evaluating')
tf.flags.DEFINE_integer('num_train_epochs', 1000,
                        'number of full data passes for which to run training.')
tf.flags.DEFINE_integer('min_epochs', 0, 'Minimum epochs to train')


# training modes
tf.flags.DEFINE_boolean('no_steps', False, 'Update ops but no step ops')
tf.flags.DEFINE_string('restart', None, 'Restart training the model')
tf.flags.DEFINE_integer('early_stop', 25, 'Stop training if validation loss stagnates.')
