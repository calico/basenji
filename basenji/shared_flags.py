"""Common flags for various basenji training functions."""


import tensorflow as tf

tf.flags.DEFINE_string('logdir', '/tmp/zrl',
                       'directory to keep checkpoints and summaries in')
tf.flags.DEFINE_boolean('learn_rate_drop', False,
                        'Drop learning rate when training loss stalls')
tf.flags.DEFINE_integer('mc_n', 0, 'Monte Carlo test iterations')
tf.flags.DEFINE_integer('min_epochs', 0, 'Minimum epochs to train')
tf.flags.DEFINE_string('restart', None, 'Restart training the model')
tf.flags.DEFINE_boolean(
    'rc', False,
    'Average the forward and reverse complement predictions when testing')
tf.flags.DEFINE_string('save_prefix', 'houndnn', 'Prefix for save files')
tf.flags.DEFINE_integer('seed', 1, 'Random seed')
tf.flags.DEFINE_string('shifts', '0', 'Ensemble prediction shifts.')
tf.flags.DEFINE_string('params', '', 'File containing parameter config')
tf.flags.DEFINE_string('data', '', 'hd5 data file')
tf.flags.DEFINE_string('train_data', '', 'train tfrecord file')
tf.flags.DEFINE_string('test_data', '', 'test tfrecord file')
tf.flags.DEFINE_boolean('log_device_placement', False,
                        'Log device placement (ie, CPU or GPU)')
tf.flags.DEFINE_integer('train_steps_per_iteration', None,
                        'if > 0, use this many steps for an epoch')
tf.flags.DEFINE_integer('num_test_batches', None,
                        'if > 0, use this many test examples when evaluating')
tf.flags.DEFINE_integer('num_train_epochs', None,
                        'number of full data passes for which to run training.')

tf.flags.DEFINE_boolean('augment_with_rc', False,
                        'Do data augmentation with reverse complement.')
