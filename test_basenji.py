"""Tests for the Basenji library."""

from basenji import batcher
from basenji import dna_io
from basenji import ops
from basenji import seqnn
from basenji import tfrecord_batcher

from basenji.bin import basenji_train

import numpy as np
import tensorflow as tf
import os

FLAGS = tf.app.flags.FLAGS
TEST_DATA_DIRECTORY = 'google3/third_party/py/basenji/testdata/'


class BasenjiTest(tf.test.TestCase):

  def testOneHot(self):
    """Checks that the Basenji dna one hot coding works as expected."""
    result = dna_io.dna_1hot('ACGT')
    self.assertTrue(np.array_equal(result, np.eye(4)))

  def testRunWithNoTrainSteps(self):

    input_file_path = os.path.join(FLAGS.test_srcdir, TEST_DATA_DIRECTORY,
                                   'one_record_for_train_and_test.h5')

    params_file_path = os.path.join(FLAGS.test_srcdir, TEST_DATA_DIRECTORY,
                                    'params.small.hd5.txt')

    # Asserts that the job runs for one epoch without raising.
    basenji_train.run(params_file_path, input_file_path, 1)

  def testTFRecordLoading(self):
    data_dir = (FLAGS.test_srcdir + '/google3/third_party/py/basenji/testdata/')
    params_file = data_dir + 'params.small.tfrecord.txt'
    train_data = data_dir + 'cage10-train-small.tfrecord'

    job = dna_io.read_job_params(params_file)
    job['seq_length'] = job['batch_length'] * job['seq_depth']
    batch_size = job['batch_size']

    dataset = tfrecord_batcher.tf_record_dataset(
        train_data,
        batch_size=batch_size,
        seq_depth=job['seq_depth'],
        num_targets=job['num_targets'],
        target_width=job['target_width'],
        shuffle=True,
        trim_eos=True)

    iterator = dataset.make_initializable_iterator()
    data_op = iterator.get_next()

    with tf.Session() as sess:

      coord = tf.train.Coordinator()
      tf.train.start_queue_runners(coord=coord)
      sess.run(iterator.initializer)

      data = sess.run(data_op)
      self.assertAllEqual(data['sequence'].shape, [batch_size, 262144, 4])
      self.assertAllEqual(data['label'].shape, [batch_size, 2048, 10])
      self.assertAllEqual(data['na'].shape, [batch_size, 2048])


  def testTFRecordModelTraining(self):
    data_dir = (FLAGS.test_srcdir + '/google3/third_party/py/basenji/testdata/')
    params_file = data_dir + 'params.small.tfrecord.txt'
    train_data = data_dir + 'cage10-train-small.tfrecord'

    job = dna_io.read_job_params(params_file)
    job['seq_length'] = job['batch_length'] * job['seq_depth']

    batcher_train = tfrecord_batcher.TFRecordBatcher(
        train_data,
        load_targets=True,
        seq_length=job['seq_length'],
        seq_depth=job['seq_depth'],
        target_width=job['target_width'],
        num_targets=job['num_targets'],
        NAf=None,
        batch_size=job['batch_size'],
        pool_width=job['target_pool'],
        shuffle=True)

    basenji_model = seqnn.SeqNN()
    basenji_model.build(job)

    with tf.Session() as sess:

      coord = tf.train.Coordinator()
      tf.train.start_queue_runners(coord=coord)

      batcher_train.session = sess
      batcher_train.initialize(sess)

      sess.run(tf.local_variables_initializer())
      sess.run(tf.global_variables_initializer())

      for _ in range(2):
        train_loss = basenji_model.train_epoch(
            sess, batcher_train, False, 0, None, batches_per_epoch=2)

if __name__ == '__main__':
  tf.test.main()
