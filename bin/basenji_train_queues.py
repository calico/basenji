#!/usr/bin/env python
from __future__ import print_function

import pdb
import sys
import time

import h5py
import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow as tf

from basenji import batcher
from basenji import dna_io
from basenji import seqnn
from basenji import tfrecord_batcher
from bin import shared_flags

FLAGS = tf.app.flags.FLAGS


def make_data_ops(job, train_file, test_file):

  def make_dataset(filename, mode):
    return tfrecord_batcher.tfrecord_dataset(
        filename,
        job['batch_size'],
        job['seq_length'],
        job.get('seq_depth',4),
        job['num_targets'],
        job['target_length'],
        mode=mode)

  training_dataset = make_dataset(train_file, mode=tf.estimator.ModeKeys.TRAIN)
  test_dataset = make_dataset(test_file, mode=tf.estimator.ModeKeys.EVAL)

  iterator = tf.contrib.data.Iterator.from_structure(
      training_dataset.output_types, training_dataset.output_shapes)
  data_ops = iterator.get_next()

  training_init_op = iterator.make_initializer(training_dataset)
  test_init_op = iterator.make_initializer(test_dataset)
  return data_ops, training_init_op, test_init_op


def main(_):
  np.random.seed(FLAGS.seed)

  run(params_file=FLAGS.params,
      train_file=FLAGS.train_data,
      test_file=FLAGS.test_data,
      num_train_epochs=FLAGS.num_train_epochs,
      batches_per_epoch=FLAGS.train_steps_per_iteration,
      num_test_batches=FLAGS.num_test_batches)


def run(params_file, train_file, test_file, num_train_epochs, batches_per_epoch,
        num_test_batches):
  np.random.seed(FLAGS.seed)

  shifts = [int(shift) for shift in FLAGS.shifts.split(',')]
  job = dna_io.read_job_params(params_file)

  job['early_stop'] = job.get('early_stop', 16)
  job['rate_drop'] = job.get('rate_drop', 3)

  data_ops, training_init_op, test_init_op = make_data_ops(
      job, train_file, test_file)

  dr = seqnn.SeqNN()
  dr.build_from_data_ops(job, data_ops)

  # checkpoints
  saver = tf.train.Saver()

  with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(FLAGS.logdir + '/train',
                                         sess.graph) if FLAGS.logdir else None

    t0 = time.time()
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(coord=coord)

    if FLAGS.restart:
      # load variables into session
      saver.restore(sess, FLAGS.restart)
    else:
      # initialize variables
      print('Initializing...')
      sess.run(tf.global_variables_initializer())
      print('Initialization time %f' % (time.time() - t0))

    train_loss = None
    best_loss = None
    early_stop_i = 0
    undroppable_counter = 3
    max_drops = 8
    num_drops = 0

    for epoch in range(num_train_epochs):
      if early_stop_i < job['early_stop'] or epoch < FLAGS.min_epochs:
        t0 = time.time()

        # save previous
        train_loss_last = train_loss

        # alternate forward and reverse batches
        fwdrc = True
        if FLAGS.rc and epoch % 2 == 1:
          fwdrc = False

        # cycle shifts
        shift_i = epoch % len(shifts)

        sess.run(training_init_op)
        # train
        train_loss = dr.train_epoch_from_data_ops(sess, train_writer,
                                                  batches_per_epoch)

        sess.run(test_init_op)

        valid_acc = dr.test_from_data_ops(
            sess,
            num_test_batches=num_test_batches)
        valid_loss = valid_acc.loss
        valid_r2 = valid_acc.r2().mean()
        del valid_acc

        best_str = ''
        if best_loss is None or valid_loss < best_loss:
          best_loss = valid_loss
          best_str = ', best!'
          early_stop_i = 0
          saver.save(sess, '%s/%s_best.tf' % (FLAGS.logdir, FLAGS.save_prefix))
        else:
          early_stop_i += 1

        # measure time
        et = time.time() - t0
        if et < 600:
          time_str = '%3ds' % et
        elif et < 6000:
          time_str = '%3dm' % (et / 60)
        else:
          time_str = '%3.1fh' % (et / 3600)

        # print update
        print(
            'Epoch %3d: Train loss: %7.5f, Valid loss: %7.5f,' %
            (epoch + 1, train_loss, valid_loss),
            end='')
        print(
            ' Valid R2: %7.5f, Time: %s%s' % (valid_r2, time_str, best_str),
            end='')

        # if training stagnant
        if FLAGS.learn_rate_drop and (
            num_drops < max_drops) and undroppable_counter == 0 and (
                train_loss_last - train_loss) / train_loss_last < 0.0002:
          print(', rate drop', end='')
          dr.drop_rate(2 / 3)
          undroppable_counter = 1
          num_drops += 1
        else:
          undroppable_counter = max(0, undroppable_counter - 1)

        print('')
        sys.stdout.flush()

    if FLAGS.logdir:
      train_writer.close()


if __name__ == '__main__':
  tf.app.run(main)
