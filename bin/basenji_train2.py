#!/usr/bin/env python
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


import sys
import time

import numpy as np
import tensorflow as tf

from basenji import params
from basenji import seqnn
from basenji import shared_flags
from basenji import tfrecord_batcher

FLAGS = tf.app.flags.FLAGS


def main(_):
  np.random.seed(FLAGS.seed)

  run(params_file=FLAGS.params,
      train_files=[FLAGS.train0_data, FLAGS.train1_data],
      test_files=[FLAGS.test1_data, FLAGS.test2_data],
      train_epochs=FLAGS.train_epochs,
      train_epoch_batches=FLAGS.train_epoch_batches,
      test_epoch_batches=FLAGS.test_epoch_batches)


def run(params_file, train_file, test_file, train_epochs, train_epoch_batches,
        test_epoch_batches):
  # read parameters
  job = params.read_job_params(params_file)

  # load data
  data_ops, train_iterators, test_iterators, handle = make_data_ops(
      job, train_files, test_files)

  # initialize model
  model = seqnn.SeqNN()
  model.build_from_data_ops(job, data_ops)

  # checkpoints
  saver = tf.train.Saver()

  with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(FLAGS.logdir + '/train',
                                         sess.graph) if FLAGS.logdir else None

    # start queue runners
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(coord=coord)

    # generate handles
    train_handles = []
    test_handles = []
    for gi in range(job['num_genomes']):
      train_handles.append(sess.run(train_iterators[gi].string_handle()))
      test_handles.append(sess.run(test_iterators[gi].string_handle()))

    if FLAGS.restart:
      # load variables into session
      saver.restore(sess, FLAGS.restart)
    else:
      # initialize variables
      t0 = time.time()
      print('Initializing...')
      sess.run(tf.local_variables_initializer())
      sess.run(tf.global_variables_initializer())
      print('Initialization time %f' % (time.time() - t0))

    train_loss = None
    best_loss = None
    early_stop_i = 0

    for epoch in range(train_epochs):
      if early_stop_i < FLAGS.early_stop or epoch < FLAGS.min_epochs:
        t0 = time.time()

        # save previous
        train_loss_last = train_loss

        # initialize training data epochs
        for gi in range(job['num_genomes']):
          sess.run(train_iterators[gi].initializer)

        # train epoch
        train_losses, steps = model.train2_epoch_ops(sess, handle, train_handles)

        # summarize
        train_loss = np.mean(train_losses)

        # test validation
        valid_losses = []
        valid_r2s = []
        for gi in range(job['num_genomes']):
          # initialize
          sess.run(test_iterators[gi].initializer)

          # compute
          valid_acc = model.test_from_data_ops(sess, handle, test_handles[gi], test_epoch_batches)

          # save
          valid_losses.append(valid_acc.loss)
          valid_r2s.append(valid_acc.r2().mean())
          del valid_acc

        # summarize
        valid_loss = np.mean(valid_losses)
        valid_r2 = np.mean(valid_r2s)

        best_str = ''
        if best_loss is None or valid_loss < best_loss:
          best_loss = valid_loss
          best_str = ', best!'
          early_stop_i = 0
          saver.save(sess, '%s/model_best.tf' % FLAGS.logdir)
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
        print('Epoch: %3d,  Steps: %7d,  Train loss: %7.5f,' % (epoch+1, steps, train_loss), end='')
        print(' Valid loss: %7.5f,  Valid R2: %7.5f,' % (valid_loss, valid_r2), end='')
        print(' Time: %s%s' % (time_str, best_str))

        # print genome-specific updates
        for gi in range(job['num_genomes']):
          print('%30s Train loss: %7.5f, Valid loss: %7.5f, Valid R2: %7.5f' % (train_losses[gi], valid_losses[gi], valid_r2s[gi]))
        sys.stdout.flush()

    if FLAGS.logdir:
      train_writer.close()


def make_data_ops(job, train_files, test_files):
  """Make input data operations."""

  def make_dataset(filename, num_targets, mode):
    return tfrecord_batcher.tfrecord_dataset(
        filename,
        job['batch_size'],
        job['seq_length'],
        job.get('seq_depth', 4),
        job['target_length'],
        job['num_targets'],
        mode=mode,
        repeat=False)

  train_datasets = []
  test_datasets = []
  train_iterators = []
  test_iterators = []

  # make datasets and iterators for each genome's train/test
  for gi in range(job['num_genomes']):
    train_dataset = make_dataset(train_files[gi], job['num_targets'], mode=tf.estimator.ModeKeys.TRAIN)
    train_iterator = train_dataset.make_initializable_iterator()
    train_datasets.append(train_dataset)
    train_iterators.append(train_iterator)

    test_dataset = make_dataset(test_files[gi], job['num_targets'], mode=tf.estimator.ModeKeys.EVAL)
    test_iterator = test_dataset.make_initializable_iterator()
    test_datasets.append(test_dataset)
    test_iterators.append(test_iterator)

  # create feedable iterator
  handle = tf.placeholder(tf.string, shape=[])
  iterator = tf.data.Iterator.from_string_handle(handle,
                                                 train_datasets[0].output_types,
                                                 train_datasets[0].output_shapes)
  data_ops = iterator.get_next()

  return data_ops, train_iterators, test_iterators, handle


if __name__ == '__main__':
  tf.app.run(main)
