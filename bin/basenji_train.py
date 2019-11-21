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

import os
import pdb
from queue import Queue
import sys
from threading import Thread
import time

import numpy as np
import tensorflow as tf

from basenji import params
from basenji import seqnn
from basenji import shared_flags
from basenji import dataset

FLAGS = tf.app.flags.FLAGS


def main(_):
  np.random.seed(FLAGS.seed)

  # Target train and valid tfr files
  train_files = 'path_to_tfrecords_directory/train-*.tfr'
  test_files = 'path_to_tfrecords_directory/valid-*.tfr'

  # determine whether to save predictions/targets
  if not FLAGS.r2 and not FLAGS.r:
    FLAGS.metrics_sample = 0.

  run(params_file=FLAGS.params,
      train_files=train_files,
      test_files=test_files,
      train_epochs=FLAGS.train_epochs,
      train_epoch_batches=FLAGS.train_epoch_batches,
      test_epoch_batches=FLAGS.test_epoch_batches)


def run(params_file, train_files, test_files, train_epochs, train_epoch_batches,
        test_epoch_batches):

  # parse shifts
  augment_shifts = [int(shift) for shift in FLAGS.augment_shifts.split(',')]
  ensemble_shifts = [int(shift) for shift in FLAGS.ensemble_shifts.split(',')]

  # read parameters
  job = params.read_job_params(params_file)
  job['num_genomes'] = job.get('num_genomes', 1)
  if not isinstance(job['num_targets'], list):
    job['num_targets'] = [job['num_targets']]

  # load data
  data_ops, handle, train_dataseqs, test_dataseqs = make_data_ops(
      job, train_files, test_files)

  # initialize model
  model = seqnn.SeqNN()
  model.build_from_data_ops(job, data_ops,
                            FLAGS.augment_rc, augment_shifts,
                            FLAGS.ensemble_rc, ensemble_shifts)

  # launch accuracy metrics compute thread
  if FLAGS.metrics_thread:
    metrics_queue = Queue()
    metrics_thread = MetricsWorker(metrics_queue)
    metrics_thread.start()


  # checkpoints
  saver = tf.train.Saver()

  # specify CPU parallelism
  session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=2,
        inter_op_parallelism_threads=5)

  # with tf.Session(config=session_conf) as sess:
  with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(FLAGS.logdir + '/train',
                                         sess.graph) if FLAGS.logdir else None

    # generate handles
    for gi in range(job['num_genomes']):
      train_dataseqs[gi].make_handle(sess)
      test_dataseqs[gi].make_handle(sess)

    if FLAGS.restart:
      # load variables into session
      saver.restore(sess, FLAGS.restart)
    else:
      # initialize variables
      print('Initializing...')
      sess.run(tf.local_variables_initializer())
      sess.run(tf.global_variables_initializer())

    train_loss = None
    best_loss = None
    early_stop_i = 0

    epoch = 0

    while (train_epochs is not None and epoch < train_epochs) or \
          (train_epochs is None and early_stop_i < FLAGS.early_stop):
      t0 = time.time()

      # initialize training data epochs
      for gi in range(job['num_genomes']):
        if train_dataseqs[gi].iterator is not None:
          sess.run(train_dataseqs[gi].iterator.initializer)

      # train epoch
      train_losses, steps = model.train2_epoch_ops(sess, handle, train_dataseqs)

      if FLAGS.metrics_thread:
        # block for previous metrics compute
        metrics_queue.join()

      # test validation
      valid_accs = []
      valid_losses = []
      for gi in range(job['num_genomes']):
        if test_dataseqs[gi].iterator is None:
          valid_accs.append(None)
          valid_losses.append(np.nan)

        else:
          # initialize
          sess.run(test_dataseqs[gi].iterator.initializer)

          # compute
          valid_acc = model.test_tfr(sess, test_dataseqs[gi], handle, test_epoch_batches, FLAGS.metrics_sample)

          # save
          valid_accs.append(valid_acc)
          valid_losses.append(valid_acc.loss)

      # summarize
      train_loss = np.nanmean(train_losses)
      valid_loss = np.nanmean(valid_losses)

      # consider as best
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

      # compute and write accuracy metrics update
      update_args = (epoch, steps, train_losses, valid_losses, valid_accs, time_str, best_str)
      if FLAGS.metrics_thread:
        metrics_queue.put(update_args)
      else:
        metrics_update(*update_args)

      # checkpoint
      saver.save(sess, '%s/model_check.tf' % FLAGS.logdir)

      # update epoch
      epoch += 1

    # block for final metrics compute
    if FLAGS.metrics_thread:
      metrics_queue.join()

    if FLAGS.logdir:
      train_writer.close()


def make_data_ops(job, train_patterns, test_patterns):
  """Make input data operations."""

  def make_dataset(tfr_pattern, mode):
    return dataset.DatasetSeq(
        tfr_pattern,
        job['batch_size'],
        job['seq_length'],
        job['target_length'],
        mode=mode)

  train_dataseqs = []
  test_dataseqs = []

  # make datasets and iterators for each genome's train/test
  if job['num_genomes'] > 1 :
    for gi in range(job['num_genomes']):
      train_dataseq = make_dataset(train_patterns[gi], mode=tf.estimator.ModeKeys.TRAIN)
      train_dataseq.make_iterator_initializable()
      train_dataseqs.append(train_dataseq)
      
      test_dataseq = make_dataset(test_patterns[gi], mode=tf.estimator.ModeKeys.EVAL)
      test_dataseq.make_iterator_initializable()
      test_dataseqs.append(test_dataseq)
      
      # verify dataset shapes
      if train_dataseq.num_targets_nonzero != job['num_targets'][gi]:
        print('WARNING: %s nonzero targets found, but %d specified for genome %d.' % (train_dataseq.num_targets_nonzero, job['num_targets'][gi], gi), file=sys.stderr)
        
      if train_dataseq.seq_depth is not None:
        if 'seq_depth' in job:
          assert(job['seq_depth'] == train_dataseq.seq_depth)
        else:
          job['seq_depth'] = train_dataseq.seq_depth
  else :
    train_dataseq = make_dataset(train_patterns, mode=tf.estimator.ModeKeys.TRAIN)
    train_dataseq.make_iterator_initializable()
    train_dataseqs.append(train_dataseq)

    test_dataseq = make_dataset(test_patterns, mode=tf.estimator.ModeKeys.EVAL)
    test_dataseq.make_iterator_initializable()
    test_dataseqs.append(test_dataseq)

    # verify dataset shapes

    if train_dataseq.num_targets_nonzero != job['num_targets']:
      print('WARNING: %s nonzero targets found, but %d specified.' % (train_dataseq.num_targets_nonzero, job['num_targets']), file=sys.stderr)

      if train_dataseq.seq_depth is not None:
        if 'seq_depth' in job:
          assert(job['seq_depth'] == train_dataseq.seq_depth)

      else:
        job['seq_depth'] = train_dataseq.seq_depth
         
  # create feedable iterator
  handle = tf.placeholder(tf.string, shape=[])

  for gi in range(job['num_genomes']):
    # find a non-empty dataset
    if train_dataseqs[gi].iterator is not None:
      iterator = tf.data.Iterator.from_string_handle(handle,
                                                     train_dataseqs[gi].dataset.output_types,
                                                     train_dataseqs[gi].dataset.output_shapes)
      break

  data_ops = iterator.get_next()

  return data_ops, handle, train_dataseqs, test_dataseqs


def metrics_update(epoch, steps, train_losses, valid_losses, valid_accs, time_str, best_str):
  """Compute and print accuracy metrics update."""
  num_genomes = len(train_losses)

  # summarize losses
  train_loss = np.nanmean(train_losses)
  valid_loss = np.nanmean(valid_losses)


  # take means across targets within genoem
  valid_rs = np.zeros(num_genomes)
  valid_r2s = np.zeros(num_genomes)
  for gi, valid_acc in enumerate(valid_accs):
    if np.isnan(valid_losses[gi]):
      valid_rs[gi] = np.nan
      valid_r2s[gi] = np.nan
    else:
      if FLAGS.r:
        valid_rs[gi] = valid_acc.pearsonr().mean()
      if FLAGS.r2:
        valid_r2s[gi] = valid_acc.r2().mean()
  valid_r2_mean = np.nanmean(valid_r2s)
  valid_r_mean = np.nanmean(valid_rs)

  # print cross-genome update
  print('Epoch: %3d,  Steps: %7d,  Train loss: %7.5f,' % (epoch+1, steps, train_loss), end='')
  print(' Valid loss: %7.5f,' % valid_loss, end='')
  if FLAGS.r2:
    print(' Valid R2: %7.5f,' % valid_r2_mean, end='')
  if FLAGS.r:
    print(' Valid R: %7.5f,' % valid_r_mean, end='')
  print(' Time: %s%s' % (time_str, best_str))

  # print genome-specific updates
  if num_genomes > 1:
    for gi in range(num_genomes):
      if not np.isnan(valid_losses[gi]):
        print(' Genome:%d,                    Train loss: %7.5f,' % (gi, train_losses[gi]), end='')
        print(' Valid loss: %7.5f,' % valid_losses[gi], end='')
        if FLAGS.r2:
          print(' Valid R2: %7.5f,' % valid_r2s[gi], end='')
        if FLAGS.r:
          print(' Valid R: %7.5f,' % valid_rs[gi], end='')
        print('')
  sys.stdout.flush()

  # delete predictions and targets
  for valid_acc in valid_accs:
    if valid_acc is not None:
      del valid_acc

class MetricsWorker(Thread):
  """Compute accuracy metrics and print update line."""
  def __init__(self, metrics_queue):
    Thread.__init__(self)
    self.queue = metrics_queue
    self.daemon = True

  def run(self):
    while True:
      try:
        metrics_update(*self.queue.get())

      except:
        # communicate error
        print('ERROR: epoch accuracy and progress update failed.', flush=True)

      # communicate finished task
      self.queue.task_done()

if __name__ == '__main__':
  tf.app.run(main)
