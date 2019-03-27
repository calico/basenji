# Copyright 2017 Calico LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
from __future__ import print_function
import os
import pdb
import sys

import numpy as np
import tensorflow as tf

from basenji import tfrecord_util

# Multiplier for how many items to have in the shuffle buffer, invariant
# of how many files we're parallel-interleaving for our input datasets.
SHUFFLE_BUFFER_DEPTH_PER_FILE = 8
# Number of files to concurrently read from, and interleave,
# for our input datasets.
NUM_FILES_TO_PARALLEL_INTERLEAVE = 4

class DatasetSeq:
  def __init__(self, tfr_pattern, batch_size, seq_length, target_length,
               mode, static_batch=False, repeat=False):
    """Initialize basic parameters; run compute_stats; run make_dataset."""

    self.tfr_pattern = tfr_pattern

    self.num_seqs = None
    self.batch_size = batch_size
    self.seq_length = seq_length
    self.seq_depth = None
    self.target_length = target_length
    self.num_targets = None

    self.mode = mode
    self.static_batch = static_batch
    self.repeat = repeat

    self.compute_stats()
    self.make_dataset()


  def make_dataset(self):
    """Make Dataset w/ transformations."""

    # initialize dataset from
    # dataset = tf.data.Dataset.list_files(self.tfr_pattern)
    tfr_files = order_tfrecords(self.tfr_pattern)
    if tfr_files:
      dataset = tf.data.Dataset.list_files(tf.constant(tfr_files), shuffle=False)
    else:
      print('Cannot order TFRecords %s' % self.tfr_pattern, file=sys.stderr)
      dataset = tf.data.Dataset.list_files(self.tfr_pattern)

    # map_func
    def file_to_records(filename):
      return tf.data.TFRecordDataset(filename, compression_type='ZLIB')

    if self.mode == tf.estimator.ModeKeys.TRAIN:
      if self.repeat:
        dataset = dataset.repeat()

      # interleave files
      dataset = dataset.apply(
          tf.contrib.data.parallel_interleave(
              map_func=file_to_records, sloppy=True,
              cycle_length=NUM_FILES_TO_PARALLEL_INTERLEAVE))

      # shuffle
      shuffle_buffer_size = NUM_FILES_TO_PARALLEL_INTERLEAVE * SHUFFLE_BUFFER_DEPTH_PER_FILE
      dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    else:
      # flat mix files
      dataset = dataset.flat_map(file_to_records)

    if self.static_batch:
      dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size))
    else:
      dataset = dataset.batch(self.batch_size)

    def _parse(example_protos):
      features = {
        'genome': tf.FixedLenFeature([1], tf.int64),
        tfrecord_util.TFR_INPUT: tf.FixedLenFeature([], tf.string),
        tfrecord_util.TFR_OUTPUT: tf.FixedLenFeature([], tf.string)
      }
      parsed_features = tf.parse_example(example_protos, features=features)

      static_batch_size = self.batch_size if self.static_batch else -1

      genome = parsed_features['genome']

      seq = tf.decode_raw(parsed_features[tfrecord_util.TFR_INPUT], tf.uint8)
      seq = tf.reshape(seq, [static_batch_size, self.seq_length, self.seq_depth])
      seq = tf.cast(seq, tf.float32)

      label = tf.decode_raw(parsed_features[tfrecord_util.TFR_OUTPUT], tf.float16)
      label = tf.reshape(label, [static_batch_size, self.target_length, self.num_targets])
      label = tf.cast(label, tf.float32)

      if self.static_batch:
        na = tf.zeros(label.shape[:-1], dtype=tf.bool)
      else:
        na = tf.zeros(tf.shape(label)[:-1], dtype=tf.bool)

      return {'genome': genome, 'sequence': seq, 'label': label, 'na': na}

    # helper for training on single genomes in a multiple genome mode
    if self.num_seqs > 0:
      dataset = dataset.map(_parse)

    # hold on
    self.dataset = dataset


  def compute_stats(self):
    """ Iterate over the TFRecords to count sequences, and infer
        seq_depth and num_targets."""

    def parse_proto(example_protos):
      features = {
        'genome': tf.FixedLenFeature([1], tf.int64),
        tfrecord_util.TFR_INPUT: tf.FixedLenFeature([], tf.string),
        tfrecord_util.TFR_OUTPUT: tf.FixedLenFeature([], tf.string)
      }
      parsed_features = tf.parse_example(example_protos, features=features)
      genome = parsed_features['genome']
      seq = tf.decode_raw(parsed_features[tfrecord_util.TFR_INPUT], tf.uint8)
      targets = tf.decode_raw(parsed_features[tfrecord_util.TFR_OUTPUT], tf.float16)
      return {'genome': genome, 'sequence': seq, 'target': targets}

    # read TF Records
    dataset = tf.data.Dataset.list_files(self.tfr_pattern)

    def file_to_records(filename):
      return tf.data.TFRecordDataset(filename, compression_type='ZLIB')
    dataset = dataset.flat_map(file_to_records)

    dataset = dataset.batch(1)
    dataset = dataset.map(parse_proto)

    iterator = dataset.make_one_shot_iterator()
    try:
      next_op = iterator.get_next()
    except tf.errors.OutOfRangeError:
      print('TFRecord pattern %s is empty' % self.tfr_pattern, file=sys.stderr)
      exit(1)

    self.num_seqs = 0

    with tf.Session() as sess:
      try:
        next_datum = sess.run(next_op)
      except tf.errors.OutOfRangeError:
        next_datum = False

      while next_datum:
        # infer seq_depth
        seq_1hot = next_datum['sequence'].reshape((self.seq_length,-1))
        if self.seq_depth is None:
          self.seq_depth = seq_1hot.shape[-1]
        else:
          assert(self.seq_depth == seq_1hot.shape[-1])

        # infer num_targets
        targets1 = next_datum['target'].reshape(self.target_length,-1)
        if self.num_targets is None:
          self.num_targets = targets1.shape[-1]
          targets_nonzero = (targets1.sum(axis=0, dtype='float32') > 0)
        else:
          assert(self.num_targets == targets1.shape[-1])
          targets_nonzero = np.logical_or(targets_nonzero, targets1.sum(axis=0, dtype='float32') > 0)

        # count sequences
        self.num_seqs += 1

        try:
          next_datum = sess.run(next_op)
        except tf.errors.OutOfRangeError:
          next_datum = False

    if self.num_seqs > 0:
      self.num_targets_nonzero = (targets_nonzero > 0).sum()
      print('%s has %d sequences with %d/%d targets' % (self.tfr_pattern, self.num_seqs, self.num_targets_nonzero, self.num_targets), flush=True)
    else:
      self.num_targets_nonzero = None
      print('%s has %d sequences with 0 targets' % (self.tfr_pattern, self.num_seqs), flush=True)

  def epoch_reset(self):
    self.epoch_seqs = self.num_seqs

  def epoch_batch(self, batch_size):
    self.epoch_seqs = max(0, self.epoch_seqs - batch_size)

  def make_initializer(self):
    """Make initializer."""
    return self.iterator.make_initializer(self.dataset)

  def make_iterator_initializable(self):
    """Make initializable iterator."""
    if self.num_seqs > 0:
      self.iterator = self.dataset.make_initializable_iterator()
    else:
      self.iterator = None

  def make_iterator_structure(self):
    """Make iterator from structure."""
    if self.num_seqs > 0:
      self.iterator = tf.data.Iterator.from_structure(
        self.dataset.output_types, self.dataset.output_shapes)
    else:
      self.iterator = None


  def make_handle(self, sess):
    """Make iterator string handle."""
    if self.iterator is None:
      self.handle = None
    else:
      self.handle = sess.run(self.iterator.string_handle())


def order_tfrecords(tfr_pattern):
  """Check for TFRecords files fitting my pattern in succession,
     else return empty list."""
  tfr_files = []

  if tfr_pattern.count('*') == 1:
    i = 0
    tfr_file = tfr_pattern.replace('*', str(i))

    while os.path.isfile(tfr_file):
      tfr_files.append(tfr_file)
      i += 1
      tfr_file = tfr_pattern.replace('*', str(i))

  return tfr_files