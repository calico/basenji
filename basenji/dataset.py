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

# Multiplier for how many items to have in the shuffle buffer, invariant
# of how many files we're parallel-interleaving for our input datasets.
SHUFFLE_BUFFER_DEPTH_PER_FILE = 8
# Number of files to concurrently read from, and interleave,
# for our input datasets.
NUM_FILES_TO_PARALLEL_INTERLEAVE = 4

# TFRecord constants
TFR_INPUT = 'sequence'
TFR_OUTPUT = 'target'
TFR_GENOME = 'genome'

def file_to_records(filename):
  return tf.data.TFRecordDataset(filename, compression_type='ZLIB')


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


class SeqDataset:
  def __init__(self, tfr_pattern, batch_size, seq_length,
               target_length, mode, seq_end_ignore=0):
    """Initialize basic parameters; run compute_stats; run make_dataset."""

    self.tfr_pattern = tfr_pattern

    self.num_seqs = None
    self.batch_size = batch_size
    self.seq_length = seq_length
    self.seq_end_ignore = seq_end_ignore
    self.seq_depth = None
    self.target_length = target_length
    self.num_targets = None

    self.mode = mode

    self.compute_stats()
    self.make_dataset()


  def batches_per_epoch(self):
    return self.num_seqs // self.batch_size


  def generate_parser(self, raw=False):

    def parse_proto(example_protos):
      """Parse TFRecord protobuf."""

      # features = {
      #   TFR_GENOME: tf.io.FixedLenFeature([1], tf.int64),
      #   TFR_INPUT: tf.io.FixedLenFeature([], tf.string),
      #   TFR_OUTPUT: tf.io.FixedLenFeature([], tf.string)
      # }
      features = {
        TFR_INPUT: tf.io.FixedLenFeature([], tf.string),
        TFR_OUTPUT: tf.io.FixedLenFeature([], tf.string)
      }
      parsed_features = tf.io.parse_single_example(example_protos, features=features)

      # genome = parsed_features[TFR_GENOME]

      sequence = tf.io.decode_raw(parsed_features[TFR_INPUT], tf.uint8)
      if not raw:
        sequence = tf.reshape(sequence, [self.seq_length, self.seq_depth])
        sequence = tf.cast(sequence, tf.float32)

      targets = tf.io.decode_raw(parsed_features[TFR_OUTPUT], tf.float16)
      if not raw:
        targets = tf.reshape(targets, [self.target_length, self.num_targets])

        if self.seq_end_ignore > 0:
          target_pool = self.seq_length // self.target_length
          slice_left = self.seq_end_ignore // target_pool
          slice_right = self.target_length - slice_left
          targets = targets[slice_left:slice_right, :]

        targets = tf.cast(targets, tf.float32)

      # return (sequence, genome), targets
      return sequence, targets

    return parse_proto


  def make_dataset(self):
    """Make Dataset w/ transformations."""

    # initialize dataset from TFRecords globf
    tfr_files = order_tfrecords(self.tfr_pattern)
    if tfr_files:
      dataset = tf.data.Dataset.list_files(tf.constant(tfr_files), shuffle=False)
    else:
      print('Cannot order TFRecords %s' % self.tfr_pattern, file=sys.stderr)
      dataset = tf.data.Dataset.list_files(self.tfr_pattern)

    # train
    if self.mode == tf.estimator.ModeKeys.TRAIN:
      # repeat
      dataset = dataset.repeat()

      # interleave files
      dataset = dataset.interleave(
        map_func=file_to_records,
        cycle_length=NUM_FILES_TO_PARALLEL_INTERLEAVE,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

      # shuffle
      shuffle_buffer_size = NUM_FILES_TO_PARALLEL_INTERLEAVE * SHUFFLE_BUFFER_DEPTH_PER_FILE
      dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    # valid/test
    else:
      # flat mix files
      dataset = dataset.flat_map(file_to_records)

    # helper for training on single genomes in a multiple genome mode
    if self.num_seqs > 0:
      dataset = dataset.map(self.generate_parser())

    # batch
    dataset = dataset.batch(self.batch_size)

    # hold on
    self.dataset = dataset


  def compute_stats(self):
    """ Iterate over the TFRecords to count sequences, and infer
        seq_depth and num_targets."""

    # read TF Records
    dataset = tf.data.Dataset.list_files(self.tfr_pattern)
    dataset = dataset.flat_map(file_to_records)
    dataset = dataset.map(self.generate_parser(raw=True))
    dataset = dataset.batch(1)

    self.num_seqs = 0
    # for (seq_raw, genome), targets_raw in dataset:
    for seq_raw, targets_raw in dataset:
      # infer seq_depth
      seq_1hot = seq_raw.numpy().reshape((self.seq_length,-1))
      if self.seq_depth is None:
        self.seq_depth = seq_1hot.shape[-1]
      else:
        assert(self.seq_depth == seq_1hot.shape[-1])

      # infer num_targets
      targets1 = targets_raw.numpy().reshape(self.target_length,-1)
      if self.num_targets is None:
        self.num_targets = targets1.shape[-1]
        targets_nonzero = (targets1.sum(axis=0, dtype='float32') > 0)
      else:
        assert(self.num_targets == targets1.shape[-1])
        targets_nonzero = np.logical_or(targets_nonzero, targets1.sum(axis=0, dtype='float32') > 0)

      # count sequences
      self.num_seqs += 1

    # warn user about nonzero targets
    if self.num_seqs > 0:
      self.num_targets_nonzero = (targets_nonzero > 0).sum()
      print('%s has %d sequences with %d/%d targets' % (self.tfr_pattern, self.num_seqs, self.num_targets_nonzero, self.num_targets), flush=True)
    else:
      self.num_targets_nonzero = None
      print('%s has %d sequences with 0 targets' % (self.tfr_pattern, self.num_seqs), flush=True)


  def numpy(self, return_inputs=True, return_outputs=True):
    """ Convert TFR inputs and/or outputs to numpy arrays."""

    # read TF Records
    dataset = tf.data.Dataset.list_files(self.tfr_pattern)
    dataset = dataset.flat_map(file_to_records)
    dataset = dataset.map(self.generate_parser())
    dataset = dataset.batch(1)

    # initialize inputs and outputs
    seqs_1hot = []
    targets = []

    # collect inputs and outputs
    for seq1_1hot, targets1 in dataset:
      if return_inputs:
        seqs_1hot.append(seq1_1hot)
      if return_outputs:
        targets.append(targets1)

    # make arrays
    seqs_1hot = np.array(seqs_1hot)
    targets = np.array(targets)

    # return
    if return_inputs and return_outputs:
      return seqs_1hot, targets
    elif return_inputs:
      return seqs_1hot
    else:
      return targets
