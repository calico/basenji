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
import glob
import json
import os
import pdb
import sys

from natsort import natsorted
import numpy as np
import tensorflow as tf

# TFRecord constants
TFR_INPUT = 'sequence'
TFR_OUTPUT = 'target'

def file_to_records(filename):
  return tf.data.TFRecordDataset(filename, compression_type='ZLIB')


# class SeqDataset:
#   def __init__(self, tfr_pattern, seq_length, seq_depth=4, seq_length_crop=None,
#                target_length=None, num_targets=None, batch_size=1,
#                mode=tf.estimator.ModeKeys.EVAL, compute_stats=True):
#     """Initialize basic parameters; run compute_stats; run make_dataset."""

#     self.tfr_pattern = tfr_pattern

#     self.num_seqs = None
#     self.batch_size = batch_size
#     self.seq_length = seq_length
#     self.seq_length_crop = seq_length_crop
#     self.seq_depth = seq_depth
#     self.target_length = target_length
#     self.num_targets = num_targets

#     self.mode = mode

#     if compute_stats:
#       self.compute_stats()
#     self.make_dataset()

class SeqDataset:
  def __init__(self, data_dir, split_label, batch_size, shuffle_buffer=128,
               seq_length_crop=None, mode='eval', tfr_pattern=None):
    """Initialize basic parameters; run compute_stats; run make_dataset."""

    self.data_dir = data_dir
    self.split_label = split_label
    self.batch_size = batch_size
    self.shuffle_buffer = shuffle_buffer
    self.seq_length_crop = seq_length_crop
    self.mode = mode
    self.tfr_pattern = tfr_pattern

    # read data parameters
    data_stats_file = '%s/statistics.json' % self.data_dir
    with open(data_stats_file) as data_stats_open:
      data_stats = json.load(data_stats_open)
    self.seq_length = data_stats['seq_length']
    
    self.seq_depth = data_stats.get('seq_depth',4)
    self.target_length = data_stats['target_length']
    self.num_targets = data_stats['num_targets']
    
    if self.tfr_pattern is None:
      self.tfr_path = '%s/tfrecords/%s-*.tfr' % (self.data_dir, self.split_label)
      self.num_seqs = data_stats['%s_seqs' % self.split_label]
    else:
      self.tfr_path = '%s/tfrecords/%s' % (self.data_dir, self.tfr_pattern)
      self.compute_stats()

    self.make_dataset()

  def batches_per_epoch(self):
    return self.num_seqs // self.batch_size

  def distribute(self, strategy):
    self.dataset = strategy.experimental_distribute_dataset(self.dataset)

  def generate_parser(self, raw=False):
    def parse_proto(example_protos):
      """Parse TFRecord protobuf."""

      # define features
      features = {
        TFR_INPUT: tf.io.FixedLenFeature([], tf.string),
        TFR_OUTPUT: tf.io.FixedLenFeature([], tf.string)
      }

      # parse example into features
      parsed_features = tf.io.parse_single_example(example_protos, features=features)

      # decode sequence
      sequence = tf.io.decode_raw(parsed_features[TFR_INPUT], tf.uint8)
      if not raw:
        sequence = tf.reshape(sequence, [self.seq_length, self.seq_depth])
        if self.seq_length_crop is not None:
          crop_len = (self.seq_length - self.seq_length_crop) // 2
          sequence = sequence[crop_len:-crop_len,:]
        sequence = tf.cast(sequence, tf.float32)

      # decode targets
      targets = tf.io.decode_raw(parsed_features[TFR_OUTPUT], tf.float16)
      if not raw:
        targets = tf.reshape(targets, [self.target_length, self.num_targets])
        targets = tf.cast(targets, tf.float32)

      return sequence, targets

    return parse_proto

  def make_dataset(self, cycle_length=4):
    """Make Dataset w/ transformations."""

    # initialize dataset from TFRecords glob
    tfr_files = natsorted(glob.glob(self.tfr_path))
    if tfr_files:
      dataset = tf.data.Dataset.list_files(tf.constant(tfr_files), shuffle=False)
    else:
      print('Cannot order TFRecords %s' % self.tfr_path, file=sys.stderr)
      dataset = tf.data.Dataset.list_files(self.tfr_path)

    # train
    if self.mode == 'train':
      # repeat
      dataset = dataset.repeat()

      # interleave files
      dataset = dataset.interleave(map_func=file_to_records,
        cycle_length=cycle_length,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

      # shuffle
      dataset = dataset.shuffle(buffer_size=self.shuffle_buffer,
        reshuffle_each_iteration=True)

    # valid/test
    else:
      # flat mix files
      dataset = dataset.flat_map(file_to_records)

    # (no longer necessary in tf2?)
    # helper for training on single genomes in a multiple genome mode
    # if self.num_seqs > 0:
    #  dataset = dataset.map(self.generate_parser())
    dataset = dataset.map(self.generate_parser())

    # cache (runs OOM)
    # dataset = dataset.cache()

    # batch
    dataset = dataset.batch(self.batch_size)

    # prefetch
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # hold on
    self.dataset = dataset


  def compute_stats(self):
    """ Iterate over the TFRecords to count sequences, and infer
        seq_depth and num_targets."""
    with tf.name_scope('stats'):
      # read TF Records
      dataset = tf.data.Dataset.list_files(self.tfr_path)
      dataset = dataset.flat_map(file_to_records)
      dataset = dataset.map(self.generate_parser(raw=True))
      dataset = dataset.batch(1)

    self.num_seqs = 0
    if self.num_targets is not None:
      targets_nonzero = np.zeros(self.num_targets, dtype='bool')

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
        targets_nonzero = ((targets1 != 0).sum(axis=0) > 0)
      else:
        assert(self.num_targets == targets1.shape[-1])
        targets_nonzero = np.logical_or(targets_nonzero, (targets1 != 0).sum(axis=0) > 0)

      # count sequences
      self.num_seqs += 1

    # warn user about nonzero targets
    if self.num_seqs > 0:
      self.num_targets_nonzero = (targets_nonzero > 0).sum()
      print('%s has %d sequences with %d/%d targets' % (self.tfr_path, self.num_seqs, self.num_targets_nonzero, self.num_targets), flush=True)
    else:
      self.num_targets_nonzero = None
      print('%s has %d sequences with 0 targets' % (self.tfr_path, self.num_seqs), flush=True)


  def numpy(self, return_inputs=True, return_outputs=True, step=1):
    """ Convert TFR inputs and/or outputs to numpy arrays."""
    with tf.name_scope('numpy'):
      # initialize dataset from TFRecords glob
      tfr_files = natsorted(glob.glob(self.tfr_path))
      if tfr_files:
        dataset = tf.data.Dataset.list_files(tf.constant(tfr_files), shuffle=False)
      else:
        print('Cannot order TFRecords %s' % self.tfr_path, file=sys.stderr)
        dataset = tf.data.Dataset.list_files(self.tfr_path)

      # read TF Records
      dataset = dataset.flat_map(file_to_records)
      dataset = dataset.map(self.generate_parser(raw=True))
      dataset = dataset.batch(1)

    # initialize inputs and outputs
    seqs_1hot = []
    targets = []

    # collect inputs and outputs
    for seq_raw, targets_raw in dataset:
      # sequence
      if return_inputs:
        seq_1hot = seq_raw.numpy().reshape((self.seq_length,-1))
        if self.seq_length_crop is not None:
          crop_len = (self.seq_length - self.seq_length_crop) // 2
          seq_1hot = seq_1hot[crop_len:-crop_len,:]
        seqs_1hot.append(seq_1hot)

      # targets
      if return_outputs:
        targets1 = targets_raw.numpy().reshape((self.target_length,-1))
        if step > 1:
          step_i = np.arange(0, self.target_length, step)
          targets1 = targets1[step_i,:]
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
