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
      # dataset = tf.data.Dataset.list_files(tf.constant(tfr_files), shuffle=False)
      dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
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
        # dataset = tf.data.Dataset.list_files(tf.constant(tfr_files), shuffle=False)
        dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
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


class RnaDataset:
  def __init__(self, data_dir, split_label, batch_size,
               mode='eval', shuffle_buffer=1024):
    """Initialize basic parameters; run make_dataset."""

    self.data_dir = data_dir
    self.batch_size = batch_size
    self.shuffle_buffer = shuffle_buffer
    self.mode = mode
    self.split_label = split_label

    # read data parameters
    data_stats_file = '%s/statistics.json' % self.data_dir
    with open(data_stats_file) as data_stats_open:
      data_stats = json.load(data_stats_open)
    self.length_t = data_stats['length_t']

    # self.seq_depth = data_stats.get('seq_depth',4)
    self.target_length = data_stats['target_length']
    self.num_targets = data_stats['num_targets']

    if self.split_label == '*':
      self.num_seqs = 0
      for dkey in data_stats:
        if dkey[-5:] == '_seqs':
          self.num_seqs += data_stats[dkey]
    else:
      self.num_seqs = data_stats['%s_seqs' % self.split_label]

    self.make_dataset()

  def batches_per_epoch(self):
    return self.num_seqs // self.batch_size

  def make_parser(self): #, rna_mode
    def parse_proto(example_protos):
      """Parse TFRecord protobuf."""

      feature_spec = {
        'lengths': tf.io.FixedLenFeature((1,), tf.int64),
        'sequence': tf.io.FixedLenFeature([], tf.string),
        'coding': tf.io.FixedLenFeature([], tf.string),
        'splice': tf.io.FixedLenFeature([], tf.string),
        'targets': tf.io.FixedLenFeature([], tf.string)
      }

      # parse example into features
      feature_tensors = tf.io.parse_single_example(example_protos, features=feature_spec)

      # decode targets
      targets = tf.io.decode_raw(feature_tensors['targets'], tf.float16)
      targets = tf.cast(targets, tf.float32)

      # get length
      seq_lengths = feature_tensors['lengths']

      # decode sequence
      sequence = tf.io.decode_raw(feature_tensors['sequence'], tf.uint8)
      sequence = tf.one_hot(sequence, 4)
      sequence = tf.cast(sequence, tf.float32)

      # decode coding frame
      coding = tf.io.decode_raw(feature_tensors['coding'], tf.uint8)
      coding = tf.expand_dims(coding, axis=1)
      coding = tf.cast(coding, tf.float32)

      # decode splice
      splice = tf.io.decode_raw(feature_tensors['splice'], tf.uint8)
      splice = tf.expand_dims(splice, axis=1)
      splice = tf.cast(splice, tf.float32)

      # concatenate input tracks
      inputs = tf.concat([sequence,coding,splice], axis=1)
      # inputs = tf.concat([sequence,splice], axis=1)
      # inputs = tf.concat([sequence,coding], axis=1)

      # pad to zeros to full length
      paddings = [[0, self.length_t-seq_lengths[0]],[0,0]]
      inputs = tf.pad(inputs, paddings)

      return inputs, targets

    return parse_proto

  def make_dataset(self, cycle_length=4):
    """Make Dataset w/ transformations."""

    # collect tfrecords
    tfr_path = '%s/tfrecords/%s-*.tfr' % (self.data_dir, self.split_label)
    tfr_files = natsorted(glob.glob(tfr_path))

    # initialize tf.data
    if tfr_files:
      # dataset = tf.data.Dataset.list_files(tf.constant(tfr_files), shuffle=False)
      dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
    else:
      print('Cannot order TFRecords %s' % tfr_path, file=sys.stderr)
      dataset = tf.data.Dataset.list_files(tfr_path)

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

    # map records to examples
    dataset = dataset.map(self.make_parser()) #self.rna_mode

    # batch
    dataset = dataset.batch(self.batch_size)

    # prefetch
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # hold on
    self.dataset = dataset


  def numpy(self, return_inputs=True, return_outputs=True):
    """ Convert TFR inputs and/or outputs to numpy arrays."""
    with tf.name_scope('numpy'):
      # initialize dataset from TFRecords glob
      tfr_path = '%s/tfrecords/%s-*.tfr' % (self.data_dir, self.split_label)
      tfr_files = natsorted(glob.glob(tfr_path))
      if tfr_files:
        # dataset = tf.data.Dataset.list_files(tf.constant(tfr_files), shuffle=False)
        dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
      else:
        print('Cannot order TFRecords %s' % self.tfr_path, file=sys.stderr)
        dataset = tf.data.Dataset.list_files(self.tfr_path)

      # read TF Records
      dataset = dataset.flat_map(file_to_records)
      dataset = dataset.map(self.make_parser())
      dataset = dataset.batch(1)

    # initialize inputs and outputs
    seqs_1hot = []
    targets = []

    # collect inputs and outputs
    for seq_1hot, targets1 in dataset:
      # sequence
      if return_inputs:
        seqs_1hot.append(seq_1hot.numpy())

      # targets
      if return_outputs:
        targets.append(targets1.numpy())

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


class RnaDatasetVikram:
  def __init__(self, data_dir, batch_size, mode='eval', #, rna_mode
               shuffle_buffer=1024, split_labels=None):
    """Initialize basic parameters; run make_dataset."""

    self.data_dir = data_dir
    # self.rna_mode = rna_mode
    self.batch_size = batch_size
    self.shuffle_buffer = shuffle_buffer
    self.mode = mode
    self.split_labels = split_labels

    # read data parameters
    data_stats_file = '%s/statistics.json' % self.data_dir
    with open(data_stats_file) as data_stats_open:
      data_stats = json.load(data_stats_open)
    self.length_full = data_stats['length_full']

    # self.seq_depth = data_stats.get('seq_depth',4)
    self.target_length = data_stats['target_length']
    self.num_targets = data_stats['num_targets']

    self.num_seqs = 0
    for split_label in self.split_labels:
      self.num_seqs += data_stats['%s_seqs' % split_label]

    self.make_dataset()

  def batches_per_epoch(self):
    return self.num_seqs // self.batch_size

  def make_parser(self): #, rna_mode
    def parse_proto(example_protos):
      """Parse TFRecord protobuf."""

      feature_spec = {
        'lengths': tf.io.FixedLenFeature((1,), tf.int64),
        'sequence': tf.io.FixedLenFeature([], tf.string),
        'isoFreq': tf.io.FixedLenFeature([], tf.string),
        'frame': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'splice5p': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'splice3p': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'targets': tf.io.FixedLenFeature([], tf.string)
      }

      # parse example into features
      feature_tensors = tf.io.parse_single_example(example_protos, features=feature_spec)

      # decode targets
      targets = tf.io.decode_raw(feature_tensors['targets'], tf.float16)
      targets = tf.cast(targets, tf.float32)

      # decode targets
      # isoFreq = tf.io.decode_raw(feature_tensors['isoFreq'], tf.float16)
      # isoFreq = tf.cast(isoFreq, tf.float32)
      # isoFreq = tf.expand_dims(isoFreq, axis=1)

      # get lengths
      seq_lengths = feature_tensors['lengths']

      # decode sequence
      sequence = tf.io.decode_raw(feature_tensors['sequence'], tf.uint8)
      sequence = tf.one_hot(sequence, 4)
      sequence = tf.cast(sequence, tf.float32)

      # decode coding frame
      frame = feature_tensors['frame']
      frame = tf.one_hot(frame, 1)
      frame = tf.cast(frame, tf.float32)

      # decode 5' splice sites
      splice5p = feature_tensors['splice5p']
      splice5p = tf.one_hot(splice5p, 1)
      splice5p = tf.cast(splice5p, tf.float32)

      # decode 3' splice sites
      splice3p = feature_tensors['splice3p']
      splice3p = tf.one_hot(splice3p, 1)
      splice3p = tf.cast(splice3p, tf.float32)

      # concatenate input tracks
      inputs = tf.concat([sequence,frame,splice5p,splice3p], axis=1)

      # pad to zeros to full length
      paddings = [[0, self.length_full-seq_lengths[0]],[0,0]]
      inputs = tf.pad(inputs, paddings)

      return inputs, targets

    return parse_proto

  def make_dataset(self, cycle_length=4):
    """Make Dataset w/ transformations."""

    # collect tfrecords
    tfr_files = []
    for split_label in self.split_labels:
      tfr_path = '%s/tfrecords/%s-*.tfr' % (self.data_dir, split_label)
      tfr_files += natsorted(glob.glob(tfr_path))

    # initialize tf.data
    if tfr_files:
      # dataset = tf.data.Dataset.list_files(tf.constant(tfr_files), shuffle=False)
      dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
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

    # map records to examples
    dataset = dataset.map(self.make_parser()) #self.rna_mode

    # batch
    dataset = dataset.batch(self.batch_size)

    # prefetch
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # hold on
    self.dataset = dataset


class RnaDatasetV1:
  def __init__(self, data_dir, rna_mode, batch_size, mode='eval',
               shuffle_buffer=1024, split_labels=None):
    """Initialize basic parameters; run make_dataset."""

    self.data_dir = data_dir
    self.rna_mode = rna_mode
    self.batch_size = batch_size
    self.shuffle_buffer = shuffle_buffer
    self.mode = mode
    self.split_labels = split_labels

    # read data parameters
    data_stats_file = '%s/statistics.json' % self.data_dir
    with open(data_stats_file) as data_stats_open:
      data_stats = json.load(data_stats_open)
    self.length_utr5 = data_stats['length_utr5']
    self.length_cds = data_stats['length_cds']
    self.length_utr3 = data_stats['length_utr3']
    self.length_full = data_stats['length_full']
    
    self.seq_depth = data_stats.get('seq_depth',4)
    self.target_length = data_stats['target_length']
    self.num_targets = data_stats['num_targets']
    
    self.num_seqs = 0 
    for split_label in self.split_labels:
      self.num_seqs += data_stats['%s_seqs' % split_label]

    self.make_dataset()

  def batches_per_epoch(self):
    return self.num_seqs // self.batch_size


  def make_parser(self, rna_mode):
    def parse_proto(example_protos):
      """Parse TFRecord protobuf."""

      feature_spec = {
        'lengths': tf.io.FixedLenFeature((3,), tf.int64),
        'utr5': tf.io.FixedLenFeature([], tf.string),
        'cds': tf.io.FixedLenFeature([], tf.string),
        'utr3': tf.io.FixedLenFeature([], tf.string),
        'features': tf.io.FixedLenFeature([], tf.string), 
        'targets': tf.io.FixedLenFeature([], tf.string)
      }

      # parse example into features
      feature_tensors = tf.io.parse_single_example(example_protos, features=feature_spec)

      # decode targets
      targets = tf.io.decode_raw(feature_tensors['targets'], tf.float16)
      targets = tf.cast(targets, tf.float32)

      # decode targets
      features = tf.io.decode_raw(feature_tensors['features'], tf.float16)
      features = tf.cast(features, tf.float32)

      # get lengths
      seq_lengths = feature_tensors['lengths']

      # decode utr5
      if rna_mode in ['utr5','full','sep']:
        seq_utr5 = tf.io.decode_raw(feature_tensors['utr5'], tf.uint8)
        seq_utr5 = tf.one_hot(seq_utr5, 4)
        seq_utr5 = tf.cast(seq_utr5, tf.float32)

      # decode cds
      if rna_mode in ['cds','full','sep']:
        seq_cds = tf.io.decode_raw(feature_tensors['cds'], tf.uint8)
        seq_cds = tf.one_hot(seq_cds, 4)
        seq_cds = tf.cast(seq_cds, tf.float32)

      # decode utr3
      if rna_mode in ['utr3','full','sep']:
        seq_utr3 = tf.io.decode_raw(feature_tensors['utr3'], tf.uint8)
        seq_utr3 = tf.one_hot(seq_utr3, 4)
        seq_utr3 = tf.cast(seq_utr3, tf.float32)

      # concat and pad
      if rna_mode == 'utr5':
        paddings = [[0, self.length_utr5-seq_lengths[0]],[0,0]]
        seq_input = tf.pad(seq_utr5, paddings)

      elif rna_mode == 'cds':
        paddings = [[0, self.length_cds-seq_lengths[1]],[0,0]]
        seq_input = tf.pad(seq_cds, paddings)

      elif rna_mode == 'utr3':  
        paddings = [[0, self.length_utr3-seq_lengths[2]],[0,0]]
        seq_input = tf.pad(seq_utr3, paddings)

      elif rna_mode == 'full':
        # add fifth channel
        seq_utr5 = tf.pad(seq_utr5, [[0,0],[0,1]], constant_values=0)
        seq_cds = tf.pad(seq_cds, [[0,0],[0,1]], constant_values=1)
        seq_utr3 = tf.pad(seq_utr3, [[0,0],[0,1]], constant_values=0)

        # concat
        seq_full = tf.concat([seq_utr5,seq_cds,seq_utr3], axis=0)

        # pad length
        seq_full_length = tf.reduce_sum(seq_lengths)
        if seq_full_length <= self.length_full:
          paddings = [[0, self.length_full-seq_full_length],[0,0]]
          seq_input = tf.pad(seq_full, paddings)
        
        else:
          # trim off 5' UTR start
          seq_input = seq_full[-self.length_full:,:]

      elif rna_mode == 'sep':
        # utr5
        paddings = [[0, self.length_utr5-seq_lengths[0]],[0,0]]
        seq_utr5 = tf.pad(seq_utr5, paddings)

        # cds
        paddings = [[0, self.length_cds-seq_lengths[1]],[0,0]]
        seq_cds = tf.pad(seq_cds, paddings)

        # utr3
        paddings = [[0, self.length_utr3-seq_lengths[2]],[0,0]]
        seq_utr3 = tf.pad(seq_utr3, paddings)

        # tuple
        seq_input = (seq_utr5, seq_cds, seq_utr3)

      else:
        seq_input = None
        print('Cannot parse rna_mode: %s' % rna_mode, file=sys.stderr)
        exit(1)

      return (seq_input, features), targets

    return parse_proto


  def make_dataset(self, cycle_length=4):
    """Make Dataset w/ transformations."""

    # collect tfrecords
    tfr_files = []
    for split_label in self.split_labels:
      tfr_path = '%s/tfrecords/%s-*.tfr' % (self.data_dir, split_label)
      tfr_files += natsorted(glob.glob(tfr_path))

    # initialize tf.data
    if tfr_files:
      # dataset = tf.data.Dataset.list_files(tf.constant(tfr_files), shuffle=False)
      dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
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

    # map records to examples
    dataset = dataset.map(self.make_parser(self.rna_mode))

    # batch
    dataset = dataset.batch(self.batch_size)

    # prefetch
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # hold on
    self.dataset = dataset
