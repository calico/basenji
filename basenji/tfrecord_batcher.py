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


import os
import sys

import tensorflow as tf

from basenji import dna_io
from basenji import ops
from basenji import tfrecord_util

# Multiplier for how many items to have in the shuffle buffer, invariant of how
# many files we're parallel-interleaving for our input datasets.
SHUFFLE_BUFFER_DEPTH_PER_FILE = 32
# Number of files to concurrently read from, and interleave, for our input
# datasets.
NUM_FILES_TO_PARALLEL_INTERLEAVE = 8

def tfrecord_dataset(tfr_data_files_pattern,
                     batch_size,
                     seq_length,
                     seq_depth,
                     target_length,
                     num_targets,
                     mode,
                     use_static_batch_size=False,
                     repeat=True):
  """Load TFRecord format data.

  The tf.Example assumed to be ZLIB compressed with fields:
    sequence: tf.string FixedLenFeature of length seq_length * seq_depth.
    label: tf.float32 FixedLenFeature of target_length * num_targets.

  Args:
   tfr_data_files_pattern: Pattern (potentially with globs) for TFRecord
     format files. See `tf.gfile.Glob` for more information.
    batch_size: batch_size
    seq_length: length of input sequence
    seq_depth: vocabulary size of the inputs (4 for raw DNA)
    target_length: length of the target sequence
    num_targets: number of targets at each target sequence location
j   mode: a tf.estimator.ModeKeys instance
    use_static_batch_size: whether to enforce that all batches have a fixed
      batch size. Note that for test data, where we don't take repeated passes,
      setting this to True will drop a few examples from the end of the dataset,
      if batch size doesn't divide the number of test examples, rather than
      causing an exception.
    repeat: repeat the training dataset
  Returns:
    A Dataset which will produce a dict with the following tensors:
      sequence: [batch_size, sequence_length, seq_depth]
      label: [batch_size, num_targets, target_length]
      na: [batch_size, num_targets]
  """

  tfr_files = order_tfrecords(tfr_data_files_pattern)
  if tfr_files:
    dataset = tf.data.Dataset.list_files(tf.constant(tfr_files), shuffle=False)
  else:
    print('Cannot order TFRecords %s' % tfr_data_files_pattern, file=sys.stderr)
    dataset = tf.data.Dataset.list_files(tfr_data_files_pattern)

  def file_to_records(filename):
    return tf.data.TFRecordDataset(filename, compression_type='ZLIB')

  if mode == tf.estimator.ModeKeys.TRAIN:  # Shuffle, repeat, and parallelize.
    # abstaining from repeat makes it easier to separate epochs
    if repeat:
      dataset = dataset.repeat()

    dataset = dataset.apply(
        # Interleaving allows us to pull one element from one file, and then
        # switch to pulling from another file, without fully reading the first.
        tf.contrib.data.parallel_interleave(
            map_func=file_to_records,
            # Magic number for cycle-length chosen by trying 64 (mentioned as a
            # best practice) and then noticing a significant bump in memory
            # usage. Reducing to 10 alleviated some of the memory pressure.
            cycle_length=NUM_FILES_TO_PARALLEL_INTERLEAVE, sloppy=True))

    # The shuffle applies to the whole dataset (after interleave), so it's
    # important that the shuffle buffer be larger than the number of files.
    # Otherwise, we'd go through each file individually.
    dataset = dataset.shuffle(buffer_size=NUM_FILES_TO_PARALLEL_INTERLEAVE *
                              SHUFFLE_BUFFER_DEPTH_PER_FILE)

  else:
    # Don't parallelize, shuffle, or repeat.
    # Use flat_map as opposed to map because file_to_records produces a dataset,
    # so with (non-flat) map, we'd have a dataset of datasets.
    dataset = dataset.flat_map(file_to_records)

  if batch_size is None:
    raise ValueError('batch_size is None')

  if use_static_batch_size:
    dataset = dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))
  else:
    dataset = dataset.batch(batch_size)

  def _parse(example_protos):
    features = {
        tfrecord_util.TFR_INPUT: tf.FixedLenFeature([], tf.string),
        tfrecord_util.TFR_OUTPUT: tf.FixedLenFeature([], tf.string),
    }

    parsed_features = tf.parse_example(example_protos, features=features)

    static_batch_size = batch_size if use_static_batch_size else -1

    seq = tf.decode_raw(parsed_features[tfrecord_util.TFR_INPUT], tf.uint8)
    seq = tf.reshape(seq, [static_batch_size, seq_length, seq_depth])
    seq = tf.cast(seq, tf.float32)

    label = tf.decode_raw(parsed_features[tfrecord_util.TFR_OUTPUT], tf.float16)
    label = tf.reshape(label, [static_batch_size, target_length, num_targets])
    label = tf.cast(label, tf.float32)

    if use_static_batch_size:
      na = tf.zeros(label.shape[:-1], dtype=tf.bool)
    else:
      na = tf.zeros(tf.shape(label)[:-1], dtype=tf.bool)

    return {'sequence': seq, 'label': label, 'na': na}

  dataset = dataset.map(_parse)

  return dataset

def make_data_ops(job,
                  files_pattern,
                  mode,
                  use_static_batch_size=False):
  """Get an iterator over your training data.

  Args:
    job: a dictionary of parsed parameters.
      See `basenji.google.params.read_job_params` for more information.
    batch_size: the batch size.
    files_pattern: A file path pattern that has your training data. For example,
      '/cns/sandbox/home/mlbileschi/brain/basenji/data/train/*'.
    mode: a tf.estimator.ModeKeys instance.
    use_static_batch_size: whether to enforce that all batches have a fixed
      batch size. Note that for test data, where we don't take repeated passes,
      setting this to True will drop a few examples from the end of the data
  """
  if len(tf.gfile.Glob(files_pattern)) == 0:
    raise ValueError('0 files matched files_pattern ' + files_pattern + '.')

  batcher = tfrecord_dataset(
      files_pattern,
      job['batch_size'],
      job['seq_length'],
      job['seq_depth'],
      job['target_length'],
      job['num_targets'],
      mode=mode,
      use_static_batch_size=use_static_batch_size)

  return batcher.make_one_shot_iterator().get_next()


def num_possible_augmentations(augment_with_complement, shift_augment_offsets):
  # The value of the parameter shift_augment_offsets are the set of things to
  # _augment_ the original data with, and we want to, in addition to including
  # those augmentations, actually include the original data.
  if shift_augment_offsets:
    total_set_of_shifts = [0] + shift_augment_offsets
  else:
    total_set_of_shifts = [0]

  num_augments = 2 if augment_with_complement else 1
  num_augments *= len(total_set_of_shifts)
  return num_augments


def make_input_fn(job, data_file_pattern, mode, use_static_batch_size):
  """Makes an input_fn, according to the `Experiment` `input_fn` interface.

  Args:
    job: a dictionary of parsed parameters.
      See `basenji.google.params.read_job_params` for more information.
    data_file_pattern: A file path pattern that has your training data.
      For example, '/cns/sandbox/home/mlbileschi/brain/basenji/data/train/*'.
    mode: a tf.estimator.ModeKeys instance.
    use_static_batch_size: whether to enforce that all batches have a fixed
      batch size. Note that for test data, where we don't take repeated passes,
      setting this to True will drop a few examples from the end of the data
  Returns:
    input_fn to be used by Experiment
  """

  def _input_fn():
    shuffle = mode == tf.contrib.learn.ModeKeys.TRAIN
    data_ops = make_data_ops(job, data_file_pattern, mode,
                             use_static_batch_size)
    features = {'sequence': data_ops['sequence']}
    labels = {'label': data_ops['label'], 'na': data_ops['na']}
    return features, labels

  return _input_fn


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


# TODO(dbelanger) Remove this functionality.
class TFRecordBatcher(object):
  """Load TFRecord format data. Many args are unused and for API-compatibility.

     Args:
       tfr_data_file_pattern: Pattern (potentially with globs) for TFRecord
         format files. See `tf.gfile.Glob` for more information.
       load_targets: whether to load targets (unused)
       seq_length: length of the input sequences
       seq_depth: vocabulary size of the inputs (4 for raw DNA)
       target_length: length of the target sequence
       num_targets: number of targets at each target sequence location
       mode: a tf.estimator.ModeKeys instance
       NAf: (unused)
       batch_size: batch_size
       pool_width: width of pooling layers (unused)
       shuffle: whether the batcher should shuffle the data
  """

  def __init__(self,
               tfr_data_file_pattern,
               load_targets,
               seq_length,
               seq_depth,
               target_length,
               num_targets,
               mode,
               NAf=None,
               batch_size=64,
               pool_width=1,
               shuffle=False):

    self.session = None

    filenames = tf.gfile.Glob(tfr_data_file_pattern)

    dataset = tfrecord_dataset(filenames, batch_size, seq_length, seq_depth,
                               target_length, num_targets, mode)

    self.iterator = dataset.make_initializable_iterator()
    self._next_element = self.iterator.get_next()

  def initialize(self, sess):
    sess.run(self.iterator.initializer)

  def next(self, rc=False, shift=0):
    try:
      d = self.session.run(self._next_element)

      Xb = d['sequence']
      Yb = d['label']
      NAb = d['na']
      Nb = Xb.shape[0]

      # reverse complement
      if rc:
        if Xb is not None:
          Xb = dna_io.hot1_augment(Xb, rc, shift)
        if Yb is not None:
          Yb = Yb[:, ::-1, :]
        if NAb is not None:
          NAb = NAb[:, ::-1]

      return Xb, Yb, NAb, Nb

    except tf.errors.OutOfRangeError:
      return None, None, None, None

  def reset(self):
    return self.initialize(self.session)
