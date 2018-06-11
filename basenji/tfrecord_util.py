"""Utility code for converty h5 files to TFRecords"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

TFR_INPUT = 'sequence'
TFR_OUTPUT = 'target'

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_tfrecord(seqs, targets, output_file):
  """Write data out as a TFrecord File containing TFExamples.
  Args:
    seqs: [num_examples, input_sequence_length, 4] onehot-encoded np.bool array
    targets: [num_examples, output_sequence_length, num_output_targets]
      np.float16 array.
    output_file: where to write output TFRecord.
  Raises:
    ValueError if input data does not have the correct shape or type.

  The output TFRecord file contains TFExample protos for each example. It is
  ZLIB compressed. The data can be loaded using
  tfrecord_batcher.tfrecord_dataset. The best way to see how the data are stored
  is to see how they are loaded in the _parse function in tfrecord_dataset(...).
  The input seqs data are stored as a flattened 1D array that has been converted
  to raw bytes. The target data are stored as a flattened Float list.
  """

  if len(seqs.shape) != 3 or seqs.shape[2] != 4 or  seqs.dtype != np.bool:
    raise ValueError('Input seqs should be '
                     '[num_examples, input_sequence_length, 4]'
                     ' onehot-encoded np.bool array')

  if len(targets.shape) != 3 or targets.dtype != np.float16:
    raise ValueError('Target seqs should be '
                     '[num_examples, output_sequence_length, num_output_targets]'
                      ' np.float16 array.')

  options = tf.python_io.TFRecordOptions(
      tf.python_io.TFRecordCompressionType.ZLIB)
  with tf.python_io.TFRecordWriter(output_file, options) as writer:
    for d in xrange(seqs.shape[0]):
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  TFR_INPUT:
                      _bytes_feature(seqs[d, :, :].flatten().tostring()),
                  TFR_OUTPUT:
                      _bytes_feature(targets[d, :, :].flatten().tostring())
              }))
      writer.write(example.SerializeToString())
