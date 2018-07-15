#!/usr/bin/env python
from optparse import OptionParser
import os

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf

from basenji import tfrecord_batcher

'''
tfr_hdf5.py

Convert HDF5 training file to TFRecords.
'''

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <targets_file> <tfr_dir> <hdf5_file>'
  parser = OptionParser(usage)
  parser.add_option('-p', dest='processes', default=16, type='int',
      help='Number of parallel threads to use [Default: %default]')
  parser.add_option('-s', dest='shards', default=16, type='int',
      help='Number of sharded files to output per dataset [Default: %default]')
  parser.add_option('-l', dest='seq_len',
      type='int', default=131072,
      help='Sequence length [Default: %default]')
  parser.add_option('-w', dest='pool_width',
      default=128, type='int',
      help='Pool width [Default: %default]')
  (options,args) = parser.parse_args()

  if len(args) != 3:
    parser.error('Must provide targets file, TFRecords data directory, and output HDF5 file')
  else:
    targets_file = args[0]
    tfr_data_dir = args[1]
    hdf5_file = args[2]

  # read target datasets
  targets_df = pd.read_table(targets_file)

  # write to HDF5
  hdf5_out = h5py.File(hdf5_file, 'w')

  # store pooling
  hdf5_out.create_dataset('pool_width', data=options.pool_width, dtype='int')

  # store targets
  target_ids = np.array(targets_df.identifier, dtype='S')
  hdf5_out.create_dataset('target_ids', data=target_ids)

  target_labels = np.array(targets_df.description, dtype='S')
  hdf5_out.create_dataset('target_labels', data=target_labels)

  if 'strand' in targets_df.columns:
    target_strands = np.array(targets_df.strand, dtype='S')
  else:
    target_strands = np.array(['*']*targets_df.shape[0], dtype='S')
  hdf5_out.create_dataset('target_strands', data=target_strands)

  target_len = options.seq_len // options.pool_width

  for data_set in ['train', 'valid', 'test']:
    tfr_pattern = '%s/tfrecords/%s-*.tfr' % (tfr_data_dir, data_set)
    print(tfr_pattern)

    seqs_1hot, targets = read_tfr(tfr_pattern, target_len)
    print(' seqs_1hot', seqs_1hot.shape)
    print(' targets', targets.shape)

    hdf5_out.create_dataset(
        '%s_in' % data_set,
        data=seqs_1hot,
        dtype='bool')

    hdf5_out.create_dataset(
        '%s_out' % data_set,
        data=targets,
        dtype='float16')

    hdf5_out.create_dataset(
        '%s_na' % data_set,
        data=np.zeros((targets.shape[0], targets.shape[1]), dtype='bool'),
        dtype='bool')

  hdf5_out.close()


def read_tfr(tfr_pattern, target_len):
  tfr_files = tfrecord_batcher.order_tfrecords(tfr_pattern)
  if tfr_files:
    dataset = tf.data.Dataset.list_files(tf.constant(tfr_files), shuffle=False)
  else:
    dataset = tf.data.Dataset.list_files(tfr_pattern)
  dataset = dataset.flat_map(file_to_records)
  dataset = dataset.batch(1)
  dataset = dataset.map(parse_proto)

  iterator = dataset.make_one_shot_iterator()

  next_op = iterator.get_next()

  seqs_1hot = []
  targets = []

  with tf.Session() as sess:
    next_datum = sess.run(next_op)
    while next_datum:
      seq_1hot = next_datum['sequence'].reshape((-1,4))
      targets1 = next_datum['targets'].reshape(target_len,-1)

      seqs_1hot.append(seq_1hot)
      targets.append(targets1)

      try:
        next_datum = sess.run(next_op)
      except tf.errors.OutOfRangeError:
        next_datum = False

  seqs_1hot = np.array(seqs_1hot)
  targets = np.array(targets)

  return seqs_1hot, targets


def file_to_records(filename):
    return tf.data.TFRecordDataset(filename, compression_type='ZLIB')

def parse_proto(example_protos):
  features = {
    'sequence': tf.FixedLenFeature([], tf.string),
    'target': tf.FixedLenFeature([], tf.string)
  }
  parsed_features = tf.parse_example(example_protos, features=features)
  seq = tf.decode_raw(parsed_features['sequence'], tf.uint8)
  targets = tf.decode_raw(parsed_features['target'], tf.float16)
  return {'sequence': seq, 'targets': targets}


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
