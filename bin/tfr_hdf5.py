#!/usr/bin/env python
from optparse import OptionParser
import glob
import os

import h5py
from natsort import natsorted
import numpy as np
import pandas as pd
import tensorflow as tf
if tf.__version__[0] == '1':
  tf.compat.v1.enable_eager_execution()

'''
tfr_hdf5.py

Convert TFRecords to HDF5.
'''

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <data_dir> <hdf5_file>'
  parser = OptionParser(usage)
  (options,args) = parser.parse_args()

  if len(args) != 2:
    parser.error('Must provide data directory and output HDF5 file')
  else:
    data_dir = args[0]
    hdf5_file = args[1]

  # read target datasets
  targets_file = '%s/targets.txt' % data_dir
  targets_df = pd.read_csv(targets_file, index_col=0, sep='\t')
  num_targets = targets_df.shape[0]

  # write to HDF5
  hdf5_out = h5py.File(hdf5_file, 'w')

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

  for data_set in ['train', 'valid', 'test']:
    tfr_pattern = '%s/tfrecords/%s-*.tfr' % (data_dir, data_set)
    print(tfr_pattern)

    seqs_1hot, targets = read_tfr(tfr_pattern, num_targets)
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

  hdf5_out.close()


def read_tfr(tfr_pattern, num_targets):
  tfr_files = natsorted(glob.glob(tfr_pattern))
  if tfr_files:
    dataset = tf.data.Dataset.list_files(tf.constant(tfr_files), shuffle=False)
  else:
    dataset = tf.data.Dataset.list_files(tfr_pattern)
  dataset = dataset.flat_map(file_to_records)
  dataset = dataset.map(parse_proto)
  dataset = dataset.batch(1)

  seqs_1hot = []
  targets = []

  si = 0
  for seq_1hot, targets1 in dataset:
    # TEMP!
    if si % 2 == 0:
      seq_1hot = seq_1hot.numpy()[0].astype('uint8')
      targets1 = targets1.numpy()[0].astype('float16')
      seq_1hot = seq_1hot.reshape((-1,4))
      targets1 = targets1.reshape((-1,num_targets))
      seqs_1hot.append(seq_1hot)
      targets.append(targets1)
    si += 1

  seqs_1hot = np.array(seqs_1hot, dtype='uint8')
  targets = np.array(targets, dtype='float16')

  return seqs_1hot, targets


def file_to_records(filename):
    return tf.data.TFRecordDataset(filename, compression_type='ZLIB')

def parse_proto(example_protos):
  features = {
    'sequence': tf.io.FixedLenFeature([], tf.string),
    'target': tf.io.FixedLenFeature([], tf.string)
  }
  parsed_features = tf.io.parse_single_example(example_protos, features=features)
  seq = tf.io.decode_raw(parsed_features['sequence'], tf.uint8)
  targets = tf.io.decode_raw(parsed_features['target'], tf.float16)
  return seq, targets


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
