#!/usr/bin/env python
from optparse import OptionParser
import multiprocessing
import os
import pdb

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns

from basenji import tfrecord_batcher

'''
tfr_qc.py

Print quality control statistics for a TFRecords dataset.
'''

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <tfr_data_dir>'
  parser = OptionParser(usage)
  parser.add_option('-l', dest='target_length',
      default=1024, type='int')
  parser.add_option('-o', dest='out_dir', default='tfr_qc')
  parser.add_option('-p', dest='processes', default=16, type='int',
      help='Number of parallel threads to use [Default: %default]')
  parser.add_option('-s', dest='split', default='test')
  (options,args) = parser.parse_args()

  if len(args) != 1:
    parser.error('Must provide TFRecords data directory')
  else:
    tfr_data_dir = args[0]

  if not os.path.isdir(options.out_dir):
    os.mkdir(options.out_dir)

  # read target datasets
  targets_file = '%s/targets.txt' % tfr_data_dir
  targets_df = pd.read_table(targets_file)

  # read target values
  tfr_pattern = '%s/tfrecords/%s-*.tfr' % (tfr_data_dir, options.split)
  targets = read_tfr(tfr_pattern, options.target_length)

  pdb.set_trace()

  # compute stats
  target_means = np.mean(targets, axis=(0,1,2), dtype='float64')
  target_min = np.min(targets, axis=(0,1,2))
  target_max = np.max(targets, axis=(0,1,2))

  # print statistics for each target
  table_out = open('%s/table.txt' % options.out_dir, 'w')
  for ti in range(targets.shape[-1]):
    cols = (ti, target_means[ti], target_min[ti], target_max[ti], targets_df.identifier[ti], targets_df.description[ti])
    print('%-4d  %8.3f  %7.3f  %7.3f  %16s  %s' % cols, file=table_out)
  table_out.close()

  """
  # plot distributions for each target
  distr_dir = '%s/distr' % options.out_dir
  if not os.path.isdir(distr_dir):
    os.mkdir(distr_dir)

  # initialize multiprocessing pool
  pool = multiprocessing.Pool(options.processes)

  plot_distr_args = []
  for ti in range(targets.shape[-1]):
    targets_ti = np.random.choice(targets[:,:,ti].flatten(), size=10000, replace=False)
    plot_distr_args.append((targets_ti, '%s/t%d.pdf' % (distr_dir,ti)))

  pool.starmap(plot_distr, plot_distr_args)
  """

def plot_distr(targets_ti, out_pdf):
  plt.figure()
  sns.distplot(targets_ti)
  plt.savefig(out_pdf)
  plt.close()


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

  targets = []

  with tf.Session() as sess:
    next_datum = sess.run(next_op)
    while next_datum:
      targets1 = next_datum['targets'].reshape(target_len,target_len,-1)
      targets.append(targets1)

      try:
        next_datum = sess.run(next_op)
      except tf.errors.OutOfRangeError:
        next_datum = False

  return np.array(targets)


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
