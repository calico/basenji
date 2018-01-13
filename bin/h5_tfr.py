#!/usr/bin/env python
from optparse import OptionParser

import multiprocessing
import os

import h5py
import tensorflow as tf

'''
h5_tfr.py

Convert HDF5 training file to TFRecords.
'''

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <h5_file> <tfr_dir>'
  parser = OptionParser(usage)
  parser.add_option('-p', dest='processes', default=16, type='int',
                    help='Number of parallel threads to use [Default: %default]')
  parser.add_option('-s', dest='shards', default=16, type='int',
                    help='Number of sharded files to output per dataset [Default: %default]')
  (options,args) = parser.parse_args()

  if len(args) != 2:
    parser.error('Must provide HDF5 file and TFRecords output directory')
  else:
    h5_file = args[0]
    tfr_dir = args[1]

  # define options
  tf_opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)

  # setup TFRecords output directory
  if not os.path.isdir(tfr_dir):
    os.mkdir(tfr_dir)

  # initialize multiprocessing pool
  pool = multiprocessing.Pool(options.processes)

  for dataset in ['train', 'valid', 'test']:
    tfr_files = ['%s/%s-%d.tfr' % (tfr_dir,dataset,wi) for wi in range(options.processes)]

    writer_args = [(tfr_files[wi], tf_opts, h5_file, dataset, si, options.shards) for si in range(options.shards)]

    pool.starmap(writer_worker, writer_args)


def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def writer_worker(tfr_file, tf_opts, h5_file, dataset, shard_i, num_shards):
  h5_open = h5py.File(h5_file)
  data_in = h5_open['%s_in'%dataset]
  data_out = h5_open['%s_out'%dataset]

  with tf.python_io.TFRecordWriter(tfr_file, tf_opts) as writer:
    for si in range(data_in.shape[0]):
      if si % num_shards == shard_i:
        example = tf.train.Example(features=tf.train.Features(feature={
            'sequence': _bytes_feature(data_in[si,:,:].flatten().tostring()),
            'target': _float_feature(data_out[si,:,:].flatten())}))
        writer.write(example.SerializeToString())

  h5_open.close()

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
