#!/usr/bin/env python
# Copyright 2017 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
from __future__ import print_function

from optparse import OptionParser
import pickle
import os
import sys
import threading
import time

import h5py
import numpy as np
import pandas as pd
import pysam
import tensorflow as tf
try:
  import zarr
except ImportError:
  pass

from basenji import params
from basenji import seqnn
from basenji import vcf as bvcf

from basenji_test import bigwig_open

'''
basenji_sad.py

Compute SNP Activity Difference (SAD) scores for SNPs in a VCF file.
'''

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <model_file> <vcf_file>'
  parser = OptionParser(usage)
  parser.add_option('-c', dest='csv',
      default=False, action='store_true',
      help='Print table as CSV [Default: %default]')
  parser.add_option('--cpu', dest='cpu',
      default=False, action='store_true',
      help='Run without a GPU [Default: %default]')
  parser.add_option('-f', dest='genome_fasta',
      default='%s/data/hg19.fa' % os.environ['BASENJIDIR'],
      help='Genome FASTA for sequences [Default: %default]')
  parser.add_option('--local', dest='local',
      default=1024, type='int',
      help='Local SAD score [Default: %default]')
  parser.add_option('-n', dest='norm_file',
      default=None,
      help='Normalize SAD scores')
  parser.add_option('-o',dest='out_dir',
      default='sad',
      help='Output directory for tables and plots [Default: %default]')
  parser.add_option('-p', dest='processes',
      default=None, type='int',
      help='Number of processes, passed by multi script')
  parser.add_option('--pseudo', dest='log_pseudo',
      default=1, type='float',
      help='Log2 pseudocount [Default: %default]')
  parser.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Average forward and reverse complement predictions [Default: %default]')
  parser.add_option('--shifts', dest='shifts',
      default='0', type='str',
      help='Ensemble prediction shifts [Default: %default]')
  parser.add_option('--stats', dest='sad_stats',
      default='SAD',
      help='Comma-separated list of stats to save. [Default: %default]')
  parser.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')
  parser.add_option('--ti', dest='track_indexes',
      default=None, type='str',
      help='Comma-separated list of target indexes to output BigWig tracks')
  parser.add_option('--txt', dest='out_txt',
    default=False, action='store_true',
    help='Output stats to text table [Default: %default]')
  parser.add_option('-u', dest='penultimate',
      default=False, action='store_true',
      help='Compute SED in the penultimate layer [Default: %default]')
  parser.add_option('-z', dest='out_zarr',
      default=False, action='store_true',
      help='Output stats to sad.zarr [Default: %default]')
  (options, args) = parser.parse_args()

  if len(args) == 3:
    # single worker
    params_file = args[0]
    model_file = args[1]
    vcf_file = args[2]

  elif len(args) == 5:
    # multi worker
    options_pkl_file = args[0]
    params_file = args[1]
    model_file = args[2]
    vcf_file = args[3]
    worker_index = int(args[4])

    # load options
    options_pkl = open(options_pkl_file, 'rb')
    options = pickle.load(options_pkl)
    options_pkl.close()

    # update output directory
    options.out_dir = '%s/job%d' % (options.out_dir, worker_index)

  else:
    parser.error('Must provide parameters and model files and QTL VCF file')

  if not os.path.isdir(options.out_dir):
    os.mkdir(options.out_dir)

  if options.track_indexes is None:
    options.track_indexes = []
  else:
    options.track_indexes = [int(ti) for ti in options.track_indexes.split(',')]
    if not os.path.isdir('%s/tracks' % options.out_dir):
      os.mkdir('%s/tracks' % options.out_dir)

  options.shifts = [int(shift) for shift in options.shifts.split(',')]
  options.sad_stats = options.sad_stats.split(',')


  #################################################################
  # read parameters

  job = params.read_job_params(params_file, require=['seq_length','num_targets'])

  if options.targets_file is None:
    target_ids = ['t%d' % ti for ti in range(job['num_targets'])]
    target_labels = ['']*len(target_ids)
    target_subset = None

  else:
    targets_df = pd.read_csv(options.targets_file, sep='\t', index_col=0)
    target_ids = targets_df.identifier
    target_labels = targets_df.description
    target_subset = targets_df.index
    if len(target_subset) == job['num_targets']:
        target_subset = None


  #################################################################
  # load SNPs

  # filter for worker SNPs
  if options.processes is not None:
    # determine boundaries
    num_snps = bvcf.vcf_count(vcf_file)
    worker_bounds = np.linspace(0, num_snps, options.processes+1, dtype='int')

    # read SNPs form VCF
    snps = bvcf.vcf_snps(vcf_file, start_i=worker_bounds[worker_index], end_i=worker_bounds[worker_index+1])

  else:
    # read SNPs form VCF
    snps = bvcf.vcf_snps(vcf_file)

  # open genome FASTA
  genome_open = pysam.Fastafile(options.genome_fasta)

  def snp_gen():
    for snp in snps:
      # get SNP sequences
      snp_1hot_list = bvcf.snp_seq1(snp, job['seq_length'], genome_open)

      for snp_1hot in snp_1hot_list:
        yield {'sequence':snp_1hot}

  snp_types = {'sequence': tf.float32}
  snp_shapes = {'sequence': tf.TensorShape([tf.Dimension(job['seq_length']),
                                            tf.Dimension(4)])}

  dataset = tf.data.Dataset.from_generator(snp_gen,
                                           output_types=snp_types,
                                           output_shapes=snp_shapes)
  dataset = dataset.batch(job['batch_size'])
  dataset = dataset.prefetch(2*job['batch_size'])
  if not options.cpu:
    dataset = dataset.apply(tf.contrib.data.prefetch_to_device('/device:GPU:0'))

  iterator = dataset.make_one_shot_iterator()
  data_ops = iterator.get_next()


  #################################################################
  # setup model

  # build model
  t0 = time.time()
  model = seqnn.SeqNN()
  model.build_sad(job, data_ops,
                  ensemble_rc=options.rc, ensemble_shifts=options.shifts,
                  embed_penultimate=options.penultimate, target_subset=target_subset)
  print('Model building time %f' % (time.time() - t0), flush=True)

  if options.penultimate:
    # labels become inappropriate
    target_ids = ['']*model.hp.cnn_filters[-1]
    target_labels = target_ids

  # read target normalization factors
  target_norms = np.ones(len(target_labels))
  if options.norm_file is not None:
    ti = 0
    for line in open(options.norm_file):
      target_norms[ti] = float(line.strip())
      ti += 1

  num_targets = len(target_ids)

  #################################################################
  # setup output

  if options.out_zarr:
    sad_out = initialize_output_zarr(options.out_dir, options.sad_stats,
                                     snps, target_ids, target_labels)

  elif options.out_txt:
    header_cols = ('rsid', 'ref', 'alt',
                  'ref_pred', 'alt_pred', 'sad', 'sar', 'geo_sad',
                  'ref_lpred', 'alt_lpred', 'lsad', 'lsar',
                  'ref_xpred', 'alt_xpred', 'xsad', 'xsar',
                  'target_index', 'target_id', 'target_label')

    if options.csv:
      sad_out = open('%s/sad_table.csv' % options.out_dir, 'w')
      print(','.join(header_cols), file=sad_out)
    else:
      sad_out = open('%s/sad_table.txt' % options.out_dir, 'w')
      print(' '.join(header_cols), file=sad_out)

  else:
    sad_out = initialize_output_h5(options.out_dir, options.sad_stats,
                                   snps, target_ids, target_labels)


  #################################################################
  # process

  szi = 0
  sum_write_thread = None
  sw_batch_size = 32 // job['batch_size']

  # initialize saver
  saver = tf.train.Saver()
  with tf.Session() as sess:
    # load variables into session
    saver.restore(sess, model_file)

    # predict first
    batch_preds = model.predict_tfr(sess, test_batches=sw_batch_size)

    while batch_preds.shape[0] > 0:
      # count predicted SNPs
      num_snps = batch_preds.shape[0] // 2

      # normalize
      batch_preds /= target_norms

      # block for last thread
      if sum_write_thread is not None:
        sum_write_thread.join()

      # summarize and write
      sum_write_thread = threading.Thread(target=summarize_write,
            args=(batch_preds, sad_out, szi, options.sad_stats, options.log_pseudo))
      sum_write_thread.start()

      # update SNP index
      szi += num_snps

      # predict next
      batch_preds = model.predict_tfr(sess, test_batches=sw_batch_size)

  print('Waiting for threads to finish.', flush=True)
  sum_write_thread.join()

  ###################################################
  # compute SAD distributions across variants

  if not options.out_txt:
    # define percentiles
    d_fine = 0.001
    d_coarse = 0.01
    percentiles_neg = np.arange(d_fine, 0.1, d_fine)
    percentiles_base = np.arange(0.1, 0.9, d_coarse)
    percentiles_pos = np.arange(0.9, 1, d_fine)

    percentiles = np.concatenate([percentiles_neg, percentiles_base, percentiles_pos])
    sad_out.create_dataset('percentiles', data=percentiles)
    pct_len = len(percentiles)

    for sad_stat in options.sad_stats:
      sad_stat_pct = '%s_pct' % sad_stat

      # compute
      sad_pct = np.percentile(sad_out[sad_stat], 100*percentiles, axis=0).T
      sad_pct = sad_pct.astype('float16')

      # save
      sad_out.create_dataset(sad_stat_pct, data=sad_pct, dtype='float16')

  if not options.out_zarr:
    sad_out.close()


def summarize_write(batch_preds, sad_out, szi, stats, log_pseudo):
  num_targets = batch_preds.shape[-1]
  pi = 0
  while pi < batch_preds.shape[0]:
    # get reference prediction (LxT)
    ref_preds = batch_preds[pi]
    pi += 1

    # get alternate prediction (LxT)
    alt_preds = batch_preds[pi]
    pi += 1

    # sum across length
    ref_preds_sum = ref_preds.sum(axis=0, dtype='float64')
    alt_preds_sum = alt_preds.sum(axis=0, dtype='float64')

    # compare reference to alternative via mean subtraction
    if 'SAD' in stats:
      sad = alt_preds_sum - ref_preds_sum
      sad_out['SAD'][szi,:] = sad.astype('float16')

    # compare reference to alternative via mean log division
    if 'SAR' in stats:
      sar = np.log2(alt_preds_sum + log_pseudo) \
                     - np.log2(ref_preds_sum + log_pseudo)
      sad_out['SAR'][szi,:] = sar.astype('float16')

    # compare geometric means
    if 'geoSAD' in stats:
      sar_vec = np.log2(alt_preds.astype('float64') + log_pseudo) \
                  - np.log2(ref_preds.astype('float64') + log_pseudo)
      geo_sad = sar_vec.sum(axis=0)
      sad_out['geoSAD'][szi,:] = geo_sad.astype('float16')

    # compute max difference position
    # max_li = np.argmax(np.abs(sar_vec), axis=0)
    # sad_out['xSAR'][szi,:] = np.array([sar_vec[max_li[ti],ti] for ti in range(num_targets)], dtype='float16')

    szi += 1


def initialize_output_h5(out_dir, sad_stats, snps, target_ids, target_labels):
  """Initialize an output HDF5 file for SAD stats."""

  num_targets = len(target_ids)
  num_snps = len(snps)

  sad_out = h5py.File('%s/sad.h5' % out_dir, 'w')

  # write SNPs
  snp_ids = np.array([snp.rsid for snp in snps], 'S')
  sad_out.create_dataset('snp', data=snp_ids)

  # write SNP chr
  snp_chr = np.array([snp.chr for snp in snps], 'S')
  sad_out.create_dataset('chr', data=snp_chr)

  # write SNP pos
  snp_pos = np.array([snp.pos for snp in snps], dtype='uint32')
  sad_out.create_dataset('pos', data=snp_pos)

  # check flips
  snp_flips = [snp.flipped for snp in snps]

  # write SNP reference allele
  snp_refs = []
  for snp in snps:
    if snp.flipped:
      snp_refs.append(snp.alt_alleles[0])
    else:
      snp_refs.append(snp.ref_allele)
  snp_refs = np.array(snp_refs, 'S')
  sad_out.create_dataset('ref', data=snp_refs)

  # write targets
  sad_out.create_dataset('target_ids', data=np.array(target_ids, 'S'))
  sad_out.create_dataset('target_labels', data=np.array(target_labels, 'S'))

  # initialize SAD stats
  for sad_stat in sad_stats:
    sad_out.create_dataset(sad_stat,
        shape=(num_snps, num_targets),
        dtype='float16',
        compression=None)

  return sad_out


def initialize_output_zarr(out_dir, sad_stats, snps, target_ids, target_labels):
  """Initialize an output Zarr file for SAD stats."""

  num_targets = len(target_ids)
  num_snps = len(snps)

  sad_out = zarr.open_group('%s/sad.zarr' % out_dir, 'w')

  # write SNPs
  sad_out.create_dataset('snp', data=[snp.rsid for snp in snps], chunks=(32768,))

  # write targets
  sad_out.create_dataset('target_ids', data=target_ids, compressor=None)
  sad_out.create_dataset('target_labels', data=target_labels, compressor=None)

  # initialize SAD stats
  for sad_stat in sad_stats:
    sad_out.create_dataset(sad_stat,
        shape=(num_snps, num_targets),
        chunks=(128, num_targets),
        dtype='float16')

  return sad_out


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
