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
import json
import os
import pdb
import pickle
import sys
import threading
import time

import h5py
import numpy as np
import pandas as pd
import pysam

import tensorflow as tf
if tf.__version__[0] == '1':
  tf.compat.v1.enable_eager_execution()

from basenji import seqnn
from basenji import vcf as bvcf

'''
akita_scd.py

Compute SNP Contact Difference (SCD) scores for SNPs in a VCF file.
'''

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <model_file> <vcf_file>'
  parser = OptionParser(usage)
  parser.add_option('-f', dest='genome_fasta',
      default='%s/data/hg19.fa' % os.environ['BASENJIDIR'],
      help='Genome FASTA for sequences [Default: %default]')
  parser.add_option('-o',dest='out_dir',
      default='scd',
      help='Output directory for tables and plots [Default: %default]')
  parser.add_option('-p', dest='processes',
      default=None, type='int',
      help='Number of processes, passed by multi script')
  parser.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Average forward and reverse complement predictions [Default: %default]')
  parser.add_option('--shifts', dest='shifts',
      default='0', type='str',
      help='Ensemble prediction shifts [Default: %default]')
  parser.add_option('--stats', dest='scd_stats',
      default='SCD',
      help='Comma-separated list of stats to save. [Default: %default]')
  parser.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')
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

  options.shifts = [int(shift) for shift in options.shifts.split(',')]
  options.scd_stats = options.scd_stats.split(',')


  #################################################################
  # read parameters

  # read model parameters
  with open(params_file) as params_open:
    params = json.load(params_open)
  params_train = params['train']
  params_model = params['model']

  if options.targets_file is None:
    target_ids = ['t%d' % ti for ti in range(params_model['num_targets'])]
    target_labels = ['']*len(target_ids)

  else:
    targets_df = pd.read_csv(options.targets_file, sep='\t', index_col=0)
    target_ids = targets_df.identifier
    target_labels = targets_df.description

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
      snp_1hot_list = bvcf.snp_seq1(snp, params_model['seq_length'], genome_open)

      for snp_1hot in snp_1hot_list:
        yield {'sequence':snp_1hot}

  snp_types = {'sequence': tf.float32}
  snp_shapes = {'sequence': tf.TensorShape([tf.Dimension(params_model['seq_length']),
                                            tf.Dimension(4)])}

  dataset = tf.data.Dataset.from_generator(snp_gen,
                                           output_types=snp_types,
                                           output_shapes=snp_shapes)
  dataset = dataset.batch(params_train['batch_size'])
  dataset = dataset.prefetch(2*params_train['batch_size'])
  dataset_iter = iter(dataset)

  def get_chunk(chunk_size=32):
    """Get a chunk of data from the dataset iterator."""
    x = []
    for ci in range(chunk_size):
      try:
        x.append(next(dataset_iter))
      except StopIteration:
        break

  #################################################################
  # setup model

  # load model
  seqnn_model = seqnn.SeqNN(params_model)
  seqnn_model.restore(model_file)
  seqnn_model.build_ensemble(options.rc, options.shifts)
  

  #################################################################
  # setup output

  scd_out = initialize_output_h5(options.out_dir, options.scd_stats,
                                 snps, target_ids, target_labels)

  #################################################################
  # process

  szi = 0
  sum_write_thread = None

  # predict first
  # batch_seqs = get_chunk()
  # batch_preds = seqnn_model.predict(batch_seqs, steps=batch_seqs)
  batch_preds = seqnn_model.predict(dataset_iter, steps=32)

  while len(batch_preds) > 0:
    # count predicted SNPs
    num_snps = batch_preds.shape[0] // 2

    # block for last thread
    if sum_write_thread is not None:
      sum_write_thread.join()

    # summarize and write
    sum_write_thread = threading.Thread(target=summarize_write,
          args=(batch_preds, scd_out, szi, options.scd_stats))
    sum_write_thread.start()

    # update SNP index
    szi += num_snps

    # predict next
    # batch_preds = seqnn_model.predict(get_chunk())
    batch_preds = seqnn_model.predict(dataset_iter, steps=32)

  print('Waiting for threads to finish.', flush=True)
  sum_write_thread.join()
  
  scd_out.close()


def summarize_write(batch_preds, scd_out, szi, stats):
  num_targets = batch_preds.shape[-1]

  pi = 0
  while pi < batch_preds.shape[0]:
  	# get reference prediction (LxLxT)
    ref_preds = batch_preds[pi].astype('float32')
    pi += 1

    # get alternate prediction (LxLxT)
    alt_preds = batch_preds[pi].astype('float32')
    pi += 1

    if 'SCD' in stats:
      # sum of squared diffs
      diff2_preds = (ref_preds - alt_preds)**2
      sd2_preds = diff2_preds.sum(axis=0)
      scd_out['SCD'][szi,:] = sd2_preds.astype('float16')

    if 'SSD' in stats:
      # sum of squared diffs
      ref_ss = (ref_preds**2).sum(axis=0)
      alt_ss = (alt_preds**2).sum(axis=0)
      s2d_preds = alt_ss - ref_ss
      scd_out['SSD'][szi,:] = s2d_preds.astype('float16')

    szi += 1


def initialize_output_h5(out_dir, scd_stats, snps, target_ids, target_labels):
  """Initialize an output HDF5 file for SCD stats."""

  num_targets = len(target_ids)
  num_snps = len(snps)

  scd_out = h5py.File('%s/scd.h5' % out_dir, 'w')

  # write SNPs
  snp_ids = np.array([snp.rsid for snp in snps], 'S')
  scd_out.create_dataset('snp', data=snp_ids)

  # write SNP chr
  snp_chr = np.array([snp.chr for snp in snps], 'S')
  scd_out.create_dataset('chr', data=snp_chr)

  # write SNP pos
  snp_pos = np.array([snp.pos for snp in snps], dtype='uint32')
  scd_out.create_dataset('pos', data=snp_pos)

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
  scd_out.create_dataset('ref', data=snp_refs)

  # write targets
  scd_out.create_dataset('target_ids', data=np.array(target_ids, 'S'))
  scd_out.create_dataset('target_labels', data=np.array(target_labels, 'S'))

  # initialize scd stats
  for scd_stat in scd_stats:
    scd_out.create_dataset(scd_stat,
        shape=(num_snps, num_targets),
        dtype='float16',
        compression=None)

  return scd_out


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
