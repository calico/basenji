#!/usr/bin/env python
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

from optparse import OptionParser
import gc
import os
import pdb
import sys
import time

import h5py
import numpy as np
import pandas as pd
import pyBigWig
from scipy.stats import ttest_1samp
import tensorflow as tf

from basenji import batcher
from basenji import genedata
from basenji import params
from basenji import seqnn

from basenji_test import bigwig_open

'''
basenji_map.py

Visualize a sequence's prediction's gradients as a map of influence across
the genomic region.
'''

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <model_file> <genes_hdf5_file>'
  parser = OptionParser(usage)
  parser.add_option('-b', dest='bigwig',
      default=False, action='store_true',
      help='Write BigWig tracks [Default: %default]')
  parser.add_option('-c', dest='center',
      default=False, action='store_true',
      help='Compute gradients to the center position, \
            rather than the sum across sequence [Default: %default]')
  parser.add_option('-g', dest='genome_file',
      default='%s/data/human.hg38.genome' % os.environ['BASENJIDIR'],
      help='Chromosome lengths file [Default: %default]')
  parser.add_option('-l', dest='gene_list',
      help='Process only gene ids in the given file')
  parser.add_option('--mc', dest='mc_n',
      default=0, type='int',
      help='Monte carlo test iterations [Default: %default]')
  parser.add_option('-n', dest='norm',
      default=None, type='int',
      help='Compute saliency norm [Default% default]')
  parser.add_option('-o', dest='out_dir',
      default='grad_map',
      help='Output directory [Default: %default]')
  parser.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Average the forward and reverse complement predictions when testing [Default: %default]')
  parser.add_option('--shifts', dest='shifts',
      default='0',
      help='Ensemble prediction shifts [Default: %default]')
  parser.add_option('-t', dest='targets_file',
    default=None, type='str',
    help='File specifying target indexes and labels in table format')
  (options, args) = parser.parse_args()

  if len(args) != 3:
    parser.error('Must provide parameters, model, and genomic position')
  else:
    params_file = args[0]
    model_file = args[1]
    genes_hdf5_file = args[2]

  if not os.path.isdir(options.out_dir):
    os.mkdir(options.out_dir)

  options.shifts = [int(shift) for shift in options.shifts.split(',')]

  #################################################################
  # reads in genes HDF5

  gene_data = genedata.GeneData(genes_hdf5_file)

  # subset gene sequences
  genes_subset = set()
  if options.gene_list:
    for line in open(options.gene_list):
      genes_subset.add(line.rstrip())

    gene_data.subset_genes(genes_subset)
    print('Filtered to %d sequences' % gene_data.num_seqs)

  # extract sequence chrom and start
  seqs_chrom = [gene_data.gene_seqs[si].chrom for si in range(gene_data.num_seqs)]
  seqs_start = [gene_data.gene_seqs[si].start for si in range(gene_data.num_seqs)]


  #######################################################
  # model parameters and placeholders

  job = params.read_job_params(params_file)

  job['seq_length'] = gene_data.seq_length
  job['seq_depth'] = gene_data.seq_depth
  job['target_pool'] = gene_data.pool_width

  if 'num_targets' not in job:
    print(
        "Must specify number of targets (num_targets) in the parameters file.",
        file=sys.stderr)
    exit(1)

  # read targets
  if options.targets_file is not None:
    targets_df = pd.read_table(options.targets_file, index_col=0)
    target_indexes = targets_df.index
    target_subset = target_indexes
  else:
    if gene_data.num_targets is None:
      print('No targets to test against.', file=sys.stderr)
      exit(1)
    else:
      target_indexes = np.arange(gene_data.num_targets)
      target_subset = None

  # build model
  model = seqnn.SeqNN()
  model.build_feed(job, target_subset=target_subset)

  # determine latest pre-dilated layer
  cnn_dilation = np.array([cp.dilation for cp in model.hp.cnn_params])
  dilated_mask = cnn_dilation > 1
  dilated_indexes = np.where(dilated_mask)[0]
  pre_dilated_layer = np.min(dilated_indexes)
  print('Pre-dilated layer: %d' % pre_dilated_layer)

  # build gradients ops
  t0 = time.time()
  print('Building target/position-specific gradient ops.', end='')
  model.build_grads(layers=[pre_dilated_layer], center=options.center)
  print(' Done in %ds' % (time.time()-t0), flush=True)


  #######################################################
  # acquire gradients

  # initialize saver
  saver = tf.train.Saver()

  with tf.Session() as sess:
    # load variables into session
    saver.restore(sess, model_file)

    # score sequences and write bigwigs
    score_write(sess, model, options, target_indexes, gene_data.seqs_1hot, seqs_chrom, seqs_start)


def score_write(sess, model, options, target_indexes, seqs_1hot, seqs_chrom, seqs_start):
  ''' Compute scores and write them as BigWigs for a set of sequences. '''

  num_seqs = seqs_1hot.shape[0]
  num_targets = len(target_indexes)

  # initialize scores HDF5
  scores_h5_file = '%s/scores.h5' % options.out_dir
  scores_h5_out = h5py.File(scores_h5_file, 'w')

  for si in range(num_seqs):
    # initialize batcher
    batcher_si = batcher.Batcher(seqs_1hot[si:si+1],
                                 batch_size=model.hp.batch_size,
                                 pool_width=model.hp.target_pool)

    # get layer representations
    t0 = time.time()
    print('Computing gradients.', end='', flush=True)
    _, _, _, batch_grads, batch_reprs, _ = model.gradients(sess, batcher_si,
      rc=options.rc, shifts=options.shifts, mc_n=options.mc_n, return_all=True)
    print(' Done in %ds.' % (time.time()-t0), flush=True)

    # only layer
    batch_reprs = batch_reprs[0]
    batch_grads = batch_grads[0]

    # increase resolution
    batch_reprs = batch_reprs.astype('float32')
    batch_grads = batch_grads.astype('float32')

    # S (sequences) x T (targets) x P (seq position) x U (units layer i) x E (ensembles)
    print('batch_grads', batch_grads.shape)

    # S (sequences) x P (seq position) x U (Units layer i) x E (ensembles)
    print('batch_reprs', batch_reprs.shape)

    preds_length = batch_reprs.shape[1]
    if 'score' not in scores_h5_out:
      # initialize scores
      scores_h5_out.create_dataset('score', shape=(num_seqs,preds_length,num_targets), dtype='float16')
      scores_h5_out.create_dataset('pvalue', shape=(num_seqs,preds_length,num_targets), dtype='float16')

    # write bigwigs
    t0 = time.time()
    print('Computing and writing scores.', end='', flush=True)

    # for each target
    for tii in range(len(target_indexes)):
      ti = target_indexes[tii]

      # representation x gradient
      batch_grads_scores = np.multiply(batch_reprs[0], batch_grads[0,tii,:,:,:])

      if options.norm is None:
        # sum across filters
        batch_grads_scores = batch_grads_scores.sum(axis=1)
      else:
        # raise to power
        batch_grads_scores = np.power(np.abs(batch_grads_scores), options.norm)
        # sum across filters
        batch_grads_scores = batch_grads_scores.sum(axis=1)
        # normalize w/ 1/power
        batch_grads_scores = np.power(batch_grads_scores, 1./options.norm)

      # mean across ensemble
      batch_grads_mean = batch_grads_scores.mean(axis=1)

      # compute p-values
      if options.norm is None:
        batch_grads_pval = ttest_1samp(batch_grads_scores, 0, axis=1)[1]
      else:
        batch_grads_pval = ttest_1samp(batch_grads_scores, 0, axis=1)[1]
        # batch_grads_pval = chi2(df=)
        batch_grads_pval /= 2

      # write to HDF5
      scores_h5_out['score'][si,:,tii] = batch_grads_mean.astype('float16')
      scores_h5_out['pvalue'][si,:,tii] = batch_grads_pval.astype('float16')

      if options.bigwig:
        # open bigwig
        bws_file = '%s/s%d_t%d_scores.bw' % (options.out_dir, si, ti)
        bwp_file = '%s/s%d_t%d_pvals.bw' % (options.out_dir, si, ti)
        bws_open = bigwig_open(bws_file, options.genome_file)
        # bwp_open = bigwig_open(bwp_file, options.genome_file)

        # specify bigwig locations and values
        bw_chroms = [seqs_chrom[si]]*preds_length
        bw_starts = [int(seqs_start[si] + pi*model.hp.target_pool) for pi in range(preds_length)]
        bw_ends = [int(bws + model.hp.target_pool) for bws in bw_starts]
        bws_values = [float(bgs) for bgs in batch_grads_mean]
        # bwp_values = [float(bgp) for bgp in batch_grads_pval]

        # write
        bws_open.addEntries(bw_chroms, bw_starts, ends=bw_ends, values=bws_values)
        # bwp_open.addEntries(bw_chroms, bw_starts, ends=bw_ends, values=bwp_values)

      # close
      if options.bigwig:
        bws_open.close()
        # bwp_open.close()

    print(' Done in %ds.' % (time.time()-t0), flush=True)
    gc.collect()

  scores_h5_out.close()


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
