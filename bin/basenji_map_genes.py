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

from optparse import OptionParser
import gc
import os
import pdb
import sys
import time

import h5py
import numpy as np
import pyBigWig
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
  parser.add_option('-g', dest='genome_file',
      default='%s/assembly/human.hg19.genome'%os.environ['HG19'],
      help='Chromosome lengths file [Default: %default]')
  parser.add_option('-l', dest='gene_list',
      help='Process only gene ids in the given file')
  parser.add_option('-o', dest='out_dir',
      default='grad_mapg',
      help='Output directory [Default: %default]')
  parser.add_option('-t', dest='target_indexes',
      default=None,
      help='Target indexes to plot')
  (options,args) = parser.parse_args()

  if len(args) != 3:
  	parser.error('Must provide parameters, model, and genomic position')
  else:
    params_file = args[0]
    model_file = args[1]
    genes_hdf5_file = args[2]

  if not os.path.isdir(options.out_dir):
    os.mkdir(options.out_dir)


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

  #######################################################
  # model parameters and placeholders

  job = params.read_job_params(params_file)

  job['seq_length'] = gene_data.seq_length
  job['seq_depth'] = gene_data.seq_depth
  job['target_pool'] = gene_data.pool_width

  if 'num_targets' not in job:
    print("Must specify number of targets (num_targets) in the parameters file.",
      file=sys.stderr)
    exit(1)

  # set target indexes
  if options.target_indexes is not None:
    options.target_indexes = [int(ti) for ti in options.target_indexes.split(',')]
    target_subset = options.target_indexes
  else:
    options.target_indexes = list(range(job['num_targets']))
    target_subset = None

  # build model
  model = seqnn.SeqNN()
  model.build(job, target_subset=target_subset)

  # determine latest pre-dilated layer
  cnn_dilation = np.array([cp.dilation for cp in model.hp.cnn_params])
  dilated_mask = cnn_dilation > 1
  dilated_indexes = np.where(dilated_mask)[0]
  pre_dilated_layer = np.min(dilated_indexes)
  print('Pre-dilated layer: %d' % pre_dilated_layer)

  # build gradients ops
  t0 = time.time()
  print('Building target/position-specific gradient ops.', end='')
  model.build_grads_genes(gene_data.gene_seqs, layers=[pre_dilated_layer])
  print(' Done in %ds' % (time.time()-t0), flush=True)


  #######################################################
  # acquire gradients

  # initialize saver
  saver = tf.train.Saver()

  with tf.Session() as sess:
    # load variables into session
    saver.restore(sess, model_file)

    for si in range(gene_data.num_seqs):
      # initialize batcher
      batcher_si = batcher.Batcher(gene_data.seqs_1hot[si:si+1],
                                   batch_size=model.hp.batch_size,
                                   pool_width=model.hp.target_pool)

      # get layer representations
      t0 = time.time()
      print('Computing gradients.', end='', flush=True)
      batch_grads, batch_reprs = model.gradients_genes(sess, batcher_si,
                                                       gene_data.gene_seqs[si:si+1])
      print(' Done in %ds.' % (time.time()-t0), flush=True)

      # only layer
      batch_reprs = batch_reprs[0]
      batch_grads = batch_grads[0]

      # G (TSSs) x T (targets) x P (seq position) x U (Units layer i)
      print('batch_grads', batch_grads.shape)
      pooled_length = batch_grads.shape[2]

      # S (sequences) x P (seq position) x U (Units layer i)
      print('batch_reprs', batch_reprs.shape)

      # write bigwigs
      t0 = time.time()
      print('Writing BigWigs.', end='', flush=True)

      # for each TSS
      for tss_i in range(batch_grads.shape[0]):
        tss = gene_data.gene_seqs[si].tss_list[tss_i]

        # for each target
        for tii in range(len(options.target_indexes)):
          ti = options.target_indexes[tii]

          # dot representation and gradient
          batch_grads_score = np.multiply(batch_reprs[0],
                                          batch_grads[tss_i,tii,:,:]).sum(axis=1)

          # open bigwig
          bw_file = '%s/%s-%s_t%d.bw' % \
                      (options.out_dir, tss.gene_id, tss.identifier, ti)
          bw_open = bigwig_open(bw_file, options.genome_file)

          # access gene sequence information
          seq_chrom = gene_data.gene_seqs[si].chrom
          seq_start = gene_data.gene_seqs[si].start

          # specify bigwig locations and values
          bw_chroms = [seq_chrom]*pooled_length
          bw_starts = [int(seq_start + li*model.hp.target_pool) for li in range(pooled_length)]
          bw_ends = [int(bws + model.hp.target_pool) for bws in bw_starts]
          bw_values = [float(bgs) for bgs in batch_grads_score]

          # write
          bw_open.addEntries(bw_chroms, bw_starts, ends=bw_ends, values=bw_values)

          # close
          bw_open.close()

      print(' Done in %ds.' % (time.time()-t0), flush=True)
      gc.collect()


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
