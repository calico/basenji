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
import sys
import time

import h5py
import numpy as np
import pyBigWig
import tensorflow as tf

import basenji

from basenji_test import bigwig_open

'''
basenji_map.py

Visualize a sequence's prediction's gradients as a map of influence across
the genomic region.

Notes:
 -Gradient compute time increases as the program runs. Unclear why. Very annoying.
'''

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <model_file> <genes_hdf5_file>'
  parser = OptionParser(usage)
  parser.add_option(
      '-g',
      dest='genome_file',
      default='%s/assembly/human.hg19.genome' % os.environ['HG19'],
      help='Chromosome lengths file [Default: %default]')
  parser.add_option(
      '-l',
      dest='transcript_list',
      help='Process only transcript ids in the given file')
  parser.add_option(
      '-o',
      dest='out_dir',
      default='grad_map',
      help='Output directory [Default: %default]')
  parser.add_option(
      '-t', dest='target_indexes', default=None, help='Target indexes to plot')
  (options, args) = parser.parse_args()

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

  gene_data = basenji.genes.GeneData(genes_hdf5_file)

  # subset transcripts
  transcripts_subset = set()
  if options.transcript_list:
    for line in open(options.transcript_list):
      transcripts_subset.add(line.rstrip())

    gene_data.subset_transcripts(transcripts_subset)
    print('Filtered to %d sequences' % gene_data.num_seqs)

  #######################################################
  # model parameters and placeholders

  job = basenji.dna_io.read_job_params(params_file)

  job['batch_length'] = gene_data.seq_length
  job['seq_depth'] = gene_data.seq_depth
  job['target_pool'] = gene_data.pool_width
  job['save_reprs'] = True

  if 'num_targets' not in job:
    print(
        "Must specify number of targets (num_targets) in the parameters file. I know, it's annoying. Sorry.",
        file=sys.stderr)
    exit(1)

  # build model
  model = basenji.seqnn.SeqNN()
  model.build(job)

  # determine final pooling layer
  post_pooling_layer = len(model.cnn_pool) - 1

  #######################################################
  # acquire gradients

  # set target indexes
  if options.target_indexes is not None:
    options.target_indexes = [
        int(ti) for ti in options.target_indexes.split(',')
    ]
  else:
    options.target_indexes = list(range(job['num_targets']))

  # initialize saver
  saver = tf.train.Saver()

  with tf.Session() as sess:
    # load variables into session
    saver.restore(sess, model_file)

    si = 0
    while si < gene_data.num_seqs:
      # initialize batcher
      # batcher = basenji.batcher.Batcher(seqs_1hot[si:si+model.batch_size], batch_size=model.batch_size, pool_width=model.target_pool)
      batcher = basenji.batcher.Batcher(
          gene_data.seqs_1hot[si:si + 1],
          batch_size=model.batch_size,
          pool_width=model.target_pool)

      # determine transcript positions
      transcript_positions = set()
      # for bi in range(model.batch_size):   # TEMP
      for bi in range(1):
        if si + bi < len(gene_data.seq_transcripts):
          for transcript, tx_pos in gene_data.seq_transcripts[si + bi]:
            transcript_positions.add(tx_pos)
      transcript_positions = sorted(list(transcript_positions))

      # get layer representations
      t0 = time.time()
      print('Computing gradients.', end='', flush=True)
      batch_grads, batch_reprs = model.gradients_pos(
          sess, batcher, transcript_positions, options.target_indexes,
          post_pooling_layer)
      print(' Done in %ds.' % (time.time() - t0), flush=True)

      # only layer
      batch_reprs = batch_reprs[0]
      batch_grads = batch_grads[0]

      # (B sequences) x (P pooled seq len) x (F filters) x (G gene positions) x (T targets)
      print('batch_grads', batch_grads.shape)
      print('batch_reprs', batch_reprs.shape)

      # (B sequences) x (P pooled seq len) x (G gene positions) x (T targets)
      pooled_length = batch_grads.shape[1]

      # write bigwigs
      t0 = time.time()
      print('Writing BigWigs.', end='', flush=True)
      # for bi in range(model.batch_size):   # TEMP
      for bi in range(1):
        sbi = si + bi
        if sbi < gene_data.num_seqs:
          positions_written = set()
          for transcript, tx_pos in gene_data.seq_transcripts[sbi]:
            # has this transcript position been written?
            if tx_pos not in positions_written:
              # which gene position is this tx_pos?
              gi = 0
              while transcript_positions[gi] != tx_pos:
                gi += 1

              # for each target
              for tii in range(len(options.target_indexes)):
                ti = options.target_indexes[tii]

                # dot representation and gradient
                batch_grads_score = np.multiply(
                    batch_reprs[bi], batch_grads[bi, :, :, gi, tii]).sum(
                        axis=1)

                bw_file = '%s/%s_t%d.bw' % (options.out_dir, transcript, ti)
                bw_open = bigwig_open(bw_file, options.genome_file)

                seq_chrom, seq_start, seq_end = gene_data.seq_coords[sbi]
                bw_chroms = [seq_chrom] * pooled_length
                bw_starts = [
                    int(seq_start + li * model.target_pool)
                    for li in range(pooled_length)
                ]
                bw_ends = [int(bws + model.target_pool) for bws in bw_starts]
                bw_values = [float(bgs) for bgs in batch_grads_score]

                bw_open.addEntries(
                    bw_chroms, bw_starts, ends=bw_ends, values=bw_values)

                bw_open.close()

                positions_written.add(tx_pos)
      print(' Done in %ds.' % (time.time() - t0), flush=True)
      gc.collect()

      # advance through sequences
      # si += model.batch_size
      si += 1


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
