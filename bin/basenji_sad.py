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
import time

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pysam
import tensorflow as tf

import basenji.dna_io
import basenji.vcf

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
  parser.add_option(
      '-b',
      dest='batch_size',
      default=256,
      type='int',
      help='Batch size [Default: %default]')
  parser.add_option(
      '-c',
      dest='csv',
      default=False,
      action='store_true',
      help='Print table as CSV [Default: %default]')
  parser.add_option(
      '-e',
      dest='heatmaps',
      default=False,
      action='store_true',
      help='Draw score heatmaps, grouped by index SNP [Default: %default]')
  parser.add_option(
      '-f',
      dest='genome_fasta',
      default='%s/assembly/hg19.fa' % os.environ['HG19'],
      help='Genome FASTA from which sequences will be drawn [Default: %default]'
  )
  parser.add_option(
      '-g',
      dest='genome_file',
      default='%s/assembly/human.hg19.genome' % os.environ['HG19'],
      help='Chromosome lengths file [Default: %default]')
  parser.add_option(
      '-i',
      dest='index_snp',
      default=False,
      action='store_true',
      help=
      'SNPs are labeled with their index SNP as column 6 [Default: %default]')
  parser.add_option(
      '-l',
      dest='seq_len',
      type='int',
      default=1024,
      help='Sequence length provided to the model [Default: %default]')
  parser.add_option(
      '-m',
      dest='min_limit',
      default=0.1,
      type='float',
      help='Minimum heatmap limit [Default: %default]')
  parser.add_option(
      '-o',
      dest='out_dir',
      default='sad',
      help='Output directory for tables and plots [Default: %default]')
  parser.add_option(
      '-p',
      dest='processes',
      default=None,
      type='int',
      help='Number of processes, passed by multi script')
  parser.add_option(
      '--rc',
      dest='rc',
      default=False,
      action='store_true',
      help=
      'Average the forward and reverse complement predictions when testing [Default: %default]'
  )
  parser.add_option(
      '-s',
      dest='score',
      default=False,
      action='store_true',
      help='SNPs are labeled with scores as column 7 [Default: %default]')
  parser.add_option(
      '--shifts',
      dest='shifts',
      default='0',
      help='Ensemble prediction shifts [Default: %default]')
  parser.add_option(
      '-t',
      dest='targets_file',
      default=None,
      help='File specifying target indexes and labels in table format')
  parser.add_option(
      '--ti',
      dest='track_indexes',
      help='Comma-separated list of target indexes to output BigWig tracks')
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

  #################################################################
  # setup model

  job = basenji.dna_io.read_job_params(params_file)
  job['batch_length'] = options.seq_len

  if 'num_targets' not in job:
    print(
        "Must specify number of targets (num_targets) in the parameters file. I know, it's annoying. Sorry.",
        file=sys.stderr)
    exit(1)

  if 'target_pool' not in job:
    print(
        "Must specify target pooling (target_pool) in the parameters file. I know it's annoying. Sorry",
        file=sys.stderr)
    exit(1)

  t0 = time.time()
  model = basenji.seqnn.SeqNN()
  model.build(job)
  print('Model building time %f' % (time.time() - t0), flush=True)

  # initialize saver
  saver = tf.train.Saver()

  #################################################################
  # load SNPs

  snps = basenji.vcf.vcf_snps(vcf_file, options.index_snp, options.score)

  # filter for worker SNPs
  if options.processes is not None:
    snps = [
        snps[si] for si in range(len(snps))
        if si % options.processes == worker_index
    ]

  #################################################################
  # setup output

  if options.targets_file is None:
    target_ids = ['t%d' % ti for ti in range(job['num_targets'])]
    target_labels = [''] * len(target_ids)
  else:
    target_ids = []
    target_labels = []
    for line in open(options.targets_file):
      a = line.strip().split('\t')
      target_ids.append(a[0])
      target_labels.append(a[2])

  header_cols = ('rsid', 'index', 'score', 'ref', 'alt', 'ref_upred',
                 'alt_upred', 'usad', 'usar', 'ref_xpred', 'alt_xpred', 'xsad',
                 'xsar', 'target_index', 'target_id', 'target_label')
  if options.csv:
    sad_out = open('%s/sad_table.csv' % options.out_dir, 'w')
    print(','.join(header_cols), file=sad_out)
  else:
    sad_out = open('%s/sad_table.txt' % options.out_dir, 'w')
    print(' '.join(header_cols), file=sad_out)

  # hash by index snp
  sad_matrices = {}
  sad_labels = {}
  sad_scores = {}

  #################################################################
  # process

  # open genome FASTA
  genome_open = pysam.Fastafile(options.genome_fasta)

  snp_i = 0

  with tf.Session() as sess:
    # load variables into session
    saver.restore(sess, model_file)

    # construct first batch
    batch_1hot, batch_snps, snp_i = snps_next_batch(
        snps, snp_i, options.batch_size, options.seq_len, genome_open)

    while len(batch_snps) > 0:
      ###################################################
      # predict

      # initialize batcher
      batcher = basenji.batcher.Batcher(batch_1hot, batch_size=model.batch_size)

      # predict
      batch_preds = model.predict(
          sess, batcher, rc=options.rc, shifts=options.shifts)

      ###################################################
      # collect and print SADs

      pi = 0
      for snp in batch_snps:
        # get reference prediction (LxT)
        ref_preds = batch_preds[pi]
        pi += 1

        # mean across length
        ref_preds_lmean = ref_preds.mean(axis=0)

        # print tracks
        for ti in options.track_indexes:
          ref_bw_file = '%s/tracks/%s_t%d_ref.bw' % (options.out_dir, snp.rsid,
                                                     ti)
          bigwig_write(snp, options.seq_len, ref_preds[:, ti], model,
                       ref_bw_file, options.genome_file)

        for alt_al in snp.alt_alleles:
          # get alternate prediction (LxT)
          alt_preds = batch_preds[pi]
          pi += 1

          # mean across length
          alt_preds_lmean = alt_preds.mean(axis=0)

          # compare reference to alternative via mean subtraction
          sad = alt_preds_lmean - ref_preds_lmean
          sad_matrices.setdefault(snp.index_snp, []).append(sad)

          # compare reference to alternative via mean log division
          sar = np.log2(alt_preds_lmean + 1) - np.log2(ref_preds_lmean + 1)

          # label as mutation from reference
          alt_label = '%s_%s>%s' % (snp.rsid,
                                    basenji.vcf.cap_allele(snp.ref_allele),
                                    basenji.vcf.cap_allele(alt_al))
          sad_labels.setdefault(snp.index_snp, []).append(alt_label)

          # save scores
          sad_scores.setdefault(snp.index_snp, []).append(snp.score)

          # print table lines
          for ti in range(len(sad)):
            # set index SNP
            snp_is = '%-13s' % '.'
            if options.index_snp:
              snp_is = '%-13s' % snp.index_snp

            # set score
            snp_score = '%5s' % '.'
            if options.score:
              snp_score = '%5.3f' % snp.score

            # profile the max difference position
            max_li = 0
            max_sad = alt_preds[max_li, ti] - ref_preds[max_li, ti]
            max_sar = np.log2(alt_preds[max_li, ti] + 1) - np.log2(
                ref_preds[max_li, ti] + 1)
            for li in range(ref_preds.shape[0]):
              sad_li = alt_preds[li, ti] - ref_preds[li, ti]
              sar_li = np.log2(alt_preds[li, ti] + 1) - np.log2(
                  ref_preds[li, ti] + 1)
              # if abs(sad_li) > abs(max_sad):
              if abs(sar_li) > abs(max_sar):
                max_li = li
                max_sad = sad_li
                max_sar = sar_li

            # print line
            cols = (snp.rsid, snp_is, snp_score,
                    basenji.vcf.cap_allele(snp.ref_allele),
                    basenji.vcf.cap_allele(alt_al), ref_preds_lmean[ti],
                    alt_preds_lmean[ti], sad[ti], sar[ti],
                    ref_preds[max_li, ti], alt_preds[max_li, ti], max_sad,
                    max_sar, ti, target_ids[ti], target_labels[ti])
            if options.csv:
              print(','.join([str(c) for c in cols]), file=sad_out)
            else:
              print(
                  '%-13s %s %5s %6s %6s %6.3f %6.3f %7.4f %7.4f %6.3f %6.3f %7.4f %7.4f %4d %12s %s'
                  % cols,
                  file=sad_out)

          # print tracks
          for ti in options.track_indexes:
            alt_bw_file = '%s/tracks/%s_t%d_alt.bw' % (options.out_dir,
                                                       snp.rsid, ti)
            bigwig_write(snp, options.seq_len, alt_preds[:, ti], model,
                         alt_bw_file, options.genome_file)

      ###################################################
      # construct next batch

      batch_1hot, batch_snps, snp_i = snps_next_batch(
          snps, snp_i, options.batch_size, options.seq_len, genome_open)

  sad_out.close()

  #################################################################
  # plot SAD heatmaps
  #################################################################
  if options.heatmaps:
    for ii in sad_matrices:
      # convert fully to numpy arrays
      sad_matrix = abs(np.array(sad_matrices[ii]))
      print(ii, sad_matrix.shape)

      if sad_matrix.shape[0] > 1:
        vlim = max(options.min_limit, sad_matrix.max())
        score_mat = np.reshape(np.array(sad_scores[ii]), (-1, 1))

        # plot heatmap
        plt.figure(figsize=(20, 0.5 + 0.5 * sad_matrix.shape[0]))

        if options.score:
          # lay out scores
          cols = 12
          ax_score = plt.subplot2grid((1, cols), (0, 0))
          ax_sad = plt.subplot2grid((1, cols), (0, 1), colspan=(cols - 1))

          sns.heatmap(
              score_mat,
              xticklabels=False,
              yticklabels=False,
              vmin=0,
              vmax=1,
              cmap='Reds',
              cbar=False,
              ax=ax_score)
        else:
          ax_sad = plt.gca()

        sns.heatmap(
            sad_matrix,
            xticklabels=target_labels,
            yticklabels=sad_labels[ii],
            vmin=0,
            vmax=vlim,
            ax=ax_sad)

        for tick in ax_sad.get_xticklabels():
          tick.set_rotation(-45)
          tick.set_horizontalalignment('left')
          tick.set_fontsize(5)

        plt.tight_layout()
        if ii == '.':
          out_pdf = '%s/sad_heat.pdf' % options.out_dir
        else:
          out_pdf = '%s/sad_%s_heat.pdf' % (options.out_dir, ii)
        plt.savefig(out_pdf)
        plt.close()


def bigwig_write(snp, seq_len, preds, model, bw_file, genome_file):
  bw_open = bigwig_open(bw_file, genome_file)

  seq_chrom = snp.chrom
  seq_start = snp.pos - seq_len // 2

  bw_chroms = [seq_chrom] * len(preds)
  bw_starts = [
      int(seq_start + model.batch_buffer + bi * model.target_pool)
      for bi in range(len(preds))
  ]
  bw_ends = [int(bws + model.target_pool) for bws in bw_starts]

  preds_list = [float(p) for p in preds]
  bw_open.addEntries(bw_chroms, bw_starts, ends=bw_ends, values=preds_list)

  bw_open.close()



def snps_next_batch(snps, snp_i, batch_size, seq_len, genome_open):
  """ Load the next batch of SNP sequence 1-hot. """

  batch_1hot = []
  batch_snps = []

  while len(batch_1hot) < batch_size and snp_i < len(snps):
    # get SNP sequences
    snp_1hot = basenji.vcf.snp_seq1(snps[snp_i], seq_len, genome_open)

    # if it was valid
    if len(snp_1hot) > 0:
      # accumulate
      batch_1hot += snp_1hot
      batch_snps.append(snps[snp_i])

    # advance SNP index
    snp_i += 1

  # convert to array
  batch_1hot = np.array(batch_1hot)

  return batch_1hot, batch_snps, snp_i

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
