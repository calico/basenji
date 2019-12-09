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
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from basenji import batcher
from basenji import params
from basenji import seqnn
from basenji import vcf
from basenji_sat_h5 import delta_matrix, satmut_seqs, subplot_params
from basenji_sat_h5 import plot_heat, plot_sad, plot_seqlogo

'''
basenji_sat_vcf.py

Perform an in silico saturated mutagenesis of the sequences surrounding variants
given in a VCF file.

TODO:
-Convert to tf.data following basenji_sat_bed.py. Unlike there, we can just make
 the plots here.
-Choose some plots, by computing cosine nearest neighbor graph between flattened
 target profiles and cluster. Choose cluster representatives with largest variant
 scores.
'''

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <model_file> <vcf_file>'
  parser = OptionParser(usage)
  parser.add_option('-f', dest='figure_width',
      default=20, type='float',
      help='Figure width [Default: %default]')
  parser.add_option('--f1', dest='genome1_fasta',
      default='%s/data/hg19.fa' % os.environ['BASENJIDIR'],
      help='Genome FASTA which which major allele sequences will be drawn')
  parser.add_option('--f2', dest='genome2_fasta',
      default=None,
      help='Genome FASTA which which minor allele sequences will be drawn')
  parser.add_option('-g', dest='gain',
      default=False, action='store_true',
      help='Draw a sequence logo for the gain score, too [Default: %default]')
  # parser.add_option('-k', dest='plot_k',
  #     default=None, type='int',
  #     help='Plot the top k targets at each end.')
  parser.add_option('-l', dest='satmut_len',
      default=200, type='int',
      help='Length of centered sequence to mutate [Default: %default]')
  parser.add_option('--mean', dest='mean_targets',
      default=False, action='store_true',
      help='Take the mean across targets for a single plot [Default: %default]')
  parser.add_option('-m', dest='mc_n',
      default=0, type='int',
      help='Monte carlo iterations [Default: %default]')
  parser.add_option('--min', dest='min_limit',
      default=0.01, type='float',
      help='Minimum heatmap limit [Default: %default]')
  parser.add_option('-n', dest='load_sat_npy',
      default=False, action='store_true',
      help='Load the predictions from .npy files [Default: %default]')
  parser.add_option('-o', dest='out_dir',
      default='sat_vcf',
      help='Output directory [Default: %default]')
  parser.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Ensemble forward and reverse complement predictions [Default: %default]')
  parser.add_option('--shifts', dest='shifts',
      default='0',
      help='Ensemble prediction shifts [Default: %default]')
  parser.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')
  (options, args) = parser.parse_args()

  if len(args) != 3:
    parser.error('Must provide parameters and model files and VCF')
  else:
    params_file = args[0]
    model_file = args[1]
    vcf_file = args[2]

  if not os.path.isdir(options.out_dir):
    os.mkdir(options.out_dir)

  options.shifts = [int(shift) for shift in options.shifts.split(',')]

  #################################################################
  # prep SNP sequences
  #################################################################

  # read parameters
  job = params.read_job_params(params_file, require=['seq_length', 'num_targets'])

  # load SNPs
  snps = vcf.vcf_snps(vcf_file)

  # get one hot coded input sequences
  if not options.genome2_fasta:
    seqs_1hot, seq_headers, snps, seqs = vcf.snps_seq1(
        snps, job['seq_length'], options.genome1_fasta, return_seqs=True)
  else:
    seqs_1hot, seq_headers, snps, seqs = vcf.snps2_seq1(
        snps, job['seq_length'], options.genome1_fasta,
        options.genome2_fasta, return_seqs=True)

  seqs_n = seqs_1hot.shape[0]

  #################################################################
  # setup model
  #################################################################

  if options.targets_file is None:
    target_ids = ['t%d' % ti for ti in range(job['num_targets'])]
    target_labels = ['']*len(target_ids)
    target_subset = None
    num_targets = job['num_targets']

  else:
    targets_df = pd.read_csv(options.targets_file, sep='\t', index_col=0)
    target_ids = targets_df.identifier
    target_labels = targets_df.description
    target_subset = targets_df.index
    if len(target_subset) == job['num_targets']:
        target_subset = None
    num_targets = len(target_subset)

  if not options.load_sat_npy:
    # build model
    model = seqnn.SeqNN()
    model.build_feed_sad(job, ensemble_rc=options.rc,
        ensemble_shifts=options.shifts, target_subset=target_subset)

    # initialize saver
    saver = tf.train.Saver()

  #################################################################
  # predict and process
  #################################################################

  with tf.Session() as sess:
    if not options.load_sat_npy:
      # load variables into session
      saver.restore(sess, model_file)

    for si in range(seqs_n):
      header = seq_headers[si]
      header_fs = fs_clean(header)

      print('Mutating sequence %d / %d' % (si + 1, seqs_n), flush=True)

      # write sequence
      fasta_out = open('%s/seq%d.fa' % (options.out_dir, si), 'w')
      end_len = (len(seqs[si]) - options.satmut_len) // 2
      print('>seq%d\n%s' % (si, seqs[si][end_len:-end_len]), file=fasta_out)
      fasta_out.close()

      #################################################################
      # predict modifications

      if options.load_sat_npy:
        sat_preds = np.load('%s/seq%d_preds.npy' % (options.out_dir, si))

      else:
        # supplement with saturated mutagenesis
        sat_seqs_1hot = satmut_seqs(seqs_1hot[si:si + 1], options.satmut_len)

        # initialize batcher
        batcher_sat = batcher.Batcher(
            sat_seqs_1hot, batch_size=model.hp.batch_size)

        # predict
        sat_preds = model.predict_h5(sess, batcher_sat)
        np.save('%s/seq%d_preds.npy' % (options.out_dir, si), sat_preds)

      if options.mean_targets:
        sat_preds = np.mean(sat_preds, axis=-1, keepdims=True)
        num_targets = 1

      #################################################################
      # compute delta, loss, and gain matrices

      # compute the matrix of prediction deltas: (4 x L_sm x T) array
      sat_delta = delta_matrix(seqs_1hot[si], sat_preds, options.satmut_len)

      # sat_loss, sat_gain = loss_gain(sat_delta, sat_preds[si], options.satmut_len)
      sat_loss = sat_delta.min(axis=0)
      sat_gain = sat_delta.max(axis=0)

      ##############################################
      # plot

      for ti in range(num_targets):
        # setup plot
        sns.set(style='white', font_scale=1)
        spp = subplot_params(sat_delta.shape[1])

        if options.gain:
          plt.figure(figsize=(options.figure_width, 4))
          ax_logo_loss = plt.subplot2grid(
              (4, spp['heat_cols']), (0, spp['logo_start']),
              colspan=spp['logo_span'])
          ax_logo_gain = plt.subplot2grid(
              (4, spp['heat_cols']), (1, spp['logo_start']),
              colspan=spp['logo_span'])
          ax_sad = plt.subplot2grid(
              (4, spp['heat_cols']), (2, spp['sad_start']),
              colspan=spp['sad_span'])
          ax_heat = plt.subplot2grid(
              (4, spp['heat_cols']), (3, 0), colspan=spp['heat_cols'])
        else:
          plt.figure(figsize=(options.figure_width, 3))
          ax_logo_loss = plt.subplot2grid(
              (3, spp['heat_cols']), (0, spp['logo_start']),
              colspan=spp['logo_span'])
          ax_sad = plt.subplot2grid(
              (3, spp['heat_cols']), (1, spp['sad_start']),
              colspan=spp['sad_span'])
          ax_heat = plt.subplot2grid(
              (3, spp['heat_cols']), (2, 0), colspan=spp['heat_cols'])

        # plot sequence logo
        plot_seqlogo(ax_logo_loss, seqs_1hot[si], -sat_loss[:, ti])
        if options.gain:
          plot_seqlogo(ax_logo_gain, seqs_1hot[si], sat_gain[:, ti])

        # plot SAD
        plot_sad(ax_sad, sat_loss[:, ti], sat_gain[:, ti])

        # plot heat map
        plot_heat(ax_heat, sat_delta[:, :, ti], options.min_limit)

        plt.tight_layout()
        plt.savefig('%s/%s_t%d.pdf' % (options.out_dir, header_fs, target_subset[ti]), dpi=600)
        plt.close()


def fs_clean(header):
  """ Clean up the headers to valid filenames. """
  header = header.replace(':', '_')
  header = header.replace('>', '_')
  return header


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
  # pdb.runcall(main)
