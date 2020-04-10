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
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf
if tf.__version__[0] == '1':
  tf.compat.v1.enable_eager_execution()

from basenji import seqnn
from basenji import vcf as bvcf
from basenji_sat_plot import plot_sad, plot_seqlogo

'''
akita_sat_vcf.py

Perform an in silico saturated mutagenesis of the sequences surrounding variants
given in a VCF file.
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
  parser.add_option('-g', dest='gain',
      default=False, action='store_true',
      help='Draw a sequence logo for the gain score, too [Default: %default]')
  parser.add_option('-l', dest='satmut_len',
      default=200, type='int',
      help='Length of centered sequence to mutate [Default: %default]')
  parser.add_option('-m', dest='min_limit',
      default=0.1, type='float',
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
  parser.add_option('-w', dest='figure_width',
      default=20, type='float',
      help='Figure width [Default: %default]')
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
  # read parameters

  # read model parameters
  with open(params_file) as params_open:
    params = json.load(params_open)
  params_train = params['train']
  params_model = params['model']

  """ unused
  if options.targets_file is None:
    target_ids = ['t%d' % ti for ti in range(params_model['num_targets'])]
    target_labels = ['']*len(target_ids)

  else:
    targets_df = pd.read_csv(options.targets_file, sep='\t', index_col=0)
    target_ids = targets_df.identifier
    target_labels = targets_df.description
  """

  #################################################################
  # prep SNP sequences

  # load SNPs
  snps = bvcf.vcf_snps(vcf_file)

  # get one hot coded input sequences
  seqs_1hot, seq_headers, snps, seqs = bvcf.snps_seq1(
        snps, params_model['seq_length'], options.genome_fasta, return_seqs=True)

  seqs_n = seqs_1hot.shape[0]

  #################################################################
  # setup model

  if not options.load_sat_npy:
    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(model_file)
    seqnn_model.build_ensemble(options.rc, options.shifts)


  #################################################################
  # predict and process
   
  for si in range(seqs_n):
    header = seq_headers[si]
    header_fs = fs_clean(header)

    print('Mutating sequence %d / %d' % (si + 1, seqs_n), flush=True)

    #################################################################
    # predict modifications

    if options.load_sat_npy:
      sat_preds = np.load('%s/seq%d_preds.npy' % (options.out_dir, si))

    else:
      # supplement with saturated mutagenesis
      sat_seqs_1hot = satmut_seqs(seqs_1hot[si:si + 1], options.satmut_len)

      # predict
      sat_preds = seqnn_model.predict(sat_seqs_1hot, batch_size=2)
      sat_preds = sat_preds.mean(axis=-1, dtype='float32', keepdims=True)
      np.save('%s/seq%d_preds.npy' % (options.out_dir, si), sat_preds)

    #################################################################
    # score matrices

    # compute the matrix of prediction deltas: (L_sm x 4 x T) array
    sat_scores = score_matrix(seqs_1hot[si], sat_preds)

    # plot max per position
    sat_max = sat_scores.max(axis=1)

    ##############################################
    # plot

    for ti in range(sat_scores.shape[-1]):
      # setup plot
      sns.set(style='white', font_scale=1)
      spp = subplot_params(sat_scores.shape[0])

      plt.figure(figsize=(options.figure_width, 5))
      ax_logo = plt.subplot2grid(
          (3, spp['heat_cols']), (0, spp['logo_start']),
          colspan=spp['logo_span'])
      ax_sad = plt.subplot2grid(
          (3, spp['heat_cols']), (1, spp['sad_start']),
          colspan=spp['sad_span'])
      ax_heat = plt.subplot2grid(
          (3, spp['heat_cols']), (2, 0), colspan=spp['heat_cols'])

      # plot sequence logo
      plot_seqlogo(ax_logo, seqs_1hot[si], sat_max[:,ti])

      # plot SCD
      plot_scd(ax_sad, sat_max[:, ti])

      # plot heat map
      plot_heat(ax_heat, sat_scores[:,:,ti], options.min_limit)
      
      plt.savefig('%s/%s_t%d.pdf' % (options.out_dir, header_fs, ti), dpi=600)
      plt.close()


def plot_heat(ax, sat_score_ti, min_limit=None):
  """ Plot satmut deltas.

    Args:
        ax (Axis): matplotlib axis to plot to.
        sat_delta_ti (L_sm x 4 array): Single target delta matrix for saturated mutagenesis region,
    """

  if np.max(sat_score_ti) < min_limit:
    vmax = min_limit
  else:
    vmax = None

  sns.heatmap(
      sat_score_ti.T,
      linewidths=0,
      xticklabels=False,
      yticklabels=False,
      cmap='Blues',
      vmax=vmax,
      ax=ax)

  # yticklabels break the plot for some reason
  # ax.yaxis.set_ticklabels('ACGT', rotation='horizontal')


def plot_scd(ax, sat_scd_ti):
  """ Plot SCD scores.

    Args:
        ax (Axis): matplotlib axis to plot to.
        sat_scd_ti (L_sm array): Maximum mutation delta across satmut length.
    """
  ax.plot(sat_scd_ti, linewidth=1)
  ax.set_xlim(0, len(sat_scd_ti))
  ax.xaxis.set_ticks([])
  for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(0.5)


def score_matrix(seq_1hot, sat_preds):
  # sat_preds is (1 + 3*mut_len) x (target_len) x (num_targets)
  num_preds = sat_preds.shape[0]
  num_targets = sat_preds.shape[-1]

  # reverse engineer mutagenesis position parameters
  mut_len = (num_preds - 1) // 3
  mut_mid = seq_1hot.shape[0] // 2
  mut_start = mut_mid - mut_len//2
  mut_end = mut_start + mut_len

  # mutagenized DNA
  seq_1hot_mut = seq_1hot[mut_start:mut_end,:]

  # initialize scores
  seq_scores = np.zeros((mut_len, 4, num_targets), dtype='float32')

  # predictions index (starting at first mutagenesis)
  pi = 1

  # for each mutated position
  for mi in range(mut_len):
    # for each nucleotide
    for ni in range(4):
      if seq_1hot_mut[mi,ni]:
        # reference score
        seq_scores[mi,ni,:] = 0
      else:
        # mutation score
        seq_scores[mi,ni,:] = ((sat_preds[pi] - sat_preds[0])**2).sum(axis=0)
        pi += 1

  # transform
  seq_scores = np.log1p(np.sqrt(seq_scores))

  return seq_scores


def fs_clean(header):
  """ Clean up the headers to valid filenames. """
  header = header.replace(':', '_')
  header = header.replace('>', '_')
  return header


def satmut_seqs(seqs_1hot, satmut_len):
  """ Construct a new array with the given sequences and saturated
        mutagenesis versions of them. """

  seqs_n = seqs_1hot.shape[0]
  seq_len = seqs_1hot.shape[1]
  satmut_n = seqs_n + seqs_n * satmut_len * 3

  # initialize satmut seqs 1hot
  sat_seqs_1hot = np.zeros((satmut_n, seq_len, 4), dtype='bool')

  # copy over seqs_1hot
  sat_seqs_1hot[:seqs_n, :, :] = seqs_1hot

  satmut_start = (seq_len - satmut_len) // 2
  satmut_end = satmut_start + satmut_len

  # add saturated mutagenesis
  smi = seqs_n
  for si in range(seqs_n):
    for li in range(satmut_start, satmut_end):
      for ni in range(4):
        if seqs_1hot[si, li, ni] != 1:
          # copy sequence
          sat_seqs_1hot[smi, :, :] = seqs_1hot[si, :, :]

          # mutate to ni
          sat_seqs_1hot[smi, li, :] = np.zeros(4)
          sat_seqs_1hot[smi, li, ni] = 1

          # update index
          smi += 1

  return sat_seqs_1hot


def subplot_params(seq_len):
  """ Specify subplot layout parameters for various sequence lengths. """
  if seq_len < 500:
    spp = {
        'heat_cols': 400,
        'pred_start': 0,
        'pred_span': 322,
        'sad_start': 1,
        'sad_span': 321,
        'logo_start': 0,
        'logo_span': 323
    }
  else:
    spp = {
        'heat_cols': 400,
        'pred_start': 0,
        'pred_span': 321,
        'sad_start': 1,
        'sad_span': 320,
        'logo_start': 0,
        'logo_span': 322
    }

  return spp


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
  # pdb.runcall(main)
