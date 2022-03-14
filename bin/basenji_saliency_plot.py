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
import pdb
import random
import sys

import h5py
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns

from basenji import plots

'''
basenji_sat_plot.py

Generate plots from scores HDF5 file output by saturation mutagenesis analysis
via basenji_sat_bed.py
'''

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <scores_file>'
  parser = OptionParser(usage)
  parser.add_option('-a', dest='activity_enrich',
      default=1, type='float',
      help='Enrich for the most active top % of sequences [Default: %default]')
  parser.add_option('-f', dest='figure_width',
      default=20, type='float',
      help='Figure width [Default: %default]')
  parser.add_option('-l', dest='plot_len',
      default=300, type='int',
      help='Length of centered sequence to mutate [Default: %default]')
  parser.add_option('-o', dest='out_dir',
      default='sat_plot', help='Output directory [Default: %default]')
  parser.add_option('--png', dest='save_png',
      default=False, action='store_true',
      help='Write PNG as opposed to PDF [Default: %default]')
  parser.add_option('-r', dest='rng_seed',
      default=1, type='float',
      help='Random number generator seed [Default: %default]')
  parser.add_option('-s', dest='sample',
      default=None, type='int',
      help='Sample N sequences from the set [Default:%default]')
  parser.add_option('--stat', dest='sad_stat',
      default='sum',
      help='SAD stat to display [Default: %default]')
  # Plot for all targets
  parser.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')
  (options, args) = parser.parse_args()

  if len(args) != 1:
    parser.error('Must provide scores HDF5 file')
  else:
    scores_h5_file = args[0]

  if not os.path.isdir(options.out_dir):
    os.mkdir(options.out_dir)

  save_ext = 'pdf'
  if options.save_png:
    save_ext = 'png'

  np.random.seed(options.rng_seed)

  # open scores
  scores_h5 = h5py.File(scores_h5_file)

  # check for stat
  if options.sad_stat not in scores_h5:
    print('%s does not have key %s' % (scores_h5_file, options.sad_stat), file=sys.stderr)
    exit(1)

  # extract shapes
  num_seqs = scores_h5['seqs'].shape[0]
  mut_len = scores_h5[options.sad_stat].shape[1]

  if options.plot_len > mut_len:
    print('Decreasing plot_len=%d to maximum %d' % (options.plot_len, mut_len), file=sys.stderr)
    options.plot_len = mut_len

  # determine targets
  if options.targets_file is not None:
    targets_df = pd.read_table(options.targets_file, index_col=0)
    num_targets = targets_df.shape[0]
  else:
    num_targets = scores_h5[options.sad_stat].shape[-1]

  # determine plot region
  mut_mid = mut_len // 2
  plot_start = mut_mid - (options.plot_len//2)
  plot_end = plot_start + options.plot_len

  # plot attributes
  sns.set(style='white', font_scale=1)
  spp = subplot_params(options.plot_len)

  # determine sequences
  seq_indexes = np.arange(num_seqs)

  if options.sample and options.sample < num_seqs:
    seq_indexes = np.random.choice(seq_indexes, size=options.sample, replace=False)

  for si in seq_indexes:
    # read sequence
    seq_1hot = scores_h5['seqs'][si,plot_start:plot_end]

    # read scores
    scores = scores_h5[options.sad_stat][si,plot_start:plot_end,:,:]

    # reference scores
    ref_scores = scores[seq_1hot]

    for tii in range(num_targets):
      if options.targets_file is not None:
        ti = targets_df.index[tii]
      else:
        ti = tii

      scores_ti = scores[:,:,ti]

      # For ISM scores: compute scores relative to reference
      # delta_ti = scores_ti - ref_scores[:,[ti]]

      # compute scores (gradient at reference nucleotides)
      grad_x_inp = scores_ti[np.nonzero(scores_ti)]

      # setup plot
      plt.figure(figsize=(options.figure_width, 6))
      grid_rows = 1
      row_i = 0
      ax_logo_grads = plt.subplot2grid(
          (grid_rows, spp['heat_cols']), (row_i, spp['logo_start']),
          colspan=spp['logo_span'])
      row_i += 1

      # plot sequence logo
      plot_seqlogo(ax_logo_grads, seq_1hot, grad_x_inp)
      plt.tight_layout()
      plt.savefig('%s/seq%d_t%d.%s' % (options.out_dir, si, ti, save_ext), dpi=600)
      plt.close()


def enrich_activity(seqs, seqs_1hot, targets, activity_enrich, target_indexes):
  """ Filter data for the most active sequences in the set. """

  # compute the max across sequence lengths and mean across targets
  seq_scores = targets[:, :, target_indexes].max(axis=1).mean(
      axis=1, dtype='float64')

  # sort the sequences
  scores_indexes = [(seq_scores[si], si) for si in range(seq_scores.shape[0])]
  scores_indexes.sort(reverse=True)

  # filter for the top
  enrich_indexes = sorted(
      [scores_indexes[si][1] for si in range(seq_scores.shape[0])])
  enrich_indexes = enrich_indexes[:int(activity_enrich * len(enrich_indexes))]
  seqs = [seqs[ei] for ei in enrich_indexes]
  seqs_1hot = seqs_1hot[enrich_indexes]
  targets = targets[enrich_indexes]

  return seqs, seqs_1hot, targets


def expand_4l(sat_lg_ti, seq_1hot):
  # determine satmut length
  satmut_len = sat_lg_ti.shape[0]

  # jump to satmut region in one hot coded sequence
  ssi = int((seq_1hot.shape[0] - satmut_len) // 2)

  # filter sequence for satmut region
  seq_1hot_sm = seq_1hot[ssi:ssi + satmut_len, :]

  # tile loss scores to align
  sat_lg_tile = np.tile(sat_lg_ti, (4, 1)).T

  # element-wise multiple
  sat_lg_4l = np.multiply(seq_1hot_sm, sat_lg_tile)

  return sat_lg_4l


def plot_seqlogo(ax, seq_1hot, sat_score_ti, pseudo_pct=0.05):
  """ Plot a sequence logo for the loss/gain scores.

    Args:
        ax (Axis): matplotlib axis to plot to.
        seq_1hot (Lx4 array): One-hot coding of a sequence.
        sat_score_ti (L_sm array): Minimum mutation delta across satmut length.
        pseudo_pct (float): % of the max to add as a pseudocount.
    """
  sat_score_cp = sat_score_ti.copy()
  satmut_len = len(sat_score_ti)

  # add pseudocounts
  sat_score_cp += pseudo_pct * sat_score_cp.max()

  # expand
  sat_score_4l = expand_4l(sat_score_cp, seq_1hot)

  plots.seqlogo(sat_score_4l, ax)


def subplot_params(seq_len):
  """ Specify subplot layout parameters for various sequence lengths. """
  if seq_len < 500:
    spp = {
        'heat_cols': 400,
        'sad_start': 1,
        'sad_span': 321,
        'logo_start': 0,
        'logo_span': 323
    }
  else:
    spp = {
        'heat_cols': 400,
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
