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
matplotlib.use('agg')
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
  parser.add_option('-g', dest='gain',
      default=False, action='store_true',
      help='Draw a sequence logo for the gain score, too [Default: %default]')
  parser.add_option('-l', dest='plot_len',
      default=300, type='int',
      help='Length of centered sequence to mutate [Default: %default]')
  parser.add_option('-m', dest='min_limit',
      default=0.01, type='float',
      help='Minimum heatmap limit [Default: %default]')
  parser.add_option('-o', dest='out_dir',
      default='sat_plot', help='Output directory [Default: %default]')
  parser.add_option('-r', dest='rng_seed',
      default=1, type='float',
      help='Random number generator seed [Default: %default]')
  parser.add_option('-s', dest='sample',
      default=None, type='int',
      help='Sample N sequences from the set [Default:%default]')
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

  np.random.seed(options.rng_seed)

  # determine targets
  targets_df = pd.read_table(options.targets_file, index_col=0)
  num_targets = targets_df.shape[0]

  # open scores
  scores_h5 = h5py.File(scores_h5_file)
  num_seqs = scores_h5['seqs'].shape[0]
  mut_len = scores_h5['scores'].shape[1]

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
    scores = scores_h5['scores'][si,plot_start:plot_end,:,:]

    # reference scores
    ref_scores = scores[seq_1hot]

    for tii in range(num_targets):
      ti = targets_df['index'].iloc[tii]

      scores_ti = scores[:,:,ti]

      # compute scores relative to reference
      delta_ti = scores_ti - ref_scores[:,[ti]]

      # compute loss and gain
      delta_loss = delta_ti.min(axis=1)
      delta_gain = delta_ti.max(axis=1)

      # setup plot
      plt.figure(figsize=(options.figure_width, 4))
      if options.gain:
        grid_rows = 4
      else:
        grid_rows = 3
      row_i = 0
      ax_logo_loss = plt.subplot2grid(
          (grid_rows, spp['heat_cols']), (row_i, spp['logo_start']),
          colspan=spp['logo_span'])
      row_i += 1
      if options.gain:
        ax_logo_gain = plt.subplot2grid(
          (grid_rows, spp['heat_cols']), (row_i, spp['logo_start']),
          colspan=spp['logo_span'])
        row_i += 1
      ax_sad = plt.subplot2grid(
          (grid_rows, spp['heat_cols']), (row_i, spp['sad_start']),
          colspan=spp['sad_span'])
      row_i += 1
      ax_heat = plt.subplot2grid(
          (grid_rows, spp['heat_cols']), (row_i, 0), colspan=spp['heat_cols'])

      # plot sequence logo
      plot_seqlogo(ax_logo_loss, seq_1hot, -delta_loss)
      if options.gain:
        plot_seqlogo(ax_logo_gain, seq_1hot, delta_gain)

      # plot SAD
      plot_sad(ax_sad, delta_loss, delta_gain)

      # plot heat map
      plot_heat(ax_heat, delta_ti.T, options.min_limit)

      # plt.tight_layout()
      plt.savefig('%s/seq%d_t%d.pdf' % (options.out_dir, si, ti), dpi=600)
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
  """ Expand

    In:
        sat_lg_ti (l array): Sat mut loss/gain scores for a single sequence and
        target.
        seq_1hot (Lx4 array): One-hot coding for a single sequence.

    Out:
        sat_loss_4l (lx4 array): Score-hot coding?

    """

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


def plot_heat(ax, sat_delta_ti, min_limit):
  """ Plot satmut deltas.

    Args:
        ax (Axis): matplotlib axis to plot to.
        sat_delta_ti (4 x L_sm array): Single target delta matrix for saturated mutagenesis region,
        min_limit (float): Minimum heatmap limit.
    """
  vlim = max(min_limit, np.nanmax(np.abs(sat_delta_ti)))
  sns.heatmap(
      sat_delta_ti,
      linewidths=0,
      cmap='RdBu_r',
      vmin=-vlim,
      vmax=vlim,
      xticklabels=False,
      ax=ax)
  ax.yaxis.set_ticklabels('ACGT', rotation='horizontal')  # , size=10)


def plot_predictions(ax, preds, satmut_len, seq_len, buffer):
  """ Plot the raw predictions for a sequence and target
        across the specificed saturated mutagenesis region.

    Args:
        ax (Axis): matplotlib axis to plot to.
        preds (L array): Target predictions for one sequence.
        satmut_len (int): Satmut length from which to determine
                           the values to plot.
        seq_len (int): Full sequence length.
        buffer (int): Ignored buffer sequence on each side
    """

  # repeat preds across pool width
  target_pool = (seq_len - 2 * buffer) // preds.shape[0]
  epreds = preds.repeat(target_pool)

  satmut_start = (epreds.shape[0] - satmut_len) // 2
  satmut_end = satmut_start + satmut_len

  ax.plot(epreds[satmut_start:satmut_end], linewidth=1)
  ax.set_xlim(0, satmut_len)
  ax.axhline(0, c='black', linewidth=1, linestyle='--')
  for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(0.5)


def plot_sad(ax, sat_loss_ti, sat_gain_ti):
  """ Plot loss and gain SAD scores.

    Args:
        ax (Axis): matplotlib axis to plot to.
        sat_loss_ti (L_sm array): Minimum mutation delta across satmut length.
        sat_gain_ti (L_sm array): Maximum mutation delta across satmut length.
    """

  rdbu = sns.color_palette('RdBu_r', 10)

  ax.plot(-sat_loss_ti, c=rdbu[0], label='loss', linewidth=1)
  ax.plot(sat_gain_ti, c=rdbu[-1], label='gain', linewidth=1)
  ax.set_xlim(0, len(sat_loss_ti))
  ax.legend()
  # ax_sad.grid(True, linestyle=':')

  ax.xaxis.set_ticks([])
  for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(0.5)


def plot_seqlogo(ax, seq_1hot, sat_score_ti, pseudo_pct=0.05):
  """ Plot a sequence logo for the loss/gain scores.

    Args:
        ax (Axis): matplotlib axis to plot to.
        seq_1hot (Lx4 array): One-hot coding of a sequence.
        sat_score_ti (L_sm array): Minimum mutation delta across satmut length.
        pseudo_pct (float): % of the max to add as a pseudocount.
    """

  satmut_len = len(sat_score_ti)

  # add pseudocounts
  sat_score_ti += pseudo_pct * sat_score_ti.max()

  # expand
  sat_score_4l = expand_4l(sat_score_ti, seq_1hot)

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
