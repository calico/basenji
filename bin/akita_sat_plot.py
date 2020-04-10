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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from basenji_sat_plot import plot_sad, plot_seqlogo
from akita_sat_vcf import plot_scd, plot_heat, subplot_params

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
  parser.add_option('-f', dest='figure_width',
      default=20, type='float',
      help='Figure width [Default: %default]')
  parser.add_option('-l', dest='plot_len',
      default=300, type='int',
      help='Length of centered sequence to mutate [Default: %default]')
  parser.add_option('-m', dest='min_limit',
      default=0.1, type='float',
      help='Minimum heatmap limit [Default: %default]')
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
      default='sqdiff',
      help='SAD stat to display [Default: %default]')
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
    sat_scores = scores_h5[options.sad_stat][si,plot_start:plot_end,:,:]

    # improved visualization
    sat_scores = np.log1p(sat_scores)

    # TEMP while specificity lacks
    sat_scores = sat_scores.mean(axis=-1, dtype='float32', keepdims=True)

    # plot max per position
    sat_max = sat_scores.max(axis=1)

    for tii in range(sat_scores.shape[-1]):
      if options.targets_file is not None:
        ti = targets_df.index[tii]
      else:
        ti = tii

      # setup plot
      plt.figure(figsize=(options.figure_width, 5))
      ax_logo_loss = plt.subplot2grid(
          (3, spp['heat_cols']), (0, spp['logo_start']),
          colspan=spp['logo_span'])
      ax_sad = plt.subplot2grid(
          (3, spp['heat_cols']), (1, spp['sad_start']),
          colspan=spp['sad_span'])
      ax_heat = plt.subplot2grid(
          (3, spp['heat_cols']), (2, 0), colspan=spp['heat_cols'])

      # plot sequence logo
      plot_seqlogo(ax_logo_loss, seq_1hot, sat_max[:,ti])

      # plot SCD
      plot_scd(ax_sad, sat_max[:,ti])

      # plot heat map
      plot_heat(ax_heat, sat_scores[:,:,ti], options.min_limit)

      plt.savefig('%s/seq%d_t%d.%s' % (options.out_dir, si, ti, save_ext), dpi=600)
      plt.close()


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
