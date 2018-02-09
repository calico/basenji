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
import os
import time

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import statsmodels
import tensorflow as tf

import basenji

################################################################################
# basenji_hidden.py
#
# Visualize the hidden representations of the test set.
################################################################################


################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <model_file> <data_file>'
  parser = OptionParser(usage)
  parser.add_option(
      '-l',
      dest='layers',
      default=None,
      help='Comma-separated list of layers to plot')
  parser.add_option(
      '-n',
      dest='num_seqs',
      default=None,
      type='int',
      help='Number of sequences to process')
  parser.add_option(
      '-o',
      dest='out_dir',
      default='hidden',
      help='Output directory [Default: %default]')
  parser.add_option(
      '-t', dest='target_indexes', default=None, help='Target indexes to plot')
  (options, args) = parser.parse_args()

  if len(args) != 3:
    parser.error('Must provide paramters, model, and test data HDF5')
  else:
    params_file = args[0]
    model_file = args[1]
    data_file = args[2]

  if not os.path.isdir(options.out_dir):
    os.mkdir(options.out_dir)

  if options.layers is not None:
    options.layers = [int(li) for li in options.layers.split(',')]

  #######################################################
  # load data
  #######################################################
  data_open = h5py.File(data_file)
  test_seqs = data_open['test_in']
  test_targets = data_open['test_out']

  if options.num_seqs is not None:
    test_seqs = test_seqs[:options.num_seqs]
    test_targets = test_targets[:options.num_seqs]

  #######################################################
  # model parameters and placeholders
  #######################################################
  job = basenji.dna_io.read_job_params(params_file)

  job['seq_length'] = test_seqs.shape[1]
  job['seq_depth'] = test_seqs.shape[2]
  job['num_targets'] = test_targets.shape[2]
  job['target_pool'] = int(np.array(data_open.get('pool_width', 1)))
  job['save_reprs'] = True

  t0 = time.time()
  model = basenji.seqnn.SeqNN()
  model.build(job)

  if options.target_indexes is None:
    options.target_indexes = range(job['num_targets'])
  else:
    options.target_indexes = [
        int(ti) for ti in options.target_indexes.split(',')
    ]

  #######################################################
  # test
  #######################################################
  # initialize batcher
  batcher_test = basenji.batcher.Batcher(
      test_seqs,
      test_targets,
      batch_size=model.batch_size,
      pool_width=model.target_pool)

  # initialize saver
  saver = tf.train.Saver()

  with tf.Session() as sess:
    # load variables into session
    saver.restore(sess, model_file)

    # get layer representations
    layer_reprs, _ = model.hidden(sess, batcher_test, options.layers)

    if options.layers is None:
      options.layers = range(len(layer_reprs))

    for li in options.layers:
      layer_repr = layer_reprs[li]
      try:
        print(layer_repr.shape)
      except:
        print(layer_repr)

      # sample one nt per sequence
      ds_indexes = np.arange(0, layer_repr.shape[1], 256)
      nt_reprs = layer_repr[:, ds_indexes, :].reshape((-1, layer_repr.shape[2]))

      ########################################################
      # plot raw
      sns.set(style='ticks', font_scale=1.2)
      plt.figure()
      g = sns.clustermap(nt_reprs, xticklabels=False, yticklabels=False)
      g.ax_heatmap.set_xlabel('Representation')
      g.ax_heatmap.set_ylabel('Sequences')
      plt.savefig('%s/l%d_reprs.pdf' % (options.out_dir, li))
      plt.close()

      ########################################################
      # plot variance explained ratios

      model_full = PCA()
      model_full.fit_transform(nt_reprs)
      evr = model_full.explained_variance_ratio_

      pca_n = 40

      plt.figure()
      plt.scatter(range(1, pca_n + 1), evr[:pca_n], c='black')
      ax = plt.gca()
      ax.set_xlim(0, pca_n + 1)
      ax.set_xlabel('Principal components')
      ax.set_ylim(0, evr[:pca_n].max() * 1.05)
      ax.set_ylabel('Variance explained')
      ax.grid(True, linestyle=':')
      plt.savefig('%s/l%d_pca.pdf' % (options.out_dir, li))
      plt.close()

      ########################################################
      # visualize in 2D

      model2 = PCA(n_components=2)
      nt_2d = model2.fit_transform(nt_reprs)

      for ti in options.target_indexes:
        nt_targets = np.log2(test_targets[:, ds_indexes, ti].flatten() + 1)

        plt.figure()
        plt.scatter(
            nt_2d[:, 0], nt_2d[:, 1], alpha=0.5, c=nt_targets, cmap='RdBu_r')
        plt.colorbar()
        ax = plt.gca()
        ax.grid(True, linestyle=':')
        plt.savefig('%s/l%d_nt2d_t%d.pdf' % (options.out_dir, li, ti))
        plt.close()

      ########################################################
      # plot neuron-neuron correlations

      # mean-normalize representation
      nt_reprs_norm = nt_reprs - nt_reprs.mean(axis=0)

      # compute covariance matrix
      hidden_cov = np.dot(nt_reprs_norm.T, nt_reprs_norm)

      plt.figure()
      g = sns.clustermap(hidden_cov, xticklabels=False, yticklabels=False)
      plt.savefig('%s/l%d_cov.pdf' % (options.out_dir, li))
      plt.close()

      ########################################################
      # plot neuron densities
      neuron_stats_out = open('%s/l%d_stats.txt' % (options.out_dir, li), 'w')

      for ni in range(nt_reprs.shape[1]):
        # print stats
        nu = nt_reprs[:, ni].mean()
        nstd = nt_reprs[:, ni].std()
        print('%3d  %6.3f  %6.3f' % (ni, nu, nstd), file=neuron_stats_out)

        # plot
        # plt.figure()
        # sns.distplot(nt_reprs[:,ni])
        # plt.savefig('%s/l%d_dist%d.pdf' % (options.out_dir,li,ni))
        # plt.close()

      neuron_stats_out.close()

      ########################################################
      # plot layer norms across length
      """
            layer_repr_norms = np.linalg.norm(layer_repr, axis=2)

            length_vec =
            list(range(layer_repr_norms.shape[1]))*layer_repr_norms.shape[0]
            length_vec = np.array(length_vec) +
            0.1*np.random.randn(len(length_vec))
            repr_norm_vec = layer_repr_norms.flatten()

            out_pdf = '%s/l%d_lnorm.pdf' % (options.out_dir,li)
            regplot(length_vec, repr_norm_vec, out_pdf, x_label='Position',
            y_label='Repr Norm')
            """

  data_open.close()


def regplot(vals1, vals2, out_pdf, alpha=0.5, x_label=None, y_label=None):
  plt.figure()

  gold = sns.color_palette('husl', 8)[1]
  ax = sns.regplot(
      vals1,
      vals2,
      color='black',
      lowess=True,
      scatter_kws={'color': 'black',
                   's': 4,
                   'alpha': alpha},
      line_kws={'color': gold})

  xmin, xmax = basenji.plots.scatter_lims(vals1)
  ymin, ymax = basenji.plots.scatter_lims(vals2)

  ax.set_xlim(xmin, xmax)
  if x_label is not None:
    ax.set_xlabel(x_label)
  ax.set_ylim(ymin, ymax)
  if y_label is not None:
    ax.set_ylabel(y_label)

  ax.grid(True, linestyle=':')

  plt.savefig(out_pdf)
  plt.close()


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
