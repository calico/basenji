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
from collections import OrderedDict
import copy
import os
import pdb
import random
import subprocess
import sys
import tempfile
import time

import h5py
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_1samp
import seaborn as sns
from sklearn.preprocessing import scale
import tensorflow as tf

import stats

from basenji import batcher
from basenji import gene
from basenji import genedata
from basenji import seqnn
from basenji import params
from basenji import plots
from basenji_test_reps import infer_replicates

"""basenji_test_genes.py

Compute accuracy statistics for a trained model at gene TSSs.
"""


################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <model_file> <genes_hdf5_file>'
  parser = OptionParser(usage)
  parser.add_option(
      '-b',
      dest='batch_size',
      default=None,
      type='int',
      help='Batch size [Default: %default]')
  parser.add_option(
      '-i',
      dest='ignore_bed',
      help='Ignore genes overlapping regions in this BED file')
  parser.add_option(
      '-l', dest='load_preds', help='Load tess_preds from file')
  parser.add_option(
      '--heat',
      dest='plot_heat',
      default=False,
      action='store_true',
      help='Plot big gene-target heatmaps [Default: %default]')
  parser.add_option(
      '-o',
      dest='out_dir',
      default='genes_out',
      help='Output directory for tables and plots [Default: %default]')
  parser.add_option(
      '-r',
      dest='tss_radius',
      default=0,
      type='int',
      help='Radius of bins considered to quantify TSS transcription [Default: %default]')
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
      dest='plot_scatter',
      default=False,
      action='store_true',
      help='Make time-consuming accuracy scatter plots [Default: %default]')
  parser.add_option(
      '--shifts',
      dest='shifts',
      default='0',
      help='Ensemble prediction shifts [Default: %default]')
  parser.add_option(
      '--rep',
      dest='replicate_labels_file',
      help=
      'Compare replicate experiments, aided by the given file with long labels')
  parser.add_option(
      '-t',
      dest='target_indexes',
      default=None,
      help=
      'File or Comma-separated list of target indexes to scatter plot true versus predicted values'
  )
  parser.add_option(
      '--table',
      dest='print_tables',
      default=False,
      action='store_true',
      help='Print big gene/TSS tables [Default: %default]')
  parser.add_option(
      '--tss',
      dest='tss_alt',
      default=False,
      action='store_true',
      help='Perform alternative TSS analysis [Default: %default]')
  parser.add_option(
      '-v',
      dest='gene_variance',
      default=False,
      action='store_true',
      help=
      'Study accuracy with respect to gene variance across targets [Default: %default]'
  )
  (options, args) = parser.parse_args()

  if len(args) != 3:
    parser.error('Must provide parameters and model files, and genes HDF5 file')
  else:
    params_file = args[0]
    model_file = args[1]
    genes_hdf5_file = args[2]

  if not os.path.isdir(options.out_dir):
    os.mkdir(options.out_dir)

  options.shifts = [int(shift) for shift in options.shifts.split(',')]

  #################################################################
  # read in genes and targets

  gene_data = genedata.GeneData(genes_hdf5_file)


  #################################################################
  # TSS predictions

  if options.load_preds is not None:
    # load from file
    tss_preds = np.load(options.load_preds)

  else:

    #######################################################
    # setup model

    job = params.read_job_params(params_file)

    job['seq_length'] = gene_data.seq_length
    job['seq_depth'] = gene_data.seq_depth
    job['target_pool'] = gene_data.pool_width
    if not 'num_targets' in job:
      job['num_targets'] = gene_data.num_targets

    # build model
    model = seqnn.SeqNN()
    model.build(job)

    if options.batch_size is not None:
      model.hp.batch_size = options.batch_size


    #######################################################
    # predict TSSs

    t0 = time.time()
    print('Computing gene predictions.', end='')
    sys.stdout.flush()

    # initialize batcher
    gene_batcher = batcher.Batcher(
        gene_data.seqs_1hot, batch_size=model.hp.batch_size)

    # initialie saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
      # load variables into session
      saver.restore(sess, model_file)

      # predict
      tss_preds = model.predict_genes(sess, gene_batcher, gene_data.gene_seqs,
          rc=options.rc, shifts=options.shifts, tss_radius=options.tss_radius)

    # save to file
    np.save('%s/preds' % options.out_dir, tss_preds)

    print(' Done in %ds.' % (time.time() - t0))

  #################################################################
  # convert to genes

  gene_targets, _ = gene.map_tss_genes(gene_data.tss_targets, gene_data.tss,
                                       tss_radius=options.tss_radius)
  gene_preds, _ = gene.map_tss_genes(tss_preds, gene_data.tss,
                                     tss_radius=options.tss_radius)


  #################################################################
  # determine targets

  # all targets
  if options.target_indexes is None:
    if gene_data.num_targets is None:
      print('No targets to test against')
      exit()
    else:
      options.target_indexes = np.arange(gene_data.num_targets)

  # file targets
  elif os.path.isfile(options.target_indexes):
    target_indexes_file = options.target_indexes
    targets_df = pd.read_table(target_indexes_file)
    options.target_indexes = targets_df.index

  # comma-separated targets
  else:
    options.target_indexes = [
        int(ti) for ti in options.target_indexes.split(',')
    ]

  options.target_indexes = np.array(options.target_indexes)

  #################################################################
  # correlation statistics

  t0 = time.time()
  print('Computing correlations.', end='')
  sys.stdout.flush()

  cor_table(gene_data.tss_targets, tss_preds, gene_data.target_ids,
            gene_data.target_labels, options.target_indexes,
            '%s/tss_cors.txt' % options.out_dir)

  cor_table(gene_targets, gene_preds, gene_data.target_ids,
            gene_data.target_labels, options.target_indexes,
            '%s/gene_cors.txt' % options.out_dir, draw_plots=True)

  print(' Done in %ds.' % (time.time() - t0))

  #################################################################
  # gene statistics

  if options.print_tables:
    t0 = time.time()
    print('Printing predictions.', end='')
    sys.stdout.flush()

    gene_table(gene_data.tss_targets, tss_preds, gene_data.tss_ids(),
               gene_data.target_labels, options.target_indexes,
               '%s/transcript' % options.out_dir, options.plot_scatter)

    gene_table(gene_targets, gene_preds,
               gene_data.gene_ids(), gene_data.target_labels,
               options.target_indexes, '%s/gene' % options.out_dir,
               options.plot_scatter)

    print(' Done in %ds.' % (time.time() - t0))

  #################################################################
  # gene x target heatmaps

  if options.plot_heat or options.gene_variance:
    #########################################
    # normalize predictions across targets

    t0 = time.time()
    print('Normalizing values across targets.', end='')
    sys.stdout.flush()

    gene_targets_qn = normalize_targets(gene_targets[:, options.target_indexes], log_pseudo=1)
    gene_preds_qn = normalize_targets(gene_preds[:, options.target_indexes], log_pseudo=1)

    print(' Done in %ds.' % (time.time() - t0))

  if options.plot_heat:
    #########################################
    # plot genes by targets clustermap

    t0 = time.time()
    print('Plotting heat maps.', end='')
    sys.stdout.flush()

    sns.set(font_scale=1.3, style='ticks')
    plot_genes = 1600
    plot_targets = 800

    # choose a set of variable genes
    gene_vars = gene_preds_qn.var(axis=1)
    indexes_var = np.argsort(gene_vars)[::-1][:plot_genes]

    # choose a set of random genes
    if plot_genes < gene_preds_qn.shape[0]:
      indexes_rand = np.random.choice(
        np.arange(gene_preds_qn.shape[0]), plot_genes, replace=False)
    else:
      indexes_rand = np.arange(gene_preds_qn.shape[0])

    # choose a set of random targets
    if plot_targets < 0.8 * gene_preds_qn.shape[1]:
      indexes_targets = np.random.choice(
          np.arange(gene_preds_qn.shape[1]), plot_targets, replace=False)
    else:
      indexes_targets = np.arange(gene_preds_qn.shape[1])

    # variable gene predictions
    clustermap(gene_preds_qn[indexes_var, :][:, indexes_targets],
               '%s/gene_heat_var.pdf' % options.out_dir)
    clustermap(
        gene_preds_qn[indexes_var, :][:, indexes_targets],
        '%s/gene_heat_var_color.pdf' % options.out_dir,
        color='viridis',
        table=True)

    # random gene predictions
    clustermap(gene_preds_qn[indexes_rand, :][:, indexes_targets],
               '%s/gene_heat_rand.pdf' % options.out_dir)

    # variable gene targets
    clustermap(gene_targets_qn[indexes_var, :][:, indexes_targets],
               '%s/gene_theat_var.pdf' % options.out_dir)
    clustermap(
        gene_targets_qn[indexes_var, :][:, indexes_targets],
        '%s/gene_theat_var_color.pdf' % options.out_dir,
        color='viridis',
        table=True)

    # random gene targets (crashes)
    # clustermap(gene_targets_qn[indexes_rand, :][:, indexes_targets],
    #            '%s/gene_theat_rand.pdf' % options.out_dir)

    print(' Done in %ds.' % (time.time() - t0))

  #################################################################
  # analyze replicates

  if options.replicate_labels_file is not None:
    # read long form labels, from which to infer replicates
    target_labels_long = []
    for line in open(options.replicate_labels_file):
      a = line.split('\t')
      a[-1] = a[-1].rstrip()
      target_labels_long.append(a[-1])

    # determine replicates
    replicate_lists = infer_replicates(target_labels_long)

    # compute correlations
    # replicate_correlations(replicate_lists, gene_data.tss_targets,
        # tss_preds, options.target_indexes, '%s/transcript_reps' % options.out_dir)
    replicate_correlations(
        replicate_lists, gene_targets, gene_preds, options.target_indexes,
        '%s/gene_reps' % options.out_dir)  # , scatter_plots=True)

  #################################################################
  # gene variance

  if options.gene_variance:
    variance_accuracy(gene_targets_qn, gene_preds_qn,
                      '%s/gene' % options.out_dir)

  #################################################################
  # alternative TSS

  if options.tss_alt:
    alternative_tss(gene_data.tss_targets[:,options.target_indexes],
                    tss_preds[:,options.target_indexes], gene_data,
                    options.out_dir, log_pseudo=1)


def alternative_tss(tss_targets, tss_preds, gene_data, out_base, log_pseudo=1, tss_var_t=1, scatter_pct=0.02):
  ''' Compare predicted to experimental log2 TSS1 to TSS2 ratio. '''

  sns.set(style='ticks', font_scale=1.2)

  # normalize TSS
  tss_targets_qn = normalize_targets(tss_targets, log_pseudo=log_pseudo)
  tss_preds_qn = normalize_targets(tss_preds, log_pseudo=log_pseudo)

  # compute
  tss_targets_var = tss_targets_qn.var(axis=1, dtype='float64')

  # save genes for later plotting
  gene_tss12_targets = []
  gene_tss12_preds = []
  gene_ids = []

  # output correlations
  table_out = open('%s/tss12_cor.txt' % out_base, 'w')
  gene_tss12_cors = []

  for gene_id in gene_tss:
    # sort TSS by variance
    var_tss_list = [(tss_targets_var[tss_i],tss_i) for tss_i in gene_data.gene_tss[gene_id]]
    var_tss_list.sort(reverse=True)

    # filter for high variance
    tss_list = [tss_i for (tss_var,tss_i) in var_tss_list if tss_var > tss_var_t]

    # filter for sufficient distance from TSS1
    if len(tss_list) > 1:
      tss1_pos = gene_data.tss[tss_list[0]].pos
      tss_list = [tss_list[0]] + [tss_i for tss_i in tss_list[1:] if abs(gene_data.tss[tss_i].pos - tss1_pos) > 500]

    if len(tss_list) > 1:
      tss_i1 = tss_list[0]
      tss_i2 = tss_list[1]

      # compute log2 ratio (already log2)
      tss12_targets = tss_targets_qn[tss_i1,:] - tss_targets_qn[tss_i2,:]
      tss12_preds = tss_preds_qn[tss_i1,:] - tss_preds_qn[tss_i2,:]

      # convert
      tss12_targets = tss12_targets.astype('float32')
      tss12_preds = tss12_preds.astype('float32')

      # save values
      gene_tss12_targets.append(tss12_targets)
      gene_tss12_preds.append(tss12_preds)
      gene_ids.append(gene_id)

      # compute correlation
      pcor, p = pearsonr(tss12_targets, tss12_preds)
      gene_tss12_cors.append(pcor)

      print('%-20s  %7.4f' % (gene_id, pcor), file=table_out)

  table_out.close()

  gene_tss12_cors = np.array(gene_tss12_cors)

  # T-test PearsonR > 0
  _, tp = ttest_1samp(gene_tss12_cors, 0)

  # plot PearsonR distribution
  plt.figure(figsize=(6.5,4))
  sns.distplot(gene_tss12_cors, axlabel='TSS1/TSS2 PearsonR') # , color='black')
  ax = plt.gca()
  ax.axvline(0, linestyle='--', color='black')
  xmin, xmax = ax.get_xlim()
  ymin, ymax = ax.get_ylim()
  ax.text(xmax*0.98, ymax*0.92, 'p-val < %.2e' % p, horizontalalignment='right')
  plt.tight_layout()
  plt.savefig('%s/tss12_cor.pdf' % out_base)
  plt.close()

  # save gene values for later plotting (gene_ids's in the table)
  np.save('%s/gene_tss12_targets.npy' % out_base, np.array(gene_tss12_targets, dtype='float16'))
  np.save('%s/gene_tss12_preds.npy' % out_base, np.array(gene_tss12_preds, dtype='float16'))

  # choose a range of percentiles
  genes_out = open('%s/tss12_qgenes.txt' % out_base, 'w')
  cor_indexes = np.argsort(gene_tss12_cors)
  pct_indexes = np.linspace(0, len(cor_indexes)-1, 10+1).astype('int')
  for qi in range(len(pct_indexes)):
    pct_i = pct_indexes[qi]
    cor_i = cor_indexes[pct_i]

    out_pdf = '%s/tss12_q%d.pdf' % (out_base, qi)
    plots.regplot(gene_tss12_targets[cor_i], gene_tss12_preds[cor_i],
                  out_pdf, poly_order=1, alpha=0.8, point_size=8,
                  square=False, figsize=(4,4),
                  x_label='log2 Experiment TSS1/TSS2',
                  y_label='log2 Prediction TSS1/TSS2',
                  title=gene_ids[cor_i])

    print(qi, gene_ids[cor_i], file=genes_out)

  genes_out.close()


def clustermap(gene_values, out_pdf, color=None, table=False):
  """ Generate a clustered heatmap using seaborn. """

  if table:
    np.save(out_pdf[:-4], gene_values)

  plt.figure()
  g = sns.clustermap(
      gene_values,
      metric='euclidean',
      cmap=color,
      xticklabels=False,
      yticklabels=False)
  g.ax_heatmap.set_xlabel('Experiments')
  g.ax_heatmap.set_ylabel('Genes')
  plt.savefig(out_pdf)
  plt.close()


def cor_table(gene_targets,
              gene_preds,
              target_ids,
              target_labels,
              target_indexes,
              out_file,
              draw_plots=False):
  """ Print a table and plot the distribution of target correlations. """

  table_out = open(out_file, 'w')
  cors = []
  cors_nz = []

  for ti in target_indexes:
    # convert targets and predictions to float32
    gti = np.array(gene_targets[:, ti], dtype='float32')
    gpi = np.array(gene_preds[:, ti], dtype='float32')

    # log transform
    gti = np.log2(gti + 1)
    gpi = np.log2(gpi + 1)

    # compute correlations
    scor, _ = spearmanr(gti, gpi)
    pcor, _ = pearsonr(gti, gpi)
    cors.append(pcor)

    # compute non-zero correlations
    nzi = (gti > 0)
    scor_nz, _ = spearmanr(gti[nzi], gpi[nzi])
    pcor_nz, _ = pearsonr(gti[nzi], gpi[nzi])
    cors_nz.append(pcor_nz)

    # print
    cols = (ti, scor, pcor, scor_nz, pcor_nz, target_ids[ti], target_labels[ti])
    print('%-4d  %7.3f  %7.3f  %7.3f  %7.3f  %s %s' % cols, file=table_out)

  cors = np.array(cors)
  cors_nz = np.array(cors_nz)
  table_out.close()

  if draw_plots:
    # plot correlation distribution
    out_base = os.path.splitext(out_file)[0]
    sns.set(style='ticks', font_scale=1.3)

    # plot correlations versus target signal
    gene_targets_log = np.log2(gene_targets[:, target_indexes] + 1)
    target_signal = gene_targets_log.sum(axis=0, dtype='float64')
    plots.jointplot(
        target_signal,
        cors,
        '%s_sig.pdf' % out_base,
        x_label='Aligned TSS reads',
        y_label='Pearson R',
        cor=None,
        table=True)

    # plot nonzero correlations versus target signal
    plots.jointplot(
        target_signal,
        cors_nz,
        '%s_nz_sig.pdf' % out_base,
        x_label='Aligned TSS reads',
        y_label='Pearson R',
        cor=None,
        table=True)

  return cors


def gene_table(gene_targets, gene_preds, gene_iter, target_labels,
               target_indexes, out_prefix, plot_scatter):
  """Print a gene-based statistics table and scatter plot for the given target indexes."""

  num_genes = gene_targets.shape[0]

  table_out = open('%s_table.txt' % out_prefix, 'w')

  for ti in target_indexes:
    gti = np.log2(gene_targets[:, ti].astype('float32') + 1)
    gpi = np.log2(gene_preds[:, ti].astype('float32') + 1)

    # plot scatter
    if plot_scatter:
      sns.set(font_scale=1.3, style='ticks')
      out_pdf = '%s_scatter%d.pdf' % (out_prefix, ti)
      if num_genes < 2000:
        ri = np.arange(num_genes)
      else:
        ri = np.random.choice(range(num_genes), 2000, replace=False)
      plots.regplot(
          gti[ri],
          gpi[ri],
          out_pdf,
          poly_order=3,
          alpha=0.3,
          x_label='log2 Experiment',
          y_label='log2 Prediction')

    # print table lines
    tx_i = 0
    for gid in gene_iter:
      # print TSS line
      cols = (gid, gti[tx_i], gpi[tx_i], ti, target_labels[ti])
      print('%-20s  %.3f  %.3f  %4d  %20s' % cols, file=table_out)
      tx_i += 1

  table_out.close()

  subprocess.call('gzip -f %s_table.txt' % out_prefix, shell=True)


def normalize_targets(gene_values, log_pseudo=1, outlier_mult=10):
  """ Normalize gene-target values across targets. """

  # take log
  if log_pseudo is not None:
    gene_values = np.log2(gene_values + log_pseudo)

  # identify outliers
  gene_values_tmean = gene_values.mean(axis=0, dtype='float32')
  gene_values_tmmean = gene_values_tmean.mean()

  inlier_indexes = []
  for ti in range(len(gene_values_tmean)):
    if gene_values_tmean[ti] > outlier_mult*gene_values_tmmean or gene_values_tmean[ti] < gene_values_tmmean/outlier_mult:
      print('%d (filtered) outlies: %.3f versus %.3f (%.4f)' % (ti, gene_values_tmean[ti], gene_values_tmmean, gene_values_tmean[ti] / gene_values_tmmean))
    else:
      if gene_values_tmean[ti] > (0.5*outlier_mult)*gene_values_tmmean or gene_values_tmean[ti] < gene_values_tmmean/(0.5*outlier_mult):
        print('%d (semi) outlies: %.3f versus %.3f (%.4f)' % (ti, gene_values_tmean[ti], gene_values_tmmean, gene_values_tmean[ti] / gene_values_tmmean))

      inlier_indexes.append(ti)

  inlier_indexes = np.array(inlier_indexes, dtype='int')

  # filter outliers
  gene_values = gene_values[:,inlier_indexes]

  # quantile normalize
  gene_values_qn = quantile_normalize(gene_values, quantile_stat='mean')

  return gene_values_qn


def quantile_normalize(gene_expr, quantile_stat='median'):
  """ Quantile normalize across targets. """

  # make a copy
  gene_expr_qn = copy.copy(gene_expr)

  # sort values within each column
  for ti in range(gene_expr.shape[1]):
    gene_expr_qn[:, ti].sort()

  # compute the mean/median in each row
  if quantile_stat == 'median':
    sorted_index_stats = np.median(gene_expr_qn, axis=1)
  elif quantile_stat == 'mean':
    sorted_index_stats = np.mean(gene_expr_qn, axis=1)
  else:
    print('Unrecognized quantile statistic %s' % quantile_stat, file=sys.stderr)
    exit()

  # set new values
  for ti in range(gene_expr.shape[1]):
    sorted_indexes = np.argsort(gene_expr[:, ti])
    for gi in range(gene_expr.shape[0]):
      gene_expr_qn[sorted_indexes[gi], ti] = sorted_index_stats[gi]

  return gene_expr_qn


def replicate_correlations(replicate_lists,
                           gene_targets,
                           gene_preds,
                           target_indexes,
                           out_prefix,
                           scatter_plots=False):
  """ Study replicate correlations. """

  # for intersections
  target_set = set(target_indexes)

  rep_cors = []
  pred_cors = []

  table_out = open('%s.txt' % out_prefix, 'w')
  sns.set(style='ticks', font_scale=1.3)
  num_genes = gene_targets.shape[0]

  li = 0
  replicate_labels = sorted(replicate_lists.keys())

  for label in replicate_labels:
    if len(replicate_lists[label]) > 1 and target_set & set(
        replicate_lists[label]):
      ti1 = replicate_lists[label][0]
      ti2 = replicate_lists[label][1]

      # retrieve targets
      gene_targets_rep1 = np.log2(gene_targets[:, ti1].astype('float32') + 1)
      gene_targets_rep2 = np.log2(gene_targets[:, ti2].astype('float32') + 1)

      # retrieve predictions
      gene_preds_rep1 = np.log2(gene_preds[:, ti1].astype('float32') + 1)
      gene_preds_rep2 = np.log2(gene_preds[:, ti2].astype('float32') + 1)

      #####################################
      # replicate

      # compute replicate correlation
      rcor, _ = pearsonr(gene_targets_rep1, gene_targets_rep2)
      rep_cors.append(rcor)

      # scatter plot rep vs rep
      if scatter_plots:
        out_pdf = '%s_s%d.pdf' % (out_prefix, li)
        if num_genes < 1000:
          gene_indexes = np.arange(num_genes)
        else:
          gene_indexes = np.random.choice(range(num_genes), 1000, replace=False)
        plots.regplot(
            gene_targets_rep1[gene_indexes],
            gene_targets_rep2[gene_indexes],
            out_pdf,
            poly_order=3,
            alpha=0.3,
            x_label='log2 Replicate 1',
            y_label='log2 Replicate 2')

      #####################################
      # prediction

      # compute prediction correlation
      pcor1, _ = pearsonr(gene_targets_rep1, gene_preds_rep1)
      pcor2, _ = pearsonr(gene_targets_rep2, gene_preds_rep2)
      pcor = 0.5 * pcor1 + 0.5 * pcor2
      pred_cors.append(pcor)

      # scatter plot vs pred
      if scatter_plots:
        # scatter plot rep vs pred
        out_pdf = '%s_s%d_rep1.pdf' % (out_prefix, li)
        plots.regplot(
            gene_targets_rep1[gene_indexes],
            gene_preds_rep1[gene_indexes],
            out_pdf,
            poly_order=3,
            alpha=0.3,
            x_label='log2 Experiment',
            y_label='log2 Prediction')

        # scatter plot rep vs pred
        out_pdf = '%s_s%d_rep2.pdf' % (out_prefix, li)
        plots.regplot(
            gene_targets_rep2[gene_indexes],
            gene_preds_rep2[gene_indexes],
            out_pdf,
            poly_order=3,
            alpha=0.3,
            x_label='log2 Experiment',
            y_label='log2 Prediction')

      #####################################
      # table

      print(
          '%4d  %4d  %4d  %7.4f  %7.4f  %s' % (li, ti1, ti2, rcor, pcor, label),
          file=table_out)

      # update counter
      li += 1

  table_out.close()

  #######################################################
  # scatter plot replicate versus prediction correlation

  rep_cors = np.array(rep_cors)
  pred_cors = np.array(pred_cors)

  out_pdf = '%s_scatter.pdf' % out_prefix
  plots.jointplot(
      rep_cors,
      pred_cors,
      out_pdf,
      square=True,
      x_label='Replicate R',
      y_label='Prediction R')


def quantile_accuracy(gene_targets, gene_preds, gene_stat, out_pdf, numq=4):
  ''' Plot accuracy (PearsonR) in quantile bins across targets. '''

  # plot PearsonR in variance statistic bins
  quant_indexes = stats.quantile_indexes(gene_stat, numq)

  quantiles_series = []
  targets_series = []
  pcor_series = []

  for qi in range(numq):
    # slice quantile
    gene_targets_quant = gene_targets[quant_indexes[qi]].astype('float32')
    gene_preds_quant = gene_preds[quant_indexes[qi]].astype('float32')

    # compute target PearsonR
    for ti in range(gene_targets_quant.shape[1]):
      pcor, _ = pearsonr(gene_targets_quant[:,ti],
                          gene_preds_quant[:,ti])

      quantiles_series.append(qi)
      targets_series.append(ti)
      pcor_series.append(pcor)

  # construct DataFrame
  df_quant = pd.DataFrame({'Quantile':quantiles_series,
                            'Target':targets_series,
                            'PearsonR':pcor_series})
  df_quant.to_csv('%s.csv' % out_pdf[:-4])

  # print summary table
  table_out = open('%s.txt' % out_pdf[:-4], 'w')
  for qi in range(numq):
    quantile_cors = df_quant[df_quant.Quantile == qi].PearsonR
    print('%2d  %.4f  %.4f' % \
          (qi, np.mean(quantile_cors),np.median(quantile_cors)),
          file=table_out)
  table_out.close()

  # construct figure
  plt.figure()

  # plot individual targets as light lines
  for ti in range(gene_targets.shape[1]):
    df_quant_target = df_quant[df_quant.Target == ti]
    plt.plot(df_quant_target.Quantile, df_quant_target.PearsonR, alpha=0.1)

  # plot PearsonR distributions in quantiles
  sns.violinplot(x='Quantile', y='PearsonR', data=df_quant, color='tomato')

  plt.savefig(out_pdf)
  plt.close()

  # sort targets by their decrease
  target_ratios = []
  for ti in range(gene_targets.shape[1]):
    df_quant_target = df_quant[df_quant.Target == ti]
    assert(df_quant_target.Quantile.iloc[0] == 0)
    assert(df_quant_target.Quantile.iloc[-1] == numq-1)
    cor_ratio = df_quant_target.PearsonR.iloc[-1] / df_quant_target.PearsonR.iloc[0]
    target_ratios.append((cor_ratio,ti))
  target_ratios = sorted(target_ratios)

  # take 10 samples across
  pct_indexes = np.linspace(0, len(target_ratios)-1, 10+1).astype('int')

  # write quantile targets
  table_out = open('%s_qt.txt' % out_pdf[:-4], 'w')
  sns.set(font_scale=1.2, style='ticks')

  # scatter plot each quantile
  for qi in range(numq):
    # slice quantile
    gene_targets_quant = gene_targets[quant_indexes[qi]].astype('float32')
    gene_preds_quant = gene_preds[quant_indexes[qi]].astype('float32')

    for pqi in range(len(pct_indexes)):
      pct_i = pct_indexes[pqi]
      ti = target_ratios[pct_i][1]

      print(qi, pqi, ti, target_ratios[ti], file=table_out)

      qout_pdf = '%s_pq%d_q%d.pdf' % (out_pdf[:-4], pqi, qi)
      plots.jointplot(gene_targets_quant[:,ti], gene_preds_quant[:,ti],
                              qout_pdf, alpha=0.8, point_size=8, kind='reg',
                              figsize=5, x_label='log2 Experiment',
                              y_label='log2 Prediction')

  table_out.close()


def variance_accuracy(gene_targets, gene_preds, out_prefix, log_pseudo=None):
  """ Compare MSE accuracy to gene mean and variance.

    Assumes the targets and predictions have been normalized.
    """

  # compute mean, var, and MSE across targets
  print('gene_targets', gene_targets.shape)
  gene_mean = np.mean(gene_targets, axis=1, dtype='float64')
  gene_max = np.max(gene_targets, axis=1)
  gene_std = np.std(gene_targets, axis=1, dtype='float64')
  gene_mse = np.power(gene_targets - gene_preds, 2).mean(axis=1, dtype='float64')

  # filter for sufficient expression
  expr_indexes = (gene_mean > 0.5) & (gene_max > 3)
  gene_targets = gene_targets[expr_indexes,:]
  gene_preds = gene_preds[expr_indexes,:]
  gene_mse = gene_mse[expr_indexes]
  gene_mean = gene_mean[expr_indexes]
  gene_std = gene_std[expr_indexes]
  print('%d "expressed genes" considered in variance plots' % expr_indexes.sum())


  sns.set(style='ticks', font_scale=1.3)
  if len(gene_mse) < 2000:
    ri = np.arange(len(gene_mse))
  else:
    ri = np.random.choice(np.arange(len(gene_mse)), 2000, replace=False)

  # plot mean vs std
  out_pdf = '%s_mean-std.pdf' % out_prefix
  plots.jointplot(gene_mean[ri], gene_std[ri], out_pdf, point_size=10,
    cor='spearmanr', x_label='Mean across experiments', y_label='Std Dev across experiments')

  # plot mean vs MSE
  out_pdf = '%s_mean.pdf' % out_prefix
  plots.jointplot(gene_mean[ri], gene_mse[ri], out_pdf, point_size=10,
    cor='spearmanr', x_label='Mean across experiments', y_label='Mean squared prediction error')

  # plot std vs MSE
  out_pdf = '%s_std.pdf' % out_prefix
  plots.jointplot(gene_std[ri], gene_mse[ri], out_pdf, point_size=10,
    cor='spearmanr', x_label='Std Dev across experiments', y_label='Mean squared prediction error')

  # plot CV vs MSE
  gene_cv = np.divide(gene_std, gene_mean)
  out_pdf = '%s_cv.pdf' % out_prefix
  plots.jointplot(gene_cv[ri], gene_mse[ri], out_pdf, point_size=10,
    cor='spearmanr', x_label='Coef Var across experiments', y_label='Mean squared prediction error')


  # plot MSE distributions in CV bins
  numq = 5
  quant_indexes = stats.quantile_indexes(gene_cv, numq)
  quant_mse = []
  for qi in range(numq):
    for gi in quant_indexes[qi]:
      quant_mse.append([qi, gene_mse[gi]])
  quant_mse = pd.DataFrame(quant_mse, columns=['Quantile','MSE'])

  quant_mse.to_csv('%s_quant.txt' % out_prefix, sep='\t')

  plt.figure()
  sns.boxplot(x='Quantile', y='MSE', data=quant_mse, palette=sns.cubehelix_palette(numq), showfliers=False)
  ax = plt.gca()
  ax.grid(True, linestyle=':')
  ax.set_ylabel('Mean squared prediction error')
  plt.savefig('%s_quant.pdf' % out_prefix)
  plt.close()

  # CV quantiles
  quantile_accuracy(gene_targets, gene_preds, gene_cv, '%s_qcv.pdf'%out_prefix, 4)

  # stdev quantiles
  quantile_accuracy(gene_targets, gene_preds, gene_std, '%s_qstd.pdf'%out_prefix, 4)


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
  # pdb.runcall(main)
