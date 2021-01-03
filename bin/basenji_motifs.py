#!/usr/bin/env python
# Copyright 2020 Calico LLC
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
import multiprocessing
import os
import pdb
import subprocess
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import seaborn as sns
from sklearn import preprocessing

import tensorflow as tf
if tf.__version__[0] == '1':
  tf.compat.v1.enable_eager_execution()

from basenji import dataset
from basenji import dna_io
from basenji import seqnn

'''
basenji_motifs.py

Collect statistics and make plots to explore the first convolution layer
of the given model using the given sequences.
'''

weblogo_opts = '-X NO -Y NO --errorbars NO --fineprint ""'
weblogo_opts += ' -C "#CB2026" A A'
weblogo_opts += ' -C "#34459C" C C'
weblogo_opts += ' -C "#FBB116" G G'
weblogo_opts += ' -C "#0C8040" T T'

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <model_file> <data_dir>'
  parser = OptionParser(usage)
  parser.add_option('-a', dest='act_t',
      default=0.5, type='float',
      help='Activation threshold (as proportion of max) to consider for PWM [Default: %default]')
  parser.add_option('-c', dest='post_conv',
                    default=False, action='store_true',
                    help='Embed first layer post-convolution rather than batch norm [Default: %default]')
  parser.add_option('-d', dest='plot_density',
                    default=False, action='store_true',
                    help='Plot filter activation density [Default: %default]')
  parser.add_option('--heat', dest='plot_heats',
      default=False, action='store_true',
      help='Plot heat maps describing filter activations in the test sequences [Default: %default]')
  parser.add_option('-l', dest='seq_length_crop',
      default=None, type='int',
      help='Crop sequences to shorter length [Default: %default]')
  parser.add_option('-o', dest='out_dir',
      default='basenji_motifs')
  parser.add_option('-m', dest='meme_db',
      default='%s/cisbp/Homo_sapiens.meme' % os.environ['HG38'],
      help='MEME database used to annotate motifs')
  parser.add_option('-p', dest='parallel_threads',
      default=1, type='int',
      help='Generate weblogos in parallal threads [Default: %default]')
  parser.add_option('-s', dest='sample',
      default=None, type='int',
      help='Sample sequences from the test set [Default:%default]')
  parser.add_option('-t', dest='trim_filters',
      default=False, action='store_true',
      help='Trim uninformative positions off the filter ends [Default: %default]')
  parser.add_option('--tfr', dest='tfr_pattern',
      default=None,
      help='TFR pattern string appended to data_dir/tfrecords for subsetting [Default: %default]')
  parser.add_option('-v', dest='high_var_pct',
      default=1.0, type='float',
      help='Highly variable site proportion to take [Default: %default]')
  (options, args) = parser.parse_args()

  if len(args) != 3:
    parser.error('Must provide Basenji params and model files and data directory')
  else:
    params_file = args[0]
    model_file = args[1]
    data_dir = args[2]

  if not os.path.isdir(options.out_dir):
    os.mkdir(options.out_dir)

  #######################################################
  # inputs

  # read model parameters
  with open(params_file) as params_open:
    params = json.load(params_open)
  params_model = params['model']
  params_train = params['train']
  if options.seq_length_crop is not None:
    params_model['seq_length'] = options.seq_length_crop

  # construct data
  eval_data = dataset.SeqDataset(data_dir,
    split_label=options.split_label,
    batch_size=params_train['batch_size'],
    mode='eval',
    tfr_pattern=options.tfr_pattern)

  # obtain sequences
  eval_seqs_1hot = eval_data.numpy(return_inputs=True, return_outputs=False)
  eval_seqs_dna = dna_io.hot1_dna(eval_seqs_1hot)
  del eval_seqs_1hot

  #################################################################
  # model

  # initialize model
  seqnn_model = seqnn.SeqNN(params_model)
  seqnn_model.restore(model_file)

  # first layer embedding
  seqnn_model.build_embed(0, batch_norm=(~options.post_conv))
  _, preds_length, preds_depth  = seqnn_model.embed.output.shape 

  # get weights
  filter_weights = seqnn_model.get_conv_weights()
  print(filter_weights.shape)
  num_filters, _, filter_size = filter_weights.shape

  # compute filter activations
  filter_outs = seqnn_model.predict(eval_data)
  print(filter_outs.shape)

  #################################################################
  # individual filter plots

  # save information contents
  filters_ic = []
  meme_out = meme_intro('%s/filters_meme.txt' % options.out_dir, eval_seqs_dna)

  # plot weblogo of high scoring outputs (in parallel)
  if options.parallel_threads > 1:
    pfl_args = []
    for f in range(num_filters):
      pfl_args.append((filter_outs[:, :, f], filter_size,
          eval_seqs_dna, '%s/filter%d_logo'%(options.out_dir,f),
          options.act_t))
    with multiprocessing.get_context('spawn').Pool(options.parallel_threads) as pool:
      pool.starmap(plot_filter_logo, pfl_args)

  for f in range(num_filters):
    print('Filter %d' % f)

    # plot filter parameters as a heatmap
    plot_filter_heat(filter_weights[f, :, :],
                     '%s/filter%d_heat.pdf' % (options.out_dir, f))

    if options.parallel_threads == 1:
      plot_filter_logo(filter_outs[:, :, f], filter_size,
          eval_seqs_dna, '%s/filter%d_logo'%(options.out_dir,f),
          options.act_t)

    # write possum motif file
    # filter_possum(filter_weights[f, :, :], 'filter%d' % f,
    #               '%s/filter%d_possum.txt' % (options.out_dir,
    #                                           f), options.trim_filters)

    # make a PWM for the filter
    filter_pwm, nsites = make_filter_pwm('%s/filter%d_logo.fa' %
                                         (options.out_dir, f))

    if nsites < 10:
      # no information
      filters_ic.append(0)
    else:
      # compute and save information content
      filters_ic.append(info_content(filter_pwm))

      # add to the meme motif file
      meme_add(meme_out, f, filter_pwm, nsites, options.trim_filters)

  meme_out.close()

  #################################################################
  # annotate filters
  #################################################################
  # run tomtom
  subprocess.call(
      'tomtom -dist pearson -thresh 0.1 -oc %s/tomtom %s/filters_meme.txt %s' %
      (options.out_dir, options.out_dir, options.meme_db),
      shell=True)

  # read in annotations
  filter_names = name_filters(
      num_filters, '%s/tomtom/tomtom.tsv' % options.out_dir, options.meme_db)

  #################################################################
  # print a table of information
  #################################################################
  table_out = open('%s/table.txt' % options.out_dir, 'w')

  # print header for later panda reading
  header_cols = ('', 'consensus', 'annotation', 'ic', 'mean', 'std')
  print('%3s  %19s  %10s  %5s  %6s  %6s' % header_cols, file=table_out)

  for f in range(num_filters):
    # collapse to a consensus motif
    consensus = filter_motif(filter_weights[f, :, :])

    # grab annotation
    annotation = '.'
    name_pieces = filter_names[f].split('_')
    if len(name_pieces) > 1:
      annotation = name_pieces[1]

    f_scores = np.ravel(filter_outs[:, :, f])
    fmean, fstd = f_scores.mean(), f_scores.std()
    if options.plot_density:
      # plot density of filter output scores
      plot_score_density(f_scores,
          '%s/filter%d_dens.pdf' % (options.out_dir, f))      

    row_cols = (f, consensus, annotation, filters_ic[f], fmean, fstd)
    print('%-3d  %19s  %10s  %5.2f  %6.4f  %6.4f' % row_cols, file=table_out)

  table_out.close()

  #################################################################
  # global filter plots
  #################################################################

  # these methods make less sense for longer sequences;
  # I should fragment the sequences first.

  if options.plot_heats:
    # plot filter-sequence heatmap
    plot_filter_seq_heat(filter_outs, '%s/filter_seqs.pdf' % options.out_dir)

    # plot filter-segment heatmap
    plot_filter_seg_heat(filter_outs, '%s/filter_segs.pdf' % options.out_dir)
    plot_filter_seg_heat(
        filter_outs, '%s/filter_segs_raw.pdf' % options.out_dir, whiten=False)

    # plot filter-target correlation heatmap
    plot_target_corr(filter_outs, seq_targets, filter_names, target_names,
                     '%s/filter_target_cors_mean.pdf' % options.out_dir, 'mean')
    plot_target_corr(filter_outs, seq_targets, filter_names, target_names,
                     '%s/filter_target_cors_max.pdf' % options.out_dir, 'max')


def get_motif_proteins(meme_db_file):
  """ Hash motif_id's to protein names using the MEME DB file """
  motif_protein = {}
  for line in open(meme_db_file):
    a = line.split()
    if len(a) > 0 and a[0] == 'MOTIF':
      if a[2][0] == '(':
        motif_protein[a[1]] = a[2][1:a[2].find(')')]
      else:
        motif_protein[a[1]] = a[2]
  return motif_protein


def info_content(pwm, transpose=False, bg_gc=0.415):
  """ Compute PWM information content.

    In the original analysis, I used a bg_gc=0.5. For any
    future analysis, I ought to switch to the true hg19
    value of 0.415.
    """
  pseudoc = 1e-9

  if transpose:
    pwm = np.transpose(pwm)

  bg_pwm = [1 - bg_gc, bg_gc, bg_gc, 1 - bg_gc]

  ic = 0
  for i in range(pwm.shape[0]):
    for j in range(4):
      # ic += 0.5 + pwm[i][j]*np.log2(pseudoc+pwm[i][j])
      ic += -bg_pwm[j] * np.log2(
          bg_pwm[j]) + pwm[i][j] * np.log2(pseudoc + pwm[i][j])

  return ic


def make_filter_pwm(filter_fasta):
  """ Make a PWM for this filter from its top hits """

  nts = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
  pwm_counts = []
  nsites = 4  # pseudocounts
  for line in open(filter_fasta):
    if line[0] != '>':
      seq = line.rstrip()
      nsites += 1
      if len(pwm_counts) == 0:
        # initialize with the length
        for i in range(len(seq)):
          pwm_counts.append(np.array([1.0] * 4))

      # count
      for i in range(len(seq)):
        try:
          pwm_counts[i][nts[seq[i]]] += 1
        except KeyError:
          pwm_counts[i] += np.array([0.25] * 4)

  # normalize
  pwm_freqs = []
  for i in range(len(pwm_counts)):
    pwm_freqs.append([pwm_counts[i][j] / float(nsites) for j in range(4)])

  return np.array(pwm_freqs), nsites - 4


def meme_add(meme_out, f, filter_pwm, nsites, trim_filters=False):
  """ Print a filter to the growing MEME file

    Attrs:
        meme_out : open file
        f (int) : filter index #
        filter_pwm (array) : filter PWM array
        nsites (int) : number of filter sites
    """
  if not trim_filters:
    ic_start = 0
    ic_end = filter_pwm.shape[0] - 1
  else:
    ic_t = 0.2

    # trim PWM of uninformative prefix
    ic_start = 0
    while ic_start < filter_pwm.shape[0] and info_content(
        filter_pwm[ic_start:ic_start + 1]) < ic_t:
      ic_start += 1

    # trim PWM of uninformative suffix
    ic_end = filter_pwm.shape[0] - 1
    while ic_end >= 0 and info_content(filter_pwm[ic_end:ic_end + 1]) < ic_t:
      ic_end -= 1

  if ic_start < ic_end:
    print('MOTIF filter%d' % f, file=meme_out)
    print(
        'letter-probability matrix: alength= 4 w= %d nsites= %d' %
        (ic_end - ic_start + 1, nsites),
        file=meme_out)

    for i in range(ic_start, ic_end + 1):
      print('%.4f %.4f %.4f %.4f' % tuple(filter_pwm[i]), file=meme_out)
    print('', file=meme_out)


def meme_intro(meme_file, seqs):
  """ Open MEME motif format file and print intro

    Attrs:
        meme_file (str) : filename
        seqs [str] : list of strings for obtaining background freqs

    Returns:
        mem_out : open MEME file
    """
  nts = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

  # count
  nt_counts = [1] * 4
  for i in range(len(seqs)):
    for nt in seqs[i]:
      try:
        nt_counts[nts[nt]] += 1
      except KeyError:
        pass

  # normalize
  nt_sum = float(sum(nt_counts))
  nt_freqs = [nt_counts[i] / nt_sum for i in range(4)]

  # open file for writing
  meme_out = open(meme_file, 'w')

  # print intro material
  print('MEME version 4', file=meme_out)
  print('', file=meme_out)
  print('ALPHABET= ACGT', file=meme_out)
  print('', file=meme_out)
  print('Background letter frequencies:', file=meme_out)
  print('A %.4f C %.4f G %.4f T %.4f' % tuple(nt_freqs), file=meme_out)
  print('', file=meme_out)

  return meme_out


def name_filters(num_filters, tomtom_file, meme_db_file):
  """ Name the filters using Tomtom matches.

    Attrs:
        num_filters (int) : total number of filters
        tomtom_file (str) : filename of Tomtom output table.
        meme_db_file (str) : filename of MEME db

    Returns:
        filter_names [str] :
    """
  # name by number
  filter_names = ['f%d' % fi for fi in range(num_filters)]

  # name by protein
  if tomtom_file is not None and meme_db_file is not None:
    motif_protein = get_motif_proteins(meme_db_file)

    # hash motifs and q-value's by filter
    filter_motifs = {}

    tt_in = open(tomtom_file)
    tt_in.readline()
    for line in tt_in:
      a = line.split()
      if line[0] != '#' and len(a) > 0:
        fi = int(a[0][6:])
        motif_id = a[1]
        qval = float(a[5])
        filter_motifs.setdefault(fi, []).append((qval, motif_id))

    tt_in.close()

    # assign filter's best match
    for fi in filter_motifs:
      top_motif = sorted(filter_motifs[fi])[0][1]
      filter_names[fi] += '_%s' % motif_protein[top_motif]

  return np.array(filter_names)


################################################################################
# plot_target_corr
#
# Plot a clustered heatmap of correlations between filter activations and
# targets.
#
# Input
#  filter_outs:
#  filter_names:
#  target_names:
#  out_pdf:
################################################################################
def plot_target_corr(filter_outs, seq_targets, filter_names, target_names, out_pdf, seq_op='mean'):
  num_seqs = filter_outs.shape[0]
  num_targets = len(target_names)

  if seq_op == 'mean':
    filter_outs_seq = filter_outs.mean(axis=2)
  else:
    filter_outs_seq = filter_outs.max(axis=2)

  # std is sequence by filter.
  filter_seqs_std = filter_outs_seq.std(axis=0)
  filter_outs_seq = filter_outs_seq[:, filter_seqs_std > 0]
  filter_names_live = filter_names[filter_seqs_std > 0]

  filter_target_cors = np.zeros((len(filter_names_live), num_targets))
  for fi in range(len(filter_names_live)):
    for ti in range(num_targets):
      cor, p = spearmanr(filter_outs_seq[:, fi], seq_targets[:num_seqs, ti])
      filter_target_cors[fi, ti] = cor

  cor_df = pd.DataFrame(
      filter_target_cors, index=filter_names_live, columns=target_names)

  sns.set(font_scale=0.3)
  plt.figure()
  sns.clustermap(cor_df, cmap='BrBG', center=0, figsize=(8, 10))
  plt.savefig(out_pdf)
  plt.close()


################################################################################
# plot_filter_seq_heat
#
# Plot a clustered heatmap of filter activations in
#
# Input
#  param_matrix: np.array of the filter's parameter matrix
#  out_pdf:
################################################################################
def plot_filter_seq_heat(filter_outs, out_pdf, whiten=True, drop_dead=True):
  # compute filter output means per sequence
  filter_seqs = filter_outs.mean(axis=2)

  # whiten
  if whiten:
    filter_seqs = preprocessing.scale(filter_seqs)

  # transpose
  filter_seqs = np.transpose(filter_seqs)

  if drop_dead:
    filter_stds = filter_seqs.std(axis=1)
    filter_seqs = filter_seqs[filter_stds > 0]

  # downsample sequences
  seqs_i = np.random.randint(0, filter_seqs.shape[1], 500)

  hmin = np.percentile(filter_seqs[:, seqs_i], 0.1)
  hmax = np.percentile(filter_seqs[:, seqs_i], 99.9)

  sns.set(font_scale=0.3)

  plt.figure()
  sns.clustermap(
      filter_seqs[:, seqs_i],
      row_cluster=True,
      col_cluster=True,
      linewidths=0,
      xticklabels=False,
      vmin=hmin,
      vmax=hmax)
  plt.savefig(out_pdf)
  #out_png = out_pdf[:-2] + 'ng'
  #plt.savefig(out_png, dpi=300)
  plt.close()


################################################################################
# plot_filter_seq_heat
#
# Plot a clustered heatmap of filter activations in sequence segments.
#
# Mean doesn't work well for the smaller segments for some reason, but taking
# the max looks OK. Still, similar motifs don't cluster quite as well as you
# might expect.
#
# Input
#  filter_outs
################################################################################
def plot_filter_seg_heat(filter_outs, out_pdf, whiten=True, drop_dead=True):
  b = filter_outs.shape[0]
  f = filter_outs.shape[1]
  l = filter_outs.shape[2]

  s = 5
  while l / float(s) - (l / s) > 0:
    s += 1
  print('%d segments of length %d' % (s, l / s))

  # split into multiple segments
  filter_outs_seg = np.reshape(filter_outs, (b, f, s, l / s))

  # mean across the segments
  filter_outs_mean = filter_outs_seg.max(axis=3)

  # break each segment into a new instance
  filter_seqs = np.reshape(np.swapaxes(filter_outs_mean, 2, 1), (s * b, f))

  # whiten
  if whiten:
    filter_seqs = preprocessing.scale(filter_seqs)

  # transpose
  filter_seqs = np.transpose(filter_seqs)

  if drop_dead:
    filter_stds = filter_seqs.std(axis=1)
    filter_seqs = filter_seqs[filter_stds > 0]

  # downsample sequences
  seqs_i = np.random.randint(0, filter_seqs.shape[1], 500)

  hmin = np.percentile(filter_seqs[:, seqs_i], 0.1)
  hmax = np.percentile(filter_seqs[:, seqs_i], 99.9)

  sns.set(font_scale=0.3)
  if whiten:
    dist = 'euclidean'
  else:
    dist = 'cosine'

  plt.figure()
  sns.clustermap(
      filter_seqs[:, seqs_i],
      metric=dist,
      row_cluster=True,
      col_cluster=True,
      linewidths=0,
      xticklabels=False,
      vmin=hmin,
      vmax=hmax)
  plt.savefig(out_pdf)
  #out_png = out_pdf[:-2] + 'ng'
  #plt.savefig(out_png, dpi=300)
  plt.close()


################################################################################
# filter_motif
#
# Collapse the filter parameter matrix to a single DNA motif.
#
# Input
#  param_matrix: np.array of the filter's parameter matrix
#  out_pdf:
################################################################################
def filter_motif(param_matrix):
  nts = 'ACGT'

  motif_list = []
  for v in range(param_matrix.shape[1]):
    max_n = 0
    for n in range(1, 4):
      if param_matrix[n, v] > param_matrix[max_n, v]:
        max_n = n

    if param_matrix[max_n, v] > 0:
      motif_list.append(nts[max_n])
    else:
      motif_list.append('N')

  return ''.join(motif_list)


################################################################################
# filter_possum
#
# Write a Possum-style motif
#
# Input
#  param_matrix: np.array of the filter's parameter matrix
#  out_pdf:
################################################################################
def filter_possum(param_matrix, motif_id, possum_file, trim_filters=False, mult=200):
  # possible trim
  trim_start = 0
  trim_end = param_matrix.shape[1] - 1
  trim_t = 0.3
  if trim_filters:
    # trim PWM of uninformative prefix
    while trim_start < param_matrix.shape[1] and np.max(
        param_matrix[:, trim_start]) - np.min(
            param_matrix[:, trim_start]) < trim_t:
      trim_start += 1

    # trim PWM of uninformative suffix
    while trim_end >= 0 and np.max(param_matrix[:, trim_end]) - np.min(
        param_matrix[:, trim_end]) < trim_t:
      trim_end -= 1

  if trim_start < trim_end:
    possum_out = open(possum_file, 'w')
    print('BEGIN GROUP', file=possum_out)
    print('BEGIN FLOAT', file=possum_out)
    print('ID %s' % motif_id, file=possum_out)
    print('AP DNA', file=possum_out)
    print('LE %d' % (trim_end + 1 - trim_start), file=possum_out)
    for ci in range(trim_start, trim_end + 1):
      print(
          'MA %s' % ' '.join(['%.2f' % (mult * n)
                              for n in param_matrix[:, ci]]),
          file=possum_out)
    print('END', file=possum_out)
    print('END', file=possum_out)

    possum_out.close()


################################################################################
# plot_filter_heat
#
# Plot a heatmap of the filter's parameters.
#
# Input
#  param_matrix: np.array of the filter's parameter matrix
#  out_pdf:
################################################################################
def plot_filter_heat(param_matrix, out_pdf):
  param_range = abs(param_matrix).max()

  sns.set(font_scale=2)
  plt.figure(figsize=(param_matrix.shape[1], 4))
  sns.heatmap(
      param_matrix,
      cmap='PRGn',
      linewidths=0.2,
      vmin=-param_range,
      vmax=param_range)
  ax = plt.gca()
  ax.set_xticklabels(range(1, param_matrix.shape[1] + 1))
  ax.set_yticklabels('TGCA', rotation='horizontal')  # , size=10)
  plt.savefig(out_pdf)
  plt.close()


################################################################################
# plot_filter_logo
#
# Plot a weblogo of the filter's occurrences
#
# Input
#  param_matrix: np.array of the filter's parameter matrix
#  out_pdf:
################################################################################
def plot_filter_logo(filter_outs, filter_size, seqs, out_prefix, maxpct_t=None, raw_t=0):
  print(out_prefix)

  if maxpct_t is not None:
    all_outs = np.ravel(filter_outs)
    all_outs_mean = all_outs.mean()
    all_outs_norm = all_outs - all_outs_mean
    raw_t = maxpct_t * all_outs_norm.max() + all_outs_mean

  left_pad = (filter_size - 1) // 2
  right_pad = filter_size - left_pad

  # print fasta file of positive outputs
  filter_fasta_out = open('%s.fa' % out_prefix, 'w')
  filter_count = 0

  for i in range(filter_outs.shape[0]):
    for j in np.where(filter_outs[i] > raw_t)[0]:
      # construct kmer
      kmer = ''

      # determine boundaries, considering padding
      fstart = j - left_pad
      fend = fstart + filter_size

      # if it starts in left_pad
      if fstart < 0:
        kmer += 'N' * (-fstart)
        fstart = 0

      # add primary sequence
      kmer += seqs[i][fstart:fend]

      # if it ends in right_pad
      if fend > len(seqs[i]):
        kmer += 'N' * (fend - len(seqs[i]))

      # output
      print('>%d_%d' % (i, j), file=filter_fasta_out)
      print(kmer, file=filter_fasta_out)
      filter_count += 1

  filter_fasta_out.close()

  # make weblogo
  if filter_count > 0:
    weblogo_cmd = 'weblogo %s < %s.fa > %s.eps' % (weblogo_opts, out_prefix,
                                                   out_prefix)
    subprocess.call(weblogo_cmd, shell=True)


################################################################################
# plot_score_density
#
# Plot the score density and print to the stats table.
#
# Input
#  param_matrix: np.array of the filter's parameter matrix
#  out_pdf:
################################################################################
def plot_score_density(f_scores, out_pdf):
  sns.set(font_scale=1.3)
  plt.figure()
  sns.distplot(f_scores, kde=False)
  plt.xlabel('ReLU output')
  plt.savefig(out_pdf)
  plt.close()


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
  # pdb.runcall(main)
