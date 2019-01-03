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
import random
import sys
import time

import h5py
import joblib
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyBigWig
from scipy.stats import spearmanr, poisson
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import tensorflow as tf

from basenji import batcher
from basenji import params
from basenji import plots
from basenji import seqnn
from basenji import tfrecord_batcher

"""
basenji_test.py

Test the accuracy of a trained model.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <model_file> <data_dir>'
  parser = OptionParser(usage)
  parser.add_option('--ai', dest='accuracy_indexes',
      help='Comma-separated list of target indexes to make accuracy scatter plots.')
  parser.add_option('--clip', dest='target_clip',
      default=None, type='float',
      help='Clip targets and predictions to a maximum value [Default: %default]')
  parser.add_option('-d', dest='down_sample',
      default=1, type='int',
      help='Down sample by taking uniformly spaced positions [Default: %default]')
  parser.add_option('-g', dest='genome_file',
      default=None,
      help='Chromosome length information [Default: %default]')
  parser.add_option('--mc', dest='mc_n',
      default=0, type='int',
      help='Monte carlo test iterations [Default: %default]')
  parser.add_option('--peak','--peaks', dest='peaks',
      default=False, action='store_true',
      help='Compute expensive peak accuracy [Default: %default]')
  parser.add_option('-o', dest='out_dir',
      default='test_out',
      help='Output directory for test statistics [Default: %default]')
  parser.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Average the fwd and rc predictions [Default: %default]')
  parser.add_option('--save', dest='save',
      default=False, action='store_true',
      help='Save targets and predictions numpy arrays [Default: %default]')
  parser.add_option('--shifts', dest='shifts',
      default='0',
      help='Ensemble prediction shifts [Default: %default]')
  parser.add_option('-t', dest='track_bed',
      help='BED file describing regions so we can output BigWig tracks')
  parser.add_option('--ti', dest='track_indexes',
      help='Comma-separated list of target indexes to output BigWig tracks')
  parser.add_option('--tfr', dest='tfr_pattern',
      default='test-*.tfr',
      help='TFR pattern string appended to data_dir [Default: %default]')
  parser.add_option('-w', dest='pool_width',
      default=1, type='int',
      help='Max pool width for regressing nt preds to peak calls [Default: %default]')
  (options, args) = parser.parse_args()

  if len(args) != 3:
    parser.error('Must provide parameters, model, and test data HDF5')
  else:
    params_file = args[0]
    model_file = args[1]
    data_dir = args[2]

  if not os.path.isdir(options.out_dir):
    os.mkdir(options.out_dir)

  # parse shifts to integers
  options.shifts = [int(shift) for shift in options.shifts.split(',')]

  # read targets
  targets_file = '%s/targets.txt' % data_dir
  targets_df = pd.read_table(targets_file)

  # read model parameters
  job = params.read_job_params(params_file)

  # construct data ops
  tfr_pattern_path = '%s/tfrecords/%s' % (data_dir, options.tfr_pattern)
  data_ops, test_init_op = make_data_ops(job, tfr_pattern_path)

  # initialize model
  model = seqnn.SeqNN()
  model.build_from_data_ops(job, data_ops,
                            ensemble_rc=options.rc,
                            ensemble_shifts=options.shifts)

  # initialize saver
  saver = tf.train.Saver()

  with tf.Session() as sess:
    # start queue runners
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(coord=coord)

    # load variables into session
    saver.restore(sess, model_file)

    # test
    t0 = time.time()
    sess.run(test_init_op)
    test_acc = model.test_tfr(sess)
    print('SeqNN test: %ds' % (time.time() - t0))

    test_preds = test_acc.preds
    test_targets = test_acc.targets

    if options.save:
      np.save('%s/preds.npy' % options.out_dir, test_acc.preds)
      np.save('%s/targets.npy' % options.out_dir, test_acc.targets)

    # compute stats
    t0 = time.time()
    test_r2 = test_acc.r2(clip=options.target_clip)
    # test_log_r2 = test_acc.r2(log=True, clip=options.target_clip)
    test_pcor = test_acc.pearsonr(clip=options.target_clip)
    test_log_pcor = test_acc.pearsonr(log=True, clip=options.target_clip)
    #test_scor = test_acc.spearmanr()  # too slow; mostly driven by low values
    print('Compute stats: %ds' % (time.time()-t0))

    # print
    print('Test Loss:         %7.5f' % test_acc.loss)
    print('Test R2:           %7.5f' % test_r2.mean())
    # print('Test log R2:       %7.5f' % test_log_r2.mean())
    print('Test PearsonR:     %7.5f' % test_pcor.mean())
    print('Test log PearsonR: %7.5f' % test_log_pcor.mean())
    # print('Test SpearmanR:    %7.5f' % test_scor.mean())

    acc_out = open('%s/acc.txt' % options.out_dir, 'w')
    for ti in range(len(test_r2)):
      print(
        '%4d  %7.5f  %.5f  %.5f  %.5f  %s' %
        (ti, test_acc.target_losses[ti], test_r2[ti], test_pcor[ti],
          test_log_pcor[ti], targets_df.description.iloc[ti]), file=acc_out)
    acc_out.close()

    # print normalization factors
    target_means = test_preds.mean(axis=(0,1), dtype='float64')
    target_means_median = np.median(target_means)
    target_means /= target_means_median
    norm_out = open('%s/normalization.txt' % options.out_dir, 'w')
    print('\n'.join([str(tu) for tu in target_means]), file=norm_out)
    norm_out.close()

    # clean up
    del test_acc

  #######################################################
  # peak call accuracy

  if options.peaks:
    # sample every few bins to decrease correlations
    ds_indexes = np.arange(0, test_preds.shape[1], 8)
    # ds_indexes_preds = np.arange(0, test_preds.shape[1], 8)
    # ds_indexes_targets = ds_indexes_preds + (model.hp.batch_buffer // model.hp.target_pool)

    aurocs = []
    auprcs = []

    peaks_out = open('%s/peaks.txt' % options.out_dir, 'w')
    for ti in range(test_targets.shape[2]):
      test_targets_ti = test_targets[:, :, ti]

      # subset and flatten
      test_targets_ti_flat = test_targets_ti[:, ds_indexes].flatten(
      ).astype('float32')
      test_preds_ti_flat = test_preds[:, ds_indexes, ti].flatten().astype(
          'float32')

      # call peaks
      test_targets_ti_lambda = np.mean(test_targets_ti_flat)
      test_targets_pvals = 1 - poisson.cdf(
          np.round(test_targets_ti_flat) - 1, mu=test_targets_ti_lambda)
      test_targets_qvals = np.array(ben_hoch(test_targets_pvals))
      test_targets_peaks = test_targets_qvals < 0.01

      if test_targets_peaks.sum() == 0:
        aurocs.append(0.5)
        auprcs.append(0)

      else:
        # compute prediction accuracy
        aurocs.append(roc_auc_score(test_targets_peaks, test_preds_ti_flat))
        auprcs.append(
            average_precision_score(test_targets_peaks, test_preds_ti_flat))

      print('%4d  %6d  %.5f  %.5f' % (ti, test_targets_peaks.sum(),
                                      aurocs[-1], auprcs[-1]),
                                      file=peaks_out)

    peaks_out.close()

    print('Test AUROC:     %7.5f' % np.mean(aurocs))
    print('Test AUPRC:     %7.5f' % np.mean(auprcs))

  #######################################################
  # BigWig tracks

  # NOTE: THESE ASSUME THERE WAS NO DOWN-SAMPLING ABOVE

  # print bigwig tracks for visualization
  if options.track_bed:
    if options.genome_file is None:
      parser.error('Must provide genome file in order to print valid BigWigs')

    if not os.path.isdir('%s/tracks' % options.out_dir):
      os.mkdir('%s/tracks' % options.out_dir)

    track_indexes = range(test_preds.shape[2])
    if options.track_indexes:
      track_indexes = [int(ti) for ti in options.track_indexes.split(',')]

    bed_set = 'test'
    if options.valid:
      bed_set = 'valid'

    for ti in track_indexes:
      test_targets_ti = test_targets[:, :, ti]

      # make true targets bigwig
      bw_file = '%s/tracks/t%d_true.bw' % (options.out_dir, ti)
      bigwig_write(
          bw_file,
          test_targets_ti,
          options.track_bed,
          options.genome_file,
          bed_set=bed_set)

      # make predictions bigwig
      bw_file = '%s/tracks/t%d_preds.bw' % (options.out_dir, ti)
      bigwig_write(
          bw_file,
          test_preds[:, :, ti],
          options.track_bed,
          options.genome_file,
          model.hp.batch_buffer,
          bed_set=bed_set)

    # make NA bigwig
    # bw_file = '%s/tracks/na.bw' % options.out_dir
    # bigwig_write(
    #     bw_file,
    #     test_na,
    #     options.track_bed,
    #     options.genome_file,
    #     bed_set=bed_set)

  #######################################################
  # accuracy plots

  if options.accuracy_indexes is not None:
    accuracy_indexes = [int(ti) for ti in options.accuracy_indexes.split(',')]

    if not os.path.isdir('%s/scatter' % options.out_dir):
      os.mkdir('%s/scatter' % options.out_dir)

    if not os.path.isdir('%s/violin' % options.out_dir):
      os.mkdir('%s/violin' % options.out_dir)

    if not os.path.isdir('%s/roc' % options.out_dir):
      os.mkdir('%s/roc' % options.out_dir)

    if not os.path.isdir('%s/pr' % options.out_dir):
      os.mkdir('%s/pr' % options.out_dir)

    for ti in accuracy_indexes:
      test_targets_ti = test_targets[:, :, ti]

      ############################################
      # scatter

      # sample every few bins (adjust to plot the # points I want)
      ds_indexes = np.arange(0, test_preds.shape[1], 8)
      # ds_indexes_preds = np.arange(0, test_preds.shape[1], 8)
      # ds_indexes_targets = ds_indexes_preds + (
      #     model.hp.batch_buffer // model.hp.target_pool)

      # subset and flatten
      test_targets_ti_flat = test_targets_ti[:, ds_indexes].flatten(
      ).astype('float32')
      test_preds_ti_flat = test_preds[:, ds_indexes, ti].flatten().astype(
          'float32')

      # take log2
      test_targets_ti_log = np.log2(test_targets_ti_flat + 1)
      test_preds_ti_log = np.log2(test_preds_ti_flat + 1)

      # plot log2
      sns.set(font_scale=1.2, style='ticks')
      out_pdf = '%s/scatter/t%d.pdf' % (options.out_dir, ti)
      plots.regplot(
          test_targets_ti_log,
          test_preds_ti_log,
          out_pdf,
          poly_order=1,
          alpha=0.3,
          sample=500,
          figsize=(6, 6),
          x_label='log2 Experiment',
          y_label='log2 Prediction',
          table=True)

      ############################################
      # violin

      # call peaks
      test_targets_ti_lambda = np.mean(test_targets_ti_flat)
      test_targets_pvals = 1 - poisson.cdf(
          np.round(test_targets_ti_flat) - 1, mu=test_targets_ti_lambda)
      test_targets_qvals = np.array(ben_hoch(test_targets_pvals))
      test_targets_peaks = test_targets_qvals < 0.01
      test_targets_peaks_str = np.where(test_targets_peaks, 'Peak',
                                        'Background')

      # violin plot
      sns.set(font_scale=1.3, style='ticks')
      plt.figure()
      df = pd.DataFrame({
          'log2 Prediction': np.log2(test_preds_ti_flat + 1),
          'Experimental coverage status': test_targets_peaks_str
      })
      ax = sns.violinplot(
          x='Experimental coverage status', y='log2 Prediction', data=df)
      ax.grid(True, linestyle=':')
      plt.savefig('%s/violin/t%d.pdf' % (options.out_dir, ti))
      plt.close()

      # ROC
      plt.figure()
      fpr, tpr, _ = roc_curve(test_targets_peaks, test_preds_ti_flat)
      auroc = roc_auc_score(test_targets_peaks, test_preds_ti_flat)
      plt.plot(
          [0, 1], [0, 1], c='black', linewidth=1, linestyle='--', alpha=0.7)
      plt.plot(fpr, tpr, c='black')
      ax = plt.gca()
      ax.set_xlabel('False positive rate')
      ax.set_ylabel('True positive rate')
      ax.text(
          0.99, 0.02, 'AUROC %.3f' % auroc,
          horizontalalignment='right')  # , fontsize=14)
      ax.grid(True, linestyle=':')
      plt.savefig('%s/roc/t%d.pdf' % (options.out_dir, ti))
      plt.close()

      # PR
      plt.figure()
      prec, recall, _ = precision_recall_curve(test_targets_peaks,
                                               test_preds_ti_flat)
      auprc = average_precision_score(test_targets_peaks, test_preds_ti_flat)
      plt.axhline(
          y=test_targets_peaks.mean(),
          c='black',
          linewidth=1,
          linestyle='--',
          alpha=0.7)
      plt.plot(recall, prec, c='black')
      ax = plt.gca()
      ax.set_xlabel('Recall')
      ax.set_ylabel('Precision')
      ax.text(
          0.99, 0.95, 'AUPRC %.3f' % auprc,
          horizontalalignment='right')  # , fontsize=14)
      ax.grid(True, linestyle=':')
      plt.savefig('%s/pr/t%d.pdf' % (options.out_dir, ti))
      plt.close()


def ben_hoch(p_values):
  """ Convert the given p-values to q-values using Benjamini-Hochberg FDR. """
  m = len(p_values)

  # attach original indexes to p-values
  p_k = [(p_values[k], k) for k in range(m)]

  # sort by p-value
  p_k.sort()

  # compute q-value and attach original index to front
  k_q = [(p_k[i][1], p_k[i][0] * m // (i + 1)) for i in range(m)]

  # re-sort by original index
  k_q.sort()

  # drop original indexes
  q_values = [k_q[k][1] for k in range(m)]

  return q_values


def bigwig_open(bw_file, genome_file):
  """ Open the bigwig file for writing and write the header. """

  bw_out = pyBigWig.open(bw_file, 'w')

  chrom_sizes = []
  for line in open(genome_file):
    a = line.split()
    chrom_sizes.append((a[0], int(a[1])))

  bw_out.addHeader(chrom_sizes)

  return bw_out


def bigwig_write(bw_file,
                 signal_ti,
                 track_bed,
                 genome_file,
                 buffer=0,
                 bed_set='test'):
  """ Write a signal track to a BigWig file over the regions
         specified by track_bed.

    Args
     bw_file:     BigWig filename
     signal_ti:   Sequences X Length array for some target
     track_bed:   BED file specifying sequence coordinates
     genome_file: Chromosome lengths file
     buffer:      Length skipped on each side of the region.
    """

  bw_out = bigwig_open(bw_file, genome_file)

  si = 0
  bw_hash = {}

  # set entries
  for line in open(track_bed):
    a = line.split()
    if a[3] == bed_set:
      chrom = a[0]
      start = int(a[1])
      end = int(a[2])

      preds_pool = (end - start - 2 * buffer) // signal_ti.shape[1]

      bw_start = start + buffer
      for li in range(signal_ti.shape[1]):
        bw_end = bw_start + preds_pool
        bw_hash.setdefault((chrom,bw_start,bw_end),[]).append(signal_ti[si,li])
        bw_start = bw_end

      si += 1

  # average duplicates
  bw_entries = []
  for bw_key in bw_hash:
    bw_signal = np.mean(bw_hash[bw_key])
    bwe = tuple(list(bw_key)+[bw_signal])
    bw_entries.append(bwe)

  # sort entries
  bw_entries.sort()

  # add entries
  for line in open(genome_file):
    chrom = line.split()[0]

    bw_entries_chroms = [be[0] for be in bw_entries if be[0] == chrom]
    bw_entries_starts = [be[1] for be in bw_entries if be[0] == chrom]
    bw_entries_ends = [be[2] for be in bw_entries if be[0] == chrom]
    bw_entries_values = [float(be[3]) for be in bw_entries if be[0] == chrom]

    if len(bw_entries_chroms) > 0:
      bw_out.addEntries(
          bw_entries_chroms,
          bw_entries_starts,
          ends=bw_entries_ends,
          values=bw_entries_values)

  bw_out.close()


def make_data_ops(job, tfr_pattern):
  def make_dataset(pat, mode):
    return tfrecord_batcher.tfrecord_dataset(
        pat,
        job['batch_size'],
        job['seq_length'],
        job.get('seq_depth', 4),
        job['target_length'],
        job['num_targets'],
        mode=mode,
        repeat=False)

  test_dataset = make_dataset(tfr_pattern, mode=tf.estimator.ModeKeys.EVAL)

  iterator = tf.data.Iterator.from_structure(
      test_dataset.output_types, test_dataset.output_shapes)
  data_ops = iterator.get_next()

  test_init_op = iterator.make_initializer(test_dataset)

  return data_ops, test_init_op


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
