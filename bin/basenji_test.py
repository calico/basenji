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
import pdb
import sys
import time

import h5py
import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
import tensorflow as tf

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import seaborn as sns

from basenji import dataset
from basenji import plots
from basenji import seqnn
from basenji import trainer

if tf.__version__[0] == '1':
  tf.compat.v1.enable_eager_execution()

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
  parser.add_option('--head', dest='head_i',
      default=0, type='int',
      help='Parameters head to test [Default: %default]')
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
  parser.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')
  parser.add_option('--split', dest='split_label',
      default='test',
      help='Dataset split label for eg TFR pattern [Default: %default]')
  parser.add_option('--tfr', dest='tfr_pattern',
      default=None,
      help='TFR pattern string appended to data_dir/tfrecords for subsetting [Default: %default]')
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

  #######################################################
  # inputs

  # read targets
  if options.targets_file is None:
    options.targets_file = '%s/targets.txt' % data_dir
  targets_df = pd.read_csv(options.targets_file, index_col=0, sep='\t')

  # read model parameters
  with open(params_file) as params_open:
    params = json.load(params_open)
  params_model = params['model']
  params_train = params['train']
  
  # construct eval data
  eval_data = dataset.SeqDataset(data_dir,
    split_label=options.split_label,
    batch_size=params_train['batch_size'],
    mode='eval',
    tfr_pattern=options.tfr_pattern)

  # initialize model
  seqnn_model = seqnn.SeqNN(params_model)
  seqnn_model.restore(model_file, options.head_i)
  seqnn_model.build_ensemble(options.rc, options.shifts)

  #######################################################
  # evaluate

  loss_label = params_train.get('loss', 'poisson').lower()
  spec_weight = params_train.get('spec_weight', 1)
  loss_fn = trainer.parse_loss(loss_label, spec_weight=spec_weight)

  # evaluate
  test_loss, test_metric1, test_metric2 = seqnn_model.evaluate(eval_data, loss=loss_fn)

  # print summary statistics
  print('\nTest Loss:         %7.5f' % test_loss)

  if loss_label == 'bce':
    print('Test AUROC:        %7.5f' % test_metric1.mean())
    print('Test AUPRC:        %7.5f' % test_metric2.mean())

    # write target-level statistics
    targets_acc_df = pd.DataFrame({
      'index': targets_df.index,
      'auroc': test_metric1,
      'auprc': test_metric2,
      'identifier': targets_df.identifier,
      'description': targets_df.description
      })

  else:
    print('Test PearsonR:     %7.5f' % test_metric1.mean())
    print('Test R2:           %7.5f' % test_metric2.mean())

    # write target-level statistics
    targets_acc_df = pd.DataFrame({
      'index': targets_df.index,
      'pearsonr': test_metric1,
      'r2': test_metric2,
      'identifier': targets_df.identifier,
      'description': targets_df.description
      })

  targets_acc_df.to_csv('%s/acc.txt'%options.out_dir, sep='\t',
                        index=False, float_format='%.5f')

  #######################################################
  # predict?

  if options.save or options.peaks or options.accuracy_indexes is not None:
    # compute predictions
    test_preds = seqnn_model.predict(eval_data).astype('float16')

    # read targets
    test_targets = eval_data.numpy(return_inputs=False)

  if options.save:
    preds_h5 = h5py.File('%s/preds.h5' % options.out_dir, 'w')
    preds_h5.create_dataset('preds', data=test_preds)
    preds_h5.close()
    targets_h5 = h5py.File('%s/targets.h5' % options.out_dir, 'w')
    targets_h5.create_dataset('targets', data=test_targets)
    targets_h5.close()


  #######################################################
  # peak call accuracy

  if options.peaks:
    peaks_out_file = '%s/peaks.txt' % options.out_dir
    test_peaks(test_preds, test_targets, peaks_out_file)


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


def test_peaks(test_preds, test_targets, peaks_out_file):
    # sample every few bins to decrease correlations
    ds_indexes = np.arange(0, test_preds.shape[1], 8)
    # ds_indexes_preds = np.arange(0, test_preds.shape[1], 8)
    # ds_indexes_targets = ds_indexes_preds + (model.hp.batch_buffer // model.hp.target_pool)

    aurocs = []
    auprcs = []

    peaks_out = open(peaks_out_file, 'w')
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


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
