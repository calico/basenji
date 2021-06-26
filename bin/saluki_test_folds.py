#!/usr/bin/env python
# Copyright 2019 Calico LLC

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
from optparse import OptionParser, OptionGroup
import glob
import json
import os
import pdb
import shutil
import sys

from natsort import natsorted
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel
import matplotlib.pyplot as plt
import seaborn as sns

import slurm

"""
basenji_test_folds.py

Train Basenji model replicates using given parameters and data.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <ref_dir> <exp_dir> <data_dir>'
  parser = OptionParser(usage)
  parser.add_option('-a', '--alt', dest='alternative',
      default='two-sided', help='Statistical test alternative [Default: %default]')
  parser.add_option('-c', dest='crosses',
      default=1, type='int',
      help='Number of cross-fold rounds [Default:%default]')
  parser.add_option('--label_exp', dest='label_exp',
      default='Experiment', help='Experiment label [Default: %default]')
  parser.add_option('--label_ref', dest='label_ref',
      default='Reference', help='Reference label [Default: %default]')
  parser.add_option('-o', dest='out_stem',
      default=None, help='Output plot stem [Default: %default]')
  (options, args) = parser.parse_args()

  if len(args) < 3:
    parser.error('Must provide parameters file and data directory')
  else:
    ref_dir = args[0]
    exp_dir = args[1]
    data_dir = os.path.abspath(args[2])

  # read data parameters
  data_stats_file = '%s/statistics.json' % data_dir
  with open(data_stats_file) as data_stats_open:
    data_stats = json.load(data_stats_open)

  # count folds
  num_folds = len([dkey for dkey in data_stats if dkey.startswith('fold')])

  # read training metrics
  ref_folds = read_metrics(ref_dir, num_folds)
  exp_folds = read_metrics(exp_dir, num_folds)

  for metric in ref_folds:
    print('\n%s:' % metric)

    ref_mean = ref_folds[metric].mean()
    ref_stdm = ref_folds[metric].std() / np.sqrt(len(ref_folds[metric]))
    print('%12s: %.4f (%.4f)' % (options.label_ref, ref_mean, ref_stdm))

    exp_mean = exp_folds[metric].mean()
    exp_stdm = exp_folds[metric].std() / np.sqrt(len(exp_folds[metric]))
    print('%12s: %.4f (%.4f)' % (options.label_exp, exp_mean, exp_stdm))
  
    mwp, tp = stat_tests(ref_folds[metric], exp_folds[metric], options.alternative)
    print('Mann-Whitney U p-value: %.3g' % mwp)
    print('T-test p-value: %.3g' % tp)


def jointplot(ref_cors, exp_cors, out_pdf, label1, label2):
  vmin = min(np.min(ref_cors), np.min(exp_cors))
  vmax = max(np.max(ref_cors), np.max(exp_cors))
  vspan = vmax - vmin
  vbuf = vspan * 0.1
  vmin -= vbuf
  vmax += vbuf

  g = sns.jointplot(ref_cors, exp_cors, space=0)

  eps = 0.05
  g.ax_joint.text(1-eps, eps, 'Mean: %.4f' % np.mean(ref_cors),
    horizontalalignment='right', transform=g.ax_joint.transAxes)
  g.ax_joint.text(eps, 1-eps, 'Mean: %.4f' % np.mean(exp_cors),
    verticalalignment='top', transform=g.ax_joint.transAxes)

  g.ax_joint.plot([vmin,vmax], [vmin,vmax], linestyle='--', color='orange')
  g.ax_joint.set_xlabel(label1)
  g.ax_joint.set_ylabel(label2)
  
  plt.tight_layout(w_pad=0, h_pad=0)
  plt.savefig(out_pdf)


def read_metrics(train_dir, num_folds, metric_best='val_pearsonr'):
  metric_folds = {}

  for fi in range(num_folds):
    # read table
    train_file  = '%s/f%d/train.out' % (train_dir,fi)
    train_df = pd.read_csv(train_file, sep='\t')

    # choose best epoch
    best_epoch = np.argmax(train_df[metric_best])

    # save best epoch metrics
    for metric in train_df.columns:
        metric_folds.setdefault(metric,[]).append(train_df[metric].iloc[best_epoch])

  # arrays
  for metric in metric_folds:
    metric_folds[metric] = np.array( metric_folds[metric])

  return metric_folds


def stat_tests(ref_cors, exp_cors, alternative):
  _, mwp = wilcoxon(exp_cors, ref_cors, alternative=alternative)
  tt, tp = ttest_rel(exp_cors, ref_cors)

  if alternative == 'less':
    if tt <= 0:
      tp /= 2
    else:
      tp = 1 - (1-tp)/2
  elif alternative == 'greater':
    if tt >= 0:
      tp /= 2
    else:
      tp = 1 - (1-tp)/2

  return mwp, tp

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
