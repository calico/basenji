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
  usage = 'usage: %prog [options] <exp_dir> <params_file> <data_dir>'
  parser = OptionParser(usage)
  parser.add_option('-a', '--alt', dest='alternative',
      default='two-sided', help='Statistical test alternative [Default: %default]')
  parser.add_option('-c', dest='crosses',
      default=1, type='int',
      help='Number of cross-fold rounds [Default:%default]')
  parser.add_option('-e', dest='conda_env',
      default='tf2-gpu',
      help='Anaconda environment [Default: %default]')
  parser.add_option('--l1', dest='label1',
      default='Reference', help='Reference label [Default: %default]')
  parser.add_option('--l2', dest='label2',
      default='Experiment', help='Experiment label [Default: %default]')
  parser.add_option('--name', dest='name',
      default='test', help='SLURM name prefix [Default: %default]')
  parser.add_option('-o', dest='out_stem',
      default=None, help='Outplut plot stem [Default: %default]')
  parser.add_option('-q', dest='queue',
      default='gtx1080ti')
  parser.add_option('-r', dest='ref_dir',
      default=None, help='Reference directory for statistical tests')
  parser.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Average forward and reverse complement predictions [Default: %default]')
  parser.add_option('--shifts', dest='shifts',
      default='0', type='str',
      help='Ensemble prediction shifts [Default: %default]')
  parser.add_option('--spec', dest='specificity',
      default=False, action='store_true',
      help='Test specificity [Default: %default]')
  parser.add_option('--train', dest='train',
      default=False, action='store_true',
      help='Test on the training set, too [Default: %default]')
  (options, args) = parser.parse_args()

  if len(args) != 3:
    parser.error('Must provide parameters file and data directory')
  else:
    exp_dir = args[0]
    params_file = args[1]
    data_dir = args[2]

   # read data parameters
  data_stats_file = '%s/statistics.json' % data_dir
  with open(data_stats_file) as data_stats_open:
    data_stats = json.load(data_stats_open)

  # count folds
  num_folds = len([dkey for dkey in data_stats if dkey.startswith('fold')])
  
  ################################################################
  # test check
  ################################################################
  jobs = []

  if options.train:
    for ci in range(options.crosses):
      for fi in range(num_folds):
        it_dir = '%s/f%d_c%d' % (exp_dir, fi, ci)
        
        # check if done
        acc_file = '%s/test_train/acc.txt' % it_dir
        if os.path.isfile(acc_file):
          print('%s already generated.' % acc_file)
        else:
          # basenji test
          basenji_cmd = '. /home/drk/anaconda3/etc/profile.d/conda.sh;'
          basenji_cmd += ' conda activate %s;' % options.conda_env
          basenji_cmd += ' basenji_test.py'
          basenji_cmd += ' -o %s/test_train' % it_dir
          if options.rc:
            basenji_cmd += ' --rc'
          if options.shifts:
            basenji_cmd += ' --shifts %s' % options.shifts
          basenji_cmd += ' --split train'
          basenji_cmd += ' %s' % params_file
          basenji_cmd += ' %s/train/model_check.h5' % it_dir
          basenji_cmd += ' %s/data' % it_dir

          name = '%s-testtr-f%dc%d' % (options.name, fi, ci)
          basenji_job = slurm.Job(basenji_cmd,
                          name=name,
                          out_file='%s/test_train.out'%it_dir,
                          err_file='%s/test_train.err'%it_dir,
                          queue=options.queue,
                          cpu=1,
                          gpu=1,
                          mem=23000,
                          time='4:00:00')
          jobs.append(basenji_job)


  ################################################################
  # test best
  ################################################################
  for ci in range(options.crosses):
    for fi in range(num_folds):
      it_dir = '%s/f%d_c%d' % (exp_dir, fi, ci)

      # check if done
      acc_file = '%s/test/acc.txt' % it_dir
      if os.path.isfile(acc_file):
        print('%s already generated.' % acc_file)
      else:
        # basenji test
        basenji_cmd = '. /home/drk/anaconda3/etc/profile.d/conda.sh;'
        basenji_cmd += ' conda activate %s;' % options.conda_env
        basenji_cmd += ' basenji_test.py'
        basenji_cmd += ' -o %s/test' % it_dir
        if options.rc:
          basenji_cmd += ' --rc'
        if options.shifts:
          basenji_cmd += ' --shifts %s' % options.shifts
        basenji_cmd += ' %s' % params_file
        basenji_cmd += ' %s/train/model_best.h5' % it_dir
        basenji_cmd += ' %s/data' % it_dir

        name = '%s-test-f%dc%d' % (options.name, fi, ci)
        basenji_job = slurm.Job(basenji_cmd,
                        name=name,
                        out_file='%s/test.out'%it_dir,
                        err_file='%s/test.err'%it_dir,
                        queue=options.queue,
                        cpu=1,
                        gpu=1,
                        mem=23000,
                        time='4:00:00')
        jobs.append(basenji_job)

  ################################################################
  # test best specificity
  ################################################################
  if options.specificity:
    for ci in range(options.crosses):
      for fi in range(num_folds):
        it_dir = '%s/f%d_c%d' % (exp_dir, fi, ci)

        # check if done
        acc_file = '%s/test_spec/acc.txt' % it_dir
        if os.path.isfile(acc_file):
          print('%s already generated.' % acc_file)
        else:
          # basenji test
          basenji_cmd = '. /home/drk/anaconda3/etc/profile.d/conda.sh;'
          basenji_cmd += ' conda activate %s;' % options.conda_env
          basenji_cmd += ' basenji_test_specificity.py'
          basenji_cmd += ' -o %s/test_spec' % it_dir
          if options.rc:
            basenji_cmd += ' --rc'
          if options.shifts:
            basenji_cmd += ' --shifts %s' % options.shifts
          basenji_cmd += ' %s' % params_file
          basenji_cmd += ' %s/train/model_best.h5' % it_dir
          basenji_cmd += ' %s/data' % it_dir

          name = '%s-spec-f%dc%d' % (options.name, fi, ci)
          basenji_job = slurm.Job(basenji_cmd,
                          name=name,
                          out_file='%s/test_spec.out'%it_dir,
                          err_file='%s/test_spec.err'%it_dir,
                          queue=options.queue,
                          cpu=1,
                          gpu=1,
                          mem=60000,
                          time='6:00:00')
          jobs.append(basenji_job)

  slurm.multi_run(jobs, verbose=True)


  if options.ref_dir is not None:
    # classification or regression
    with open('%s/f0_c0/test/acc.txt' % exp_dir) as test0_open:
      header = test0_open.readline().split()
      if 'pearsonr' in header:
        metric = 'pearsonr'
      else:
        metric = 'auprc'

    ################################################################
    # compare checkpoint on training set
    ################################################################
    if options.train:
      ref_glob_str = '%s/*/test_train/acc.txt' % options.ref_dir
      ref_cors, ref_mean, ref_stdm = read_metrics(ref_glob_str, metric)

      exp_glob_str = '%s/*/test_train/acc.txt' % exp_dir
      exp_cors, exp_mean, exp_stdm = read_metrics(exp_glob_str, metric)

      mwp, tp = stat_tests(ref_cors, exp_cors, options.alternative)

      print('\nTrain:')
      print('%12s %s: %.4f (%.4f)' % (options.label1, metric, ref_mean, ref_stdm))
      print('%12s %s: %.4f (%.4f)' % (options.label2, metric, exp_mean, exp_stdm))
      print('Mann-Whitney U p-value: %.3g' % mwp)
      print('T-test p-value: %.3g' % tp)

      if options.out_stem is not None:
        jointplot(ref_cors, exp_cors,
          '%s_train.pdf' % options.out_stem,
          options.label1, options.label2)


    ################################################################
    # compare best on test set
    ################################################################
    ref_glob_str = '%s/*/test/acc.txt' % options.ref_dir
    ref_cors, ref_mean, ref_stdm = read_metrics(ref_glob_str, metric)

    exp_glob_str = '%s/*/test/acc.txt' % exp_dir
    exp_cors, exp_mean, exp_stdm = read_metrics(exp_glob_str, metric)

    mwp, tp = stat_tests(ref_cors, exp_cors, options.alternative)

    print('\nTest:')
    print('%12s %s: %.4f (%.4f)' % (options.label1, metric, ref_mean, ref_stdm))
    print('%12s %s: %.4f (%.4f)' % (options.label2, metric, exp_mean, exp_stdm))
    print('Mann-Whitney U p-value: %.3g' % mwp)
    print('T-test p-value: %.3g' % tp)

    if options.out_stem is not None:
      jointplot(ref_cors, exp_cors,
          '%s_test.pdf' % options.out_stem,
          options.label1, options.label2)

    ################################################################
    # compare best on test set specificity
    ################################################################
    if options.specificity:
      ref_glob_str = '%s/*/test_spec/acc.txt' % options.ref_dir
      ref_cors, ref_mean, ref_stdm = read_metrics(ref_glob_str, metric)

      exp_glob_str = '%s/*/test_spec/acc.txt' % exp_dir
      exp_cors, exp_mean, exp_stdm = read_metrics(exp_glob_str, metric)

      mwp, tp = stat_tests(ref_cors, exp_cors, options.alternative)

      print('\nSpecificity:')
      print('%12s %s: %.4f (%.4f)' % (options.label1, metric, ref_mean, ref_stdm))
      print('%12s %s: %.4f (%.4f)' % (options.label2, metric, exp_mean, exp_stdm))
      print('Mann-Whitney U p-value: %.3g' % mwp)
      print('T-test p-value: %.3g' % tp)

      if options.out_stem is not None:
        jointplot(ref_cors, exp_cors,
          '%s_spec.pdf' % options.out_stem,
          options.label1, options.label2)
    

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


def read_metrics(acc_glob_str, metric='pearsonr'):
  rep_cors = []
  acc_files = natsorted(glob.glob(acc_glob_str))
  for acc_file in acc_files:
    try:
      # tf2 version
      acc_df = pd.read_csv(acc_file, sep='\t', index_col=0)
      rep_cors.append(acc_df.loc[:,metric].mean())

    except:
      # tf1 version
      cors = []
      for line in open(acc_file):
        a = line.split()
        cors.append(float(a[3]))
      rep_cors.append(np.mean(cors))
  
  cors_mean = np.mean(rep_cors)
  cors_stdm = np.std(rep_cors) / np.sqrt(len(rep_cors))

  return rep_cors, cors_mean, cors_stdm


def stat_tests(ref_cors, exp_cors, alternative):
  _, mwp = wilcoxon(ref_cors, exp_cors, alternative=alternative)
  tt, tp = ttest_rel(ref_cors, exp_cors)

  if alternative == 'less':
    if tt > 0:
      tp = 1 - (1-tp)/2
    else:
      tp /= 2
  elif alternative == 'greater':
    if tt <= 0:
      tp /= 2
    else:
      tp = 1 - (1-tp)/2

  return mwp, tp

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
