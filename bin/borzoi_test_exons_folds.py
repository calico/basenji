#!/usr/bin/env python
# Copyright 2022 Calico LLC

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
import sys

from natsort import natsorted
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel
import matplotlib.pyplot as plt
import seaborn as sns

import slurm

"""
basenji_test_exons_folds.py

Test Basenji model replicates on alternative splicing via cassette exons,
given parameters and data.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <data1_dir> ...'
  parser = OptionParser(usage)
  parser.add_option('-a', '--alt', dest='alternative',
      default='two-sided', help='Statistical test alternative [Default: %default]')
  parser.add_option('-b', dest='exons_bed',
      default=None,
      help='Internal variable exons BED [Default: %default]')
  parser.add_option('-c', dest='crosses',
      default=1, type='int',
      help='Number of cross-fold rounds [Default:%default]')
  parser.add_option('-d', dest='dataset_i',
      default=None, type='int',
      help='Dataset index [Default:%default]')
  parser.add_option('--d_ref', dest='dataset_ref_i',
      default=None, type='int',
      help='Reference Dataset index [Default:%default]')
  parser.add_option('-e', dest='exon_end',
       default=2, type='int',
       help='Ignore exons this many exons at the 5p or 3p end [Default: %default]')
  parser.add_option('--env', dest='conda_env',
      default='tf210',
      help='Anaconda environment [Default: %default]')
  parser.add_option('-f', dest='fold_subset',
      default=None, type='int',
      help='Run a subset of folds [Default:%default]')
  parser.add_option('-g', dest='exons_gff',
      default='%s/genes/gencode41/gencode41_basic_nort_protein_exons.gff' % os.environ['HG38'])
  parser.add_option('--label_exp', dest='label_exp',
      default='Experiment', help='Experiment label [Default: %default]')
  parser.add_option('--label_ref', dest='label_ref',
      default='Reference', help='Reference label [Default: %default]')
  parser.add_option('-m', dest='metric',
  	  default='pearsonr', help='Train/test metric [Default: Pearsonr or AUPRC]')
  parser.add_option('--name', dest='name',
      default='teste', help='SLURM name prefix [Default: %default]')
  parser.add_option('-o', dest='exp_dir',
      default=None, help='Output experiment directory [Default: %default]')
  parser.add_option('-p', dest='out_stem',
      default=None, help='Output plot stem [Default: %default]')
  parser.add_option('-q', dest='queue',
      default='geforce')
  parser.add_option('-r', dest='ref_dir',
      default=None, help='Reference directory for statistical tests')
  parser.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Average forward and reverse complement predictions [Default: %default]')
  parser.add_option('--shifts', dest='shifts',
      default='0', type='str',
      help='Ensemble prediction shifts [Default: %default]')
  parser.add_option('--status', dest='status',
      default=False, action='store_true',
      help='Update metric status; do not run jobs [Default: %default]')
  parser.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')
  (options, args) = parser.parse_args()

  if len(args) < 2:
    parser.error('Must provide parameters file and data directory')
  else:
    params_file = args[0]
    data_dirs = [os.path.abspath(arg) for arg in args[1:]]

  # using -o for required argument for compatibility with the training script
  assert(options.exp_dir is not None)

  # read data parameters
  data_stats_file = '%s/statistics.json' % data_dirs[0]
  with open(data_stats_file) as data_stats_open:
    data_stats = json.load(data_stats_open)

  if options.dataset_i is None:
    head_i = 0
  else:
    head_i = options.dataset_i

  # count folds
  num_folds = len([dkey for dkey in data_stats if dkey.startswith('fold')])

  # subset folds
  if options.fold_subset is not None:
    num_folds = min(options.fold_subset, num_folds)

  if options.queue == 'standard':
    num_cpu = 4
    num_gpu = 0
  else:
    num_cpu = 2
    num_gpu = 1

  ################################################################
  # test best
  ################################################################
  jobs = []

  for ci in range(options.crosses):
    for fi in range(num_folds):
      it_dir = '%s/f%dc%d' % (options.exp_dir, fi, ci)

      if options.dataset_i is None:
        out_dir = '%s/teste' % it_dir
        model_file = '%s/train/model_best.h5' % it_dir
      else:
        out_dir = '%s/teste%d' % (it_dir, options.dataset_i)
        model_file = '%s/train/model%d_best.h5' % (it_dir, options.dataset_i)

      # check if done
      acc_file = '%s/acc.txt' % out_dir
      if os.path.isfile(acc_file):
        # print('%s already generated.' % acc_file)
        pass
      else:
        # basenji test
        cmd = '. /home/drk/anaconda3/etc/profile.d/conda.sh;'
        cmd += ' conda activate %s;' % options.conda_env
        cmd += ' time borzoi_test_exons.py'
        cmd += ' --head %d' % head_i
        cmd += ' -o %s' % out_dir
        if options.rc:
          cmd += ' --rc'
        if options.shifts:
          cmd += ' --shifts %s' % options.shifts
        if options.targets_file is not None:
          cmd += ' -t %s' % options.targets_file
        cmd += ' %s' % params_file
        cmd += ' %s' % model_file
        cmd += ' %s/data%d' % (it_dir, head_i)
        cmd += ' %s' % options.exons_gff

        name = '%s-f%dc%d' % (options.name, fi, ci)
        j = slurm.Job(cmd,
                      name=name,
                      out_file='%s.out'%out_dir,
                      err_file='%s.err'%out_dir,
                      sb_file='%s.sb'%out_dir,
                      queue=options.queue,
                      cpu=num_cpu, gpu=num_gpu,
                      mem=45000,
                      time='2-00:00:00')
        jobs.append(j)

  slurm.multi_run(jobs, verbose=True)

  if options.dataset_i is None:
    test_prefix = 'teste'
  else:
    test_prefix = 'teste%d' % options.dataset_i

  if options.dataset_ref_i is None:
    test_ref_prefix = 'teste'
  else:
    test_ref_prefix = 'teste%d' % options.dataset_ref_i


  ################################################################
  # compare best on test set
  ################################################################
  exp_glob_str = '%s/*/%s/acc.txt' % (options.exp_dir, test_prefix)
  exp_cors, exp_mean, exp_stdm = read_metrics(exp_glob_str, options.metric)

  if options.ref_dir is not None:
    ref_glob_str = '%s/*/%s/acc.txt' % (options.ref_dir, test_ref_prefix)
    ref_cors, ref_mean, ref_stdm = read_metrics(ref_glob_str, options.metric)

    mwp, tp = stat_tests(ref_cors, exp_cors, options.alternative)

  print('\nTest (%d reps):' % len(exp_cors))
  print('%12s %s: %.4f (%.4f)' % (options.label_exp, options.metric, exp_mean, exp_stdm))
  if options.ref_dir is not None:
    print('%12s %s: %.4f (%.4f)' % (options.label_ref, options.metric, ref_mean, ref_stdm))
    print('Mann-Whitney U p-value: %.3g' % mwp)
    print('T-test p-value: %.3g' % tp)

    if options.out_stem is not None:
      jointplot(ref_cors, exp_cors,
          '%s_test.pdf' % options.out_stem,
          options.label_ref, options.label_exp)


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
    acc_df = pd.read_csv(acc_file, sep='\t', index_col=0)
    rep_cors.append(acc_df.loc[:,metric].mean())
  cors_mean = np.mean(rep_cors)
  cors_stdm = np.std(rep_cors) / np.sqrt(len(rep_cors))

  return rep_cors, cors_mean, cors_stdm


def stat_tests(ref_cors, exp_cors, alternative):
  # hack for the common situtation where I have more reference
  # crosses than experiment crosses
  if len(ref_cors) == 2*len(exp_cors):
    ref_cors = [ref_cors[i] for i in range(len(ref_cors)) if i % 2 == 0]

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
