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
import shutil
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel

import subprocess

"""
basenji_gtex_folds.py

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
  parser.add_option('--name', dest='name',
      default='test', help='SLURM name prefix [Default: %default]')
  parser.add_option('-q', dest='queue',
      default='geforce',
      help='SLURM queue on which to run the jobs [Default: %default]')
  parser.add_option('-r', dest='ref_dir',
      default=None, help='Reference directory for statistical tests')
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
  # motifs
  ################################################################
  jobs = []

  for ci in range(options.crosses):
    for fi in range(num_folds):
      it_dir = '%s/f%dc%d' % (exp_dir, fi, ci)

      # check if done
      stats_file = '%s/gtex/Whole_Blood_class/stats.txt' % it_dir
      if os.path.isfile(stats_file):
        print('%s already generated.' % stats_file)
      else:
        basenji_cmd = ' basenji_bench_gtex.py'
        basenji_cmd += ' -o %s/gtex' % it_dir
        basenji_cmd += ' --rc'
        basenji_cmd += ' -q %s' % options.queue
        basenji_cmd += ' %s' % params_file
        basenji_cmd += ' %s/train/model_best.h5' % it_dir        

        print(basenji_cmd)
        with open('%s/gtex.out'%it_dir, 'w') as it_out:
          subprocess.call(basenji_cmd, shell=True,
              stdout=it_out, stderr=subprocess.STDOUT)

  ################################################################
  # compare
  ################################################################
  if options.ref_dir is not None:
    df_ref = read_stats(options.ref_dir, num_folds, options.crosses)
    df_exp = read_stats(exp_dir, num_folds, options.crosses)

    mwp, tp = stat_tests(df_ref.auroc, df_exp.auroc, options.alternative)

    print('\nTest:')
    print('Reference  AUROC: %.4f' % df_ref.auroc.mean())
    print('Experiment AUROC: %.4f' % df_exp.auroc.mean())
    print('Mann-Whitney U p-value: %.3g' % mwp)
    print('T-test p-value: %.3g' % tp)

    
def read_stats(exp_dir, num_folds, num_crosses):
  df_tissue = []
  df_fold = []
  df_cross = []
  df_auroc = []
  for ci in range(num_crosses):
    for fi in range(num_folds):
      it_dir = '%s/f%dc%d/gtex' % (exp_dir, fi, ci)

      tissue_stats_files = sorted(glob.glob('%s/*_class/stats.txt'%it_dir))
      for stats_file in tissue_stats_files:
        tissue = stats_file.split('/')[-2]
        tissue = tissue[:tissue.find('_class')]

        with open(stats_file,'r') as stats_open:
          auroc = float(stats_open.readline().split()[1])

        df_fold.append(fi)
        df_cross.append(ci)
        df_tissue.append(tissue)
        df_auroc.append(auroc)

  df = pd.DataFrame({
    'fold':df_fold,
    'cross':df_cross,
    'auroc':df_auroc,
    'tissue':df_tissue,
    })

  return df


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
