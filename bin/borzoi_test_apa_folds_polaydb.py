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
import sys

from natsort import natsorted
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel
import matplotlib.pyplot as plt
import seaborn as sns

import slurm

"""
borzoi_test_apa_folds_polaydb.py

"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <data1_dir> ...'
  parser = OptionParser(usage)
  parser.add_option('-a', '--alt', dest='alternative',
      default='two-sided', help='Statistical test alternative [Default: %default]')
  parser.add_option('-c', dest='crosses',
      default=1, type='int',
      help='Number of cross-fold rounds [Default:%default]')
  parser.add_option('-d', dest='dataset_i',
      default=None, type='int',
      help='Dataset index [Default:%default]')
  parser.add_option('--d_ref', dest='dataset_ref_i',
      default=None, type='int',
      help='Reference Dataset index [Default:%default]')
  parser.add_option('-e', dest='conda_env',
      default='tf210',
      help='Anaconda environment [Default: %default]')
  parser.add_option('-f', dest='fold_subset',
      default=None, type='int',
      help='Run a subset of folds [Default:%default]')
  parser.add_option('-g', dest='apa_file',
      default='polyadb_human_v3.csv.gz')
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
        cmd += ' time borzoi_test_apa_polaydb.py'
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
        cmd += ' %s' % options.apa_file
        
        name = '%s-f%dc%d' % (options.name, fi, ci)
        j = slurm.Job(cmd,
                      name=name,
                      out_file='%s.out'%out_dir,
                      err_file='%s.err'%out_dir,
                      queue=options.queue,
                      cpu=num_cpu, gpu=num_gpu,
                      mem=45000,
                      time='2-00:00:00')
        jobs.append(j)

  slurm.multi_run(jobs, verbose=True)

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
