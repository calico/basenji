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
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind

import slurm

"""
basenji_test_reps.py

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
  parser.add_option('-d', dest='dataset_i',
      default=None, type='int',
      help='Dataset index [Default:%default]')
  parser.add_option('-e', dest='conda_env',
      default='tf2-gpu',
      help='Anaconda environment [Default: %default]')
  parser.add_option('--name', dest='name',
      default='test', help='SLURM name prefix [Default: %default]')
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
    data_dirs = [os.path.abspath(arg) for arg in args[2:]]

  if options.dataset_i is None:
    head_i = 0
  else:
    head_i = options.dataset_i

  iterations = len(glob.glob('%s/*' % exp_dir))

  ################################################################
  # test check
  ################################################################
  jobs = []

  if options.train:
    for i in range(iterations):
      it_dir = '%s/%d' % (exp_dir, i)

      if options.dataset_i is None:
        out_dir = '%s/test_train' % it_dir
        model_file = '%s/train/model_check.h5' % it_dir
        data_dir = data_dirs[0]
      else:
        out_dir = '%s/test%d_train' % (it_dir, options.dataset_i)
        model_file = '%s/train/model%d_check.h5' % (it_dir, options.dataset_i)
        data_dir = data_dirs[options.dataset_i]
    
      # check if done
      acc_file = '%s/acc.txt' % out_dir
      if os.path.isfile(acc_file):
        print('%s already generated.' % acc_file)
      else:            
        cmd = '. /home/drk/anaconda3/etc/profile.d/conda.sh;'
        cmd += ' conda activate %s;' % options.conda_env
        cmd += ' basenji_test.py'
        cmd += ' --head %d' % head_i
        cmd += ' -o %s' % out_dir
        if options.rc:
          cmd += ' --rc'
        if options.shifts:
          cmd += ' --shifts %s' % options.shifts
        cmd += ' --split train'
        cmd += ' %s' % params_file
        cmd += ' %s' % model_file
        cmd += ' %s' % data_dir

        name = '%s-testtr%d' % (options.name, i)
        j = slurm.Job(cmd,
                        name=name,
                        out_file='%s.out'%out_dir,
                        err_file='%s.err'%out_dir,
                        queue=options.queue,
                        cpu=1, gpu=1,
                        mem=23000,
                        time='4:00:00')
        jobs.append(j)


  ################################################################
  # test best
  ################################################################
  for i in range(iterations):
    it_dir = '%s/%d' % (exp_dir, i)

    if options.dataset_i is None:
      out_dir = '%s/test' % it_dir
      model_file = '%s/train/model_best.h5' % it_dir
      data_dir = data_dirs[0]
    else:
      out_dir = '%s/test%d' % (it_dir, options.dataset_i)
      model_file = '%s/train/model%d_best.h5' % (it_dir, options.dataset_i)
      data_dir = data_dirs[options.dataset_i]

    # check if done
    acc_file = '%s/acc.txt' % out_dir
    if os.path.isfile(acc_file):
      print('%s already generated.' % acc_file)
    else:
      # basenji test
      cmd = '. /home/drk/anaconda3/etc/profile.d/conda.sh;'
      cmd += ' conda activate %s;' % options.conda_env
      cmd += ' basenji_test.py'
      cmd += ' --head %d' % head_i
      cmd += ' -o %s' % out_dir
      if options.rc:
        cmd += ' --rc'
      if options.shifts:
        cmd += ' --shifts %s' % options.shifts
      cmd += ' %s' % params_file
      cmd += ' %s' % model_file
      cmd += ' %s' % data_dir

      name = '%s-test%d' % (options.name, i)
      j = slurm.Job(cmd,
                    name=name,
                    out_file='%s.out'%out_dir,
                    err_file='%s.err'%out_dir,
                    queue=options.queue,
                    cpu=1, gpu=1,
                    mem=23000,
                    time='4:00:00')
      jobs.append(j)


  ################################################################
  # test best specificity
  ################################################################
  if options.specificity:
    for i in range(iterations):
      it_dir = '%s/%d' % (exp_dir, i)

      if options.dataset_i is None:
        out_dir = '%s/test_spec' % it_dir
        model_file = '%s/train/model_best.h5' % it_dir
        data_dir = data_dirs[0]
      else:
        out_dir = '%s/test%d_spec' % (it_dir, di)
        model_file = '%s/train/model%d_best.h5' % (it_dir, di)
        data_dir = data_dirs[options.dataset_i]

      # check if done
      acc_file = '%s/acc.txt' % out_dir
      if os.path.isfile(acc_file):
        print('%s already generated.' % acc_file)
      else:
        # basenji test
        cmd = '. /home/drk/anaconda3/etc/profile.d/conda.sh;'
        cmd += ' conda activate %s;' % options.conda_env
        cmd += ' basenji_test_specificity.py'
        cmd += ' --head %d' % head_i
        cmd += ' -o %s' % out_dir
        if options.rc:
          cmd += ' --rc'
        if options.shifts:
          cmd += ' --shifts %s' % options.shifts
        cmd += ' %s' % params_file
        cmd += ' %s' % model_file
        cmd += ' %s' % data_dir

        name = '%s-spec%d' % (options.name, i)
        j = slurm.Job(cmd,
                      name=name,
                      out_file='%s.out'%out_dir,
                      err_file='%s.err'%out_dir,
                      queue=options.queue,
                      cpu=1, gpu=1,
                      mem=75000,
                      time='6:00:00')
        jobs.append(j)

  slurm.multi_run(jobs, verbose=True)


  if options.ref_dir is not None:
    ################################################################
    # compare checkpoint on training set
    ################################################################
    if options.train:
      ref_glob_str = '%s/*/test_train/acc.txt' % options.ref_dir
      ref_cors, ref_mean, ref_stdm = read_cors(ref_glob_str)

      exp_glob_str = '%s/*/test_train/acc.txt' % exp_dir
      exp_cors, exp_mean, exp_stdm = read_cors(exp_glob_str)

      mwp, tp = stat_tests(ref_cors, exp_cors, options.alternative)

      print('\nTrain:')
      print('Reference  PearsonR: %.4f (%.4f)' % (ref_mean, ref_stdm))
      print('Experiment PearsonR: %.4f (%.4f)' % (exp_mean, exp_stdm))
      print('Mann-Whitney U p-value: %.3g' % mwp)
      print('T-test p-value: %.3g' % tp)


    ################################################################
    # compare best on test set
    ################################################################
    ref_glob_str = '%s/*/test/acc.txt' % options.ref_dir
    ref_cors, ref_mean, ref_stdm = read_cors(ref_glob_str)

    exp_glob_str = '%s/*/test/acc.txt' % exp_dir
    exp_cors, exp_mean, exp_stdm = read_cors(exp_glob_str)

    mwp, tp = stat_tests(ref_cors, exp_cors, options.alternative)

    print('\nTest:')
    print('Reference  PearsonR: %.4f (%.4f)' % (ref_mean, ref_stdm))
    print('Experiment PearsonR: %.4f (%.4f)' % (exp_mean, exp_stdm))
    print('Mann-Whitney U p-value: %.3g' % mwp)
    print('T-test p-value: %.3g' % tp)

    ################################################################
    # compare best on test set specificity
    ################################################################
    if options.specificity:
      ref_glob_str = '%s/*/test_spec/acc.txt' % options.ref_dir
      ref_cors, ref_mean, ref_stdm = read_cors(ref_glob_str)

      exp_glob_str = '%s/*/test_spec/acc.txt' % exp_dir
      exp_cors, exp_mean, exp_stdm = read_cors(exp_glob_str)

      mwp, tp = stat_tests(ref_cors, exp_cors, options.alternative)

      print('\nSpecificity:')
      print('Reference  PearsonR: %.4f (%.4f)' % (ref_mean, ref_stdm))
      print('Experiment PearsonR: %.4f (%.4f)' % (exp_mean, exp_stdm))
      print('Mann-Whitney U p-value: %.3g' % mwp)
      print('T-test p-value: %.3g' % tp)
    
def read_cors(acc_glob_str):
  rep_cors = []
  for acc_file in glob.glob(acc_glob_str):
    try:
      acc_df = pd.read_csv(acc_file, sep='\t', index_col=0)
      rep_cors.append(acc_df.pearsonr.mean())
    except:
      #read tf1 version
      cors = []
      for line in open(acc_file):
        a = line.split()
        cors.append(float(a[3]))
      rep_cors.append(np.mean(cors))
  
  cors_mean = np.mean(rep_cors)
  cors_stdm = np.std(rep_cors) / np.sqrt(len(rep_cors))

  return rep_cors, cors_mean, cors_stdm


def stat_tests(ref_cors, exp_cors, alternative):
  _, mwp = mannwhitneyu(ref_cors, exp_cors, alternative=alternative)
  tt, tp = ttest_ind(ref_cors, exp_cors)

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
