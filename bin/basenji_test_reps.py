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
  usage = 'usage: %prog [options] <ref_dir> <exp_dir> <params_file> <data_dir>'
  parser = OptionParser(usage)
  parser.add_option('-q', dest='queue',
      default='gtx1080ti')
  parser.add_option('--spec', dest='specificity',
      default=False, action='store_true',
      help='Test specificity [Default: %default]')
  parser.add_option('--train', dest='train',
      default=False, action='store_true',
      help='Test on the training set, too [Default: %default]')
  (options, args) = parser.parse_args()

  if len(args) != 4:
    parser.error('Must provide parameters file and data directory')
  else:
    ref_dir = args[0]
    exp_dir = args[1]
    params_file = args[2]
    data_dir = args[3]

  iterations = len(glob.glob('%s/*' % exp_dir))

  ################################################################
  # test check
  ################################################################
  jobs = []

  if options.train:
    for i in range(iterations):
      it_dir = '%s/%d' % (exp_dir, i)

      # check if done
      acc_file = '%s/test_train/acc.txt' % it_dir
      if os.path.isfile(acc_file):
        print('%s already generated.' % acc_file)
      else:
        # basenji test
        basenji_cmd = '. /home/drk/anaconda3/etc/profile.d/conda.sh;'
        basenji_cmd += ' conda activate tf1.15-gpu2;'
        basenji_cmd += ' /home/drk/code/basenji2/bin/basenji_test.py'
        basenji_cmd += ' -o %s/test_train' % it_dir
        basenji_cmd += ' --rc'
        basenji_cmd += ' --tfr "train-*.tfr"'
        basenji_cmd += ' %s' % params_file
        basenji_cmd += ' %s/train/model_check.h5' % it_dir
        basenji_cmd += ' %s' % data_dir

        basenji_job = slurm.Job(basenji_cmd,
                        name='test_train%d' % i,
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
  for i in range(iterations):
    it_dir = '%s/%d' % (exp_dir, i)

    # check if done
    acc_file = '%s/test/acc.txt' % it_dir
    if os.path.isfile(acc_file):
      print('%s already generated.' % acc_file)
    else:
      # basenji test
      basenji_cmd = '. /home/drk/anaconda3/etc/profile.d/conda.sh;'
      basenji_cmd += ' conda activate tf1.15-gpu2;'
      basenji_cmd += ' /home/drk/code/basenji2/bin/basenji_test.py'
      basenji_cmd += ' -o %s/test' % it_dir
      basenji_cmd += ' --rc --shifts "1,0,-1"'
      basenji_cmd += ' %s' % params_file
      basenji_cmd += ' %s/train/model_best.h5' % it_dir
      basenji_cmd += ' %s' % data_dir

      basenji_job = slurm.Job(basenji_cmd,
                      name='test_test%d' % i,
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
    for i in range(iterations):
      it_dir = '%s/%d' % (exp_dir, i)

      # check if done
      acc_file = '%s/test/acc.txt' % it_dir
      if os.path.isfile(acc_file):
        print('%s already generated.' % acc_file)
      else:
        # basenji test
        basenji_cmd = '. /home/drk/anaconda3/etc/profile.d/conda.sh;'
        basenji_cmd += ' conda activate tf1.15-gpu2;'
        basenji_cmd += ' /home/drk/code/basenji2/bin/basenji_test_specificity.py'
        basenji_cmd += ' -o %s/test_spec' % it_dir
        basenji_cmd += ' --rc --shifts "1,0,-1"'
        basenji_cmd += ' %s' % params_file
        basenji_cmd += ' %s/train/model_best.h5' % it_dir
        basenji_cmd += ' %s' % data_dir

        basenji_job = slurm.Job(basenji_cmd,
                        name='test_spec%d' % i,
                        out_file='%s/test_spec.out'%it_dir,
                        err_file='%s/test_spec.err'%it_dir,
                        queue=options.queue,
                        cpu=1,
                        gpu=1,
                        mem=23000,
                        time='4:00:00')
        jobs.append(basenji_job)

  slurm.multi_run(jobs, verbose=True)

  ################################################################
  # compare checkpoint on training set
  ################################################################
  if options.train:
    ref_cors = []
    for acc_file in glob.glob('%s/*/test_train/acc.txt' % ref_dir):
      acc_df = pd.read_csv(acc_file, sep='\t', index_col=0)
      ref_cors.append(acc_df.pearsonr.mean())

    exp_cors = []
    for acc_file in glob.glob('%s/*/test_train/acc.txt' % exp_dir):
      acc_df = pd.read_csv(acc_file, sep='\t', index_col=0)
      exp_cors.append(acc_df.pearsonr.mean())

    _, mwp = mannwhitneyu(ref_cors, exp_cors, alternative='two-sided')
    _, tp = ttest_ind(ref_cors, exp_cors)
    print('\nTrain:')
    print('Reference  PearsonR: %.4f (%.4f)' % (np.mean(ref_cors), np.std(ref_cors)))
    print('Experiment PearsonR: %.4f (%.4f)' % (np.mean(exp_cors), np.std(exp_cors)))
    print('Mann-Whitney U p-value: %.3g' % mwp)
    print('T-test p-value: %.3g' % tp)


  ################################################################
  # compare best on test set
  ################################################################
  ref_cors = []
  for acc_file in glob.glob('%s/*/test/acc.txt' % ref_dir):
    acc_df = pd.read_csv(acc_file, sep='\t', index_col=0)
    ref_cors.append(acc_df.pearsonr.mean())

  exp_cors = []
  for acc_file in glob.glob('%s/*/test/acc.txt' % exp_dir):
    acc_df = pd.read_csv(acc_file, sep='\t', index_col=0)
    exp_cors.append(acc_df.pearsonr.mean())

  _, mwp = mannwhitneyu(ref_cors, exp_cors, alternative='two-sided')
  _, tp = ttest_ind(ref_cors, exp_cors)
  print('\nTest:')
  print('Reference  PearsonR: %.4f (%.4f)' % (np.mean(ref_cors), np.std(ref_cors)))
  print('Experiment PearsonR: %.4f (%.4f)' % (np.mean(exp_cors), np.std(exp_cors)))
  print('Mann-Whitney U p-value: %.3g' % mwp)
  print('T-test p-value: %.3g' % tp)

  ################################################################
  # compare best on test set specificity
  ################################################################
  if options.specificity:
    ref_cors = []
    for acc_file in glob.glob('%s/*/test_spec/acc.txt' % ref_dir):
      acc_df = pd.read_csv(acc_file, sep='\t', index_col=0)
      ref_cors.append(acc_df.pearsonr.mean())

    exp_cors = []
    for acc_file in glob.glob('%s/*/test_spec/acc.txt' % exp_dir):
      acc_df = pd.read_csv(acc_file, sep='\t', index_col=0)
      exp_cors.append(acc_df.pearsonr.mean())

    _, mwp = mannwhitneyu(ref_cors, exp_cors, alternative='two-sided')
    _, tp = ttest_ind(ref_cors, exp_cors)
    print('\nTest:')
    print('Reference  PearsonR: %.4f (%.4f)' % (np.mean(ref_cors), np.std(ref_cors)))
    print('Experiment PearsonR: %.4f (%.4f)' % (np.mean(exp_cors), np.std(exp_cors)))
    print('Mann-Whitney U p-value: %.3g' % mwp)
    print('T-test p-value: %.3g' % tp)
    

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
