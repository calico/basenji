#!/usr/bin/env python
from optparse import OptionParser

import glob
import os
import shutil
import unittest

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind

import slurm

class TestTrain(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.params_file = 'train/params_x.json'
    cls.data_dir = 'train/data'
    cls.ref_dir = 'train/ref'
    cls.iterations = 4

    cls.basenji_path = '/home/drk/code/basenji2/bin'
    cls.conda_env = 'tf1.15-gpu'
    cls.queue = 'k80'

  def test_train(self):
    exp_dir = 'train/exp'
    """
    if os.path.isdir(exp_dir):
      shutil.rmtree(exp_dir)
    os.mkdir(exp_dir)

    ################################################################
    # train
    ################################################################
    jobs = []
    for i in range(self.iterations):
      it_dir = '%s/%d' % (exp_dir, i)
      os.mkdir(it_dir)

      # basenji train
      basenji_cmd = '. /home/drk/anaconda3/etc/profile.d/conda.sh;'
      basenji_cmd += ' conda activate %s;' % self.conda_env
      basenji_cmd += ' %s/basenji_train.py' % self.basenji_path
      basenji_cmd += ' -o %s/train' % it_dir
      basenji_cmd += ' %s' % self.params_file
      basenji_cmd += ' %s' % self.data_dir

      basenji_job = slurm.Job(basenji_cmd,
                      name='train%d' % i,
                      out_file='%s/train.out'%it_dir,
                      err_file='%s/train.err'%it_dir,
                      queue=self.queue,
                      cpu=1,
                      gpu=1,
                      mem=23000,
                      time='2-00:00:00')
      jobs.append(basenji_job)

    slurm.multi_run(jobs, verbose=True)

    ################################################################
    # test check
    ################################################################
    jobs = []
    for i in range(self.iterations):
      it_dir = '%s/%d' % (exp_dir, i)

      # basenji test
      basenji_cmd = '. /home/drk/anaconda3/etc/profile.d/conda.sh;'
      basenji_cmd += ' conda activate %s;' % self.conda_env
      basenji_cmd += ' %s/basenji_test.py' % self.basenji_path
      basenji_cmd += ' -o %s/test_train' % it_dir
      basenji_cmd += ' --tfr "train-*.tfr"'
      basenji_cmd += ' %s' % self.params_file
      basenji_cmd += ' %s/train/model_check.h5' % it_dir
      basenji_cmd += ' %s' % self.data_dir

      basenji_job = slurm.Job(basenji_cmd,
                      name='test%d' % i,
                      out_file='%s/test_train.out'%it_dir,
                      err_file='%s/test_train.err'%it_dir,
                      queue=self.queue,
                      cpu=1,
                      gpu=1,
                      mem=23000,
                      time='1:00:00')
      jobs.append(basenji_job)

    slurm.multi_run(jobs, verbose=True)

    ################################################################
    # test best
    ################################################################
    jobs = []
    for i in range(self.iterations):
      it_dir = '%s/%d' % (exp_dir, i)

      # basenji test
      basenji_cmd = '. /home/drk/anaconda3/etc/profile.d/conda.sh;'
      basenji_cmd += ' conda activate %s;' % self.conda_env
      basenji_cmd += ' %s/basenji_test.py' % self.basenji_path
      basenji_cmd += ' -o %s/test' % it_dir
      basenji_cmd += ' %s' % self.params_file
      basenji_cmd += ' %s/train/model_best.h5' % it_dir
      basenji_cmd += ' %s' % self.data_dir

      basenji_job = slurm.Job(basenji_cmd,
                      name='test%d' % i,
                      out_file='%s/test.out'%it_dir,
                      err_file='%s/test.err'%it_dir,
                      queue=self.queue,
                      cpu=1,
                      gpu=1,
                      mem=23000,
                      time='1:00:00')
      jobs.append(basenji_job)

    slurm.multi_run(jobs, verbose=True)
    """

    ################################################################
    # compare checkpoint on training set
    ################################################################
    ref_cors = []
    for acc_file in glob.glob('%s/*/test_train/acc.txt' % self.ref_dir):
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

    # self.assertGreater(mwp, 0.05)
    # self.assertGreater(tp, 0.05)
    
    ################################################################
    # compare best on test set
    ################################################################
    ref_cors = []
    for acc_file in glob.glob('%s/*/test/acc.txt' % self.ref_dir):
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
    
    # self.assertGreater(mwp, 0.05)
    # self.assertGreater(tp, 0.05)


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  unittest.main()
