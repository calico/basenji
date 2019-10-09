#!/usr/bin/env python
from optparse import OptionParser

import pdb
import os
import shutil
import unittest

import numpy as np
import pandas as pd

import slurm

class TestTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.data_dir = 'test/data'
    cls.ref_acc_file = 'test/ref/acc.txt'
    cls.exp_dir = 'test/exp'

    cls.params_file = 'test/params.json'
    cls.model_file = 'test/model_best.h5'
    
    cls.conda_env = 'tf1.14-gpu'
    cls.queue = 'gtx1080ti'

    if os.path.isfile(cls.ref_acc_file):
        model_time = os.path.getmtime(os.path.realpath(cls.model_file))
        ref_acc_time = os.path.getmtime(cls.ref_acc_file)
        assert(model_time < ref_acc_time)
    else:
        print('Reference accuracy not found; proceeding under the assumption that we are generating it.')

  def test_train(self):
    if os.path.isdir(self.exp_dir):
      shutil.rmtree(self.exp_dir)
    os.mkdir(self.exp_dir)

    ################################################################
    # basenji test
    ################################################################
    basenji_cmd = '. /home/drk/anaconda3/etc/profile.d/conda.sh;'
    basenji_cmd += ' conda activate %s;' % self.conda_env
    basenji_cmd += ' basenji_test.py'
    basenji_cmd += ' -o %s' % self.exp_dir
    basenji_cmd += ' --rc'
    basenji_cmd += ' --shifts "1,0,-1"'
    basenji_cmd += ' %s' % self.params_file
    basenji_cmd += ' %s' % self.model_file
    basenji_cmd += ' %s' % self.data_dir

    basenji_job = slurm.Job(basenji_cmd,
                            name='test_test',
                            out_file='%s/test.out'%self.exp_dir,
                            err_file='%s/test.err'%self.exp_dir,
                            queue=self.queue,
                            cpu=1,
                            gpu=1,
                            mem=23000,
                            time='1:00:00')

    slurm.multi_run([basenji_job], verbose=True)

    ################################################################
    # compare
    ################################################################
    if os.path.isfile(self.ref_acc_file):
        ref_df = pd.read_csv(self.ref_acc_file, sep='\t', index_col=0)

        exp_acc_file = '%s/acc.txt' % self.exp_dir
        exp_df = pd.read_csv(exp_acc_file, sep='\t', index_col=0)

        np.testing.assert_allclose(ref_df.pearsonr, exp_df.pearsonr, atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(ref_df.r2, exp_df.r2, atol=1e-3, rtol=1e-3)

    else:
        print('Moving experiment to reference.')
        os.rename(self.exp_dir, os.path.split(self.ref_acc_file)[0])

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  unittest.main()
