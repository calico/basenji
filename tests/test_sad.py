#!/usr/bin/env python
from optparse import OptionParser

import os
import pdb
import subprocess
import unittest

import h5py

import numpy as np

class TestSAD(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.params_file = 'data/params.txt'
    cls.model_file = 'data/model_best.tf'
    cls.vcf_file = 'data/regulatory_validated.vcf'
    cls.seq_length = 131072

  def test_sad(self):
    cmd = 'basenji_sad.py --h5 -l %d -o sad/test --rc --shifts "0,1" %s %s %s' % \
        (self.seq_length, self.params_file, self.model_file, self.vcf_file)
    subprocess.call(cmd, shell=True)

    saved_h5 = h5py.File('sad/saved/sad.h5', 'r')
    saved_sad = saved_h5['SAD'][:]

    this_h5 = h5py.File('sad/test/sad.h5', 'r')
    this_sad = this_h5['SAD'][:]

    np.testing.assert_allclose(this_sad, saved_sad, atol=1e-3, rtol=1e-3)

  def test_usad(self):
    cmd = 'basenji_sad.py --h5 -l %d -o sad/utest --rc --shifts "0,1" -u %s %s %s' % \
        (self.seq_length, self.params_file, self.model_file, self.vcf_file)
    subprocess.call(cmd, shell=True)

    saved_h5 = h5py.File('sad/usaved/sad.h5', 'r')
    saved_sad = saved_h5['SAD'][:]

    this_h5 = h5py.File('sad/utest/sad.h5', 'r')
    this_sad = this_h5['SAD'][:]

    np.testing.assert_allclose(this_sad, saved_sad, atol=1e-3, rtol=1e-3)


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  unittest.main()