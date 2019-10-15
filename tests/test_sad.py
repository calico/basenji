#!/usr/bin/env python
from optparse import OptionParser

import os
import pdb
import shutil
import subprocess
import unittest

import h5py
import pysam
import numpy as np
from sklearn.metrics import explained_variance_score

from basenji import params
import basenji.dna_io as dna_io
import basenji.vcf as bvcf
import basenji_sad_ref


class TestSAD(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.params_file = 'data/params.txt'
    cls.model_file = 'data/model_best.tf'
    cls.vcf_file = 'data/regulatory_validated.vcf'

  def test_sad(self):
    if os.path.isdir('sad/test'):
      shutil.rmtree('sad/test')

    sad_opts = '--rc --shifts "0,21" -o sad/test'

    cmd = 'basenji_sad.py %s %s %s %s' % \
        (sad_opts, self.params_file, self.model_file, self.vcf_file)
    subprocess.call(cmd, shell=True)

    saved_h5 = h5py.File('sad/saved/sad.h5', 'r')
    saved_sad = saved_h5['SAD'][:]
    saved_h5.close()

    this_h5 = h5py.File('sad/test/sad.h5', 'r')
    this_sad = this_h5['SAD'][:]
    this_h5.close()

    np.testing.assert_allclose(this_sad, saved_sad, atol=1e-3, rtol=1e-3)

    shutil.rmtree('sad/test')


  def test_usad(self):
    if os.path.isdir('sad/utest'):
      shutil.rmtree('sad/utest')

    sad_opts = '-u --rc --shifts "0,21" -o sad/utest'

    cmd = 'basenji_sad.py %s %s %s %s' % \
        (sad_opts, self.params_file, self.model_file, self.vcf_file)
    subprocess.call(cmd, shell=True)

    saved_h5 = h5py.File('sad/saved/usad.h5', 'r')
    saved_sad = saved_h5['SAD'][:]
    saved_h5.close()

    this_h5 = h5py.File('sad/utest/sad.h5', 'r')
    this_sad = this_h5['SAD'][:]
    this_h5.close()

    np.testing.assert_allclose(this_sad, saved_sad, atol=1e-3, rtol=1e-3)

    shutil.rmtree('sad/utest')

  def test_multi(self):
    if os.path.isdir('sad/testm'):
      shutil.rmtree('sad/testm')

    sad_opts = '--rc --shifts "0,21"'
    sad_opts += ' -o sad/testm -q "" -p 4'

    cmd = 'basenji_sad_multi.py %s %s %s %s' % \
        (sad_opts, self.params_file, self.model_file, self.vcf_file)
    subprocess.call(cmd, shell=True)

    saved_h5 = h5py.File('sad/saved/sad.h5', 'r')
    this_h5 = h5py.File('sad/testm/sad.h5', 'r')

    saved_keys = sorted(saved_h5.keys())
    this_keys = sorted(this_h5.keys())
    assert(len(saved_keys) == len(this_keys))
    assert(saved_keys == this_keys)

    for key in saved_h5:
      if key[-4:] != '_pct':
        saved_value = saved_h5[key][:]
        this_value = this_h5[key][:]

        if saved_value.dtype.char == 'S':
          np.testing.assert_array_equal(saved_value, this_value)
        else:
          np.testing.assert_allclose(saved_value, this_value, atol=1e-1, rtol=5e-2)
          r2 = explained_variance_score(saved_value.flatten(), this_value.flatten())
          assert(r2 > 0.999)

    saved_h5.close()
    this_h5.close()

    shutil.rmtree('sad/testm')


class TestSadRef(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.genome_fasta = '%s/assembly/hg19.fa' % os.environ['HG19']
    cls.params_file = 'data/params.txt'
    cls.model_file = 'data/model_best.tf'
    cls.vcf_file = 'data/regulatory_validated.vcf'

    cls.params = params.read_job_params(cls.params_file, require=['seq_length','num_targets'])

  def test_run(self):
    if os.path.isdir('sad/testr'):
      shutil.rmtree('sad/testr')

    sad_opts = '-o sad/testr --rc --shifts "0,21"'
    cmd = 'basenji_sad_ref.py %s %s %s %s' % \
        (sad_opts, self.params_file, self.model_file, self.vcf_file)
    return_code = subprocess.call(cmd, shell=True)
    self.assertEqual(return_code, 0)

    shutil.rmtree('sad/testr')

  def test_multi(self):
    if os.path.isdir('sad/testrm'):
      shutil.rmtree('sad/testrm')

    sad_opts = '--rc --shifts "0,21"'
    sad_opts += ' -o sad/testrm -q "" -p 4'
    cmd = 'basenji_sad_ref_multi.py %s %s %s %s' % \
        (sad_opts, self.params_file, self.model_file, self.vcf_file)
    subprocess.call(cmd, shell=True)

    saved_h5 = h5py.File('sad/saved/sadr.h5', 'r')
    this_h5 = h5py.File('sad/testrm/sad.h5', 'r')

    saved_keys = sorted(saved_h5.keys())
    this_keys = sorted(this_h5.keys())
    assert(len(saved_keys) == len(this_keys))
    assert(saved_keys == this_keys)

    for key in saved_h5:
      if key[-4:] != '_pct':
        saved_value = saved_h5[key][:]
        this_value = this_h5[key][:]

        if saved_value.dtype.char == 'S':
          assert((saved_value == this_value).all())
          np.testing.assert_array_equal(saved_value, this_value)
        else:
          np.testing.assert_allclose(saved_value, this_value, atol=2e-1, rtol=2e-1)
          r2 = explained_variance_score(saved_value.flatten(), this_value.flatten())
          assert(r2 > 0.999)

    saved_h5.close()
    this_h5.close()

    shutil.rmtree('sad/testrm')

  def test_misref(self):
    sad_opts = '-o sad_ref'

    vcf_misref_file = 'data/regulatory_validated_misref.vcf'
    cmd = 'basenji_sad_ref.py %s %s %s %s' % \
        (sad_opts, self.params_file, self.model_file, vcf_misref_file)
    return_code = subprocess.call(cmd, shell=True)
    self.assertEqual(return_code, 1)

    if os.path.isdir('sad_ref'):
      shutil.rmtree('sad_ref')

  def test_misorder(self):
    sad_opts = '-o sad_ref'

    vcf_misorder_file = 'data/regulatory_validated_misorder.vcf'
    cmd = 'basenji_sad_ref.py %s %s %s %s' % \
        (sad_opts, self.params_file, self.model_file, vcf_misorder_file)
    return_code = subprocess.call(cmd, shell=True)
    self.assertEqual(return_code, 1)

    if os.path.isdir('sad_ref'):
      shutil.rmtree('sad_ref')

  def test_cluster(self):
    # read sorted SNPs from VCF
    snps = bvcf.vcf_snps(self.vcf_file, require_sorted=True)

    # cluster SNPs by position
    snp_clusters = basenji_sad_ref.cluster_snps(snps, self.params['seq_length'], 0.25)

    # two SNPs should be clustered together
    self.assertEqual(len(snps)-1, len(snp_clusters))
    self.assertEqual(len(snp_clusters[0].snps), 1)
    self.assertEqual(len(snp_clusters[6].snps), 2)

  def test_get_1hots(self):
    # read sorted SNPs from VCF
    snps = bvcf.vcf_snps(self.vcf_file, require_sorted=True)

    # cluster SNPs by position
    snp_clusters = basenji_sad_ref.cluster_snps(snps, self.params['seq_length'], 0.25)

    # delimit sequence boundaries
    [sc.delimit(self.params['seq_length']) for sc in snp_clusters]

    # open genome FASTA
    genome_open = pysam.Fastafile(self.genome_fasta)

    ########################################
    # verify single SNP

    # get 1 hot coded sequences
    snp_1hot_list = snp_clusters[0].get_1hots(genome_open)

    self.assertEqual(len(snp_1hot_list), 2)
    self.assertEqual(snp_1hot_list[1].shape, (self.params['seq_length'], 4))

    mid_i = self.params['seq_length'] // 2 - 1
    self.assertEqual(mid_i, snps[0].seq_pos)

    ref_nt = dna_io.hot1_get(snp_1hot_list[0], mid_i)
    self.assertEqual(ref_nt, snps[0].ref_allele)

    alt_nt = dna_io.hot1_get(snp_1hot_list[1], mid_i)
    self.assertEqual(alt_nt, snps[0].alt_alleles[0])


    ########################################
    # verify multiple SNPs

    # get 1 hot coded sequences
    snp_1hot_list = snp_clusters[6].get_1hots(genome_open)

    self.assertEqual(len(snp_1hot_list), 3)

    snp1, snp2 = snps[6:8]

    # verify position 1 changes between 0 and 1
    nt = dna_io.hot1_get(snp_1hot_list[0], snp1.seq_pos)
    self.assertEqual(nt, snp1.ref_allele)

    nt = dna_io.hot1_get(snp_1hot_list[1], snp1.seq_pos)
    self.assertEqual(nt, snp1.alt_alleles[0])

    # verify position 2 is unchanged between 0 and 1
    nt = dna_io.hot1_get(snp_1hot_list[0], snp2.seq_pos)
    self.assertEqual(nt, snp2.ref_allele)

    nt = dna_io.hot1_get(snp_1hot_list[1], snp2.seq_pos)
    self.assertEqual(nt, snp2.ref_allele)

    # verify position 2 is unchanged between 0 and 2
    nt = dna_io.hot1_get(snp_1hot_list[0], snp1.seq_pos)
    self.assertEqual(nt, snp1.ref_allele)

    nt = dna_io.hot1_get(snp_1hot_list[2], snp1.seq_pos)
    self.assertEqual(nt, snp1.ref_allele)

    # verify position 2 changes between 0 and 2
    nt = dna_io.hot1_get(snp_1hot_list[0], snp2.seq_pos)
    self.assertEqual(nt, snp2.ref_allele)

    nt = dna_io.hot1_get(snp_1hot_list[2], snp2.seq_pos)
    self.assertEqual(nt, snp2.alt_alleles[0])

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  unittest.main()
