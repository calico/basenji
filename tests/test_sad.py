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

import basenji.dna_io as dna_io
import basenji.vcf as bvcf
import basenji_sadq_ref


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


class TestSadQRef(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.genome_fasta = '%s/assembly/hg19.fa' % os.environ['HG19']
    cls.params_file = 'data/params.txt'
    cls.model_file = 'data/model_best.tf'
    cls.vcf_file = 'data/regulatory_validated.vcf'
    cls.seq_length = 131072

  def test_run(self):
    sad_opts = '-l %d -o sadq_ref' % self.seq_length

    cmd = 'basenji_sadq_ref.py %s %s %s %s' % \
        (sad_opts, self.params_file, self.model_file, self.vcf_file)
    return_code = subprocess.call(cmd, shell=True)
    self.assertEqual(return_code, 0)

    if os.path.isdir('sadq_ref'):
      shutil.rmtree('sadq_ref')

  def test_misref(self):
    sad_opts = '-l %d -o sadq_ref' % self.seq_length

    vcf_misref_file = 'data/regulatory_validated_misref.vcf'
    cmd = 'basenji_sadq_ref.py %s %s %s %s' % \
        (sad_opts, self.params_file, self.model_file, vcf_misref_file)
    return_code = subprocess.call(cmd, shell=True)
    self.assertEqual(return_code, 1)

    if os.path.isdir('sadq_ref'):
      shutil.rmtree('sadq_ref')

  def test_misorder(self):
    sad_opts = '-l %d -o sadq_ref' % self.seq_length

    vcf_misorder_file = 'data/regulatory_validated_misorder.vcf'
    cmd = 'basenji_sadq_ref.py %s %s %s %s' % \
        (sad_opts, self.params_file, self.model_file, vcf_misorder_file)
    return_code = subprocess.call(cmd, shell=True)
    self.assertEqual(return_code, 1)

    if os.path.isdir('sadq_ref'):
      shutil.rmtree('sadq_ref')

  def test_cluster(self):
    # read sorted SNPs from VCF
    snps = bvcf.vcf_snps(self.vcf_file, require_sorted=True)

    # cluster SNPs by position
    snp_clusters = basenji_sadq_ref.cluster_snps(snps, self.seq_length, 0.25)

    # two SNPs should be clustered together
    self.assertEqual(len(snps)-1, len(snp_clusters))
    self.assertEqual(len(snp_clusters[0].snps), 1)
    self.assertEqual(len(snp_clusters[6].snps), 2)

  def test_get_1hots(self):
    # read sorted SNPs from VCF
    snps = bvcf.vcf_snps(self.vcf_file, require_sorted=True)

    # cluster SNPs by position
    snp_clusters = basenji_sadq_ref.cluster_snps(snps, self.seq_length, 0.25)

    # delimit sequence boundaries
    [sc.delimit(self.seq_length) for sc in snp_clusters]

    # open genome FASTA
    genome_open = pysam.Fastafile(self.genome_fasta)

    ########################################
    # verify single SNP

    # get 1 hot coded sequences
    snp_1hot_list = snp_clusters[0].get_1hots(genome_open)

    self.assertEqual(len(snp_1hot_list), 2)
    self.assertEqual(snp_1hot_list[1].shape, (self.seq_length, 4))

    mid_i = self.seq_length // 2 - 1
    self.assertEqual(mid_i, snps[0].seq_pos)

    ref_nt = dna_io.hot1_get(snp_1hot_list[0], mid_i)
    self.assertEqual(ref_nt, snps[0].ref_allele)

    alt_nt = dna_io.hot1_get(snp_1hot_list[1], mid_i)
    self.assertEqual(alt_nt, snps[0].alt_allele)


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
    self.assertEqual(nt, snp1.alt_allele)

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
    self.assertEqual(nt, snp2.alt_allele)


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  unittest.main()