#!/usr/bin/env python
from optparse import OptionParser

import os
import pdb
import subprocess
import unittest

import numpy as np

from basenji.genedata import GeneData

class TestGenesH5(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    fasta_file = '%s/assembly/hg19.fa' % os.environ['HG19']
    gtf_file = 'data/genes.gtf'
    h5_file = 'data/genes.h5'
    target_wigs_file = 'data/target_wigs.txt'
    cls.seq_len = 131072
    cls.pool_width = 128

    cmd = 'basenji_hdf5_genes.py -l %d -w %d -t %s --w5 %s %s %s' % (cls.seq_len, cls.pool_width, target_wigs_file, fasta_file, gtf_file, h5_file)
    subprocess.call(cmd, shell=True)

    cls.gene_data = GeneData(h5_file)


  def test_seqs(self):
    gene_seqs = self.gene_data.gene_seqs

    # check sequence number
    self.assertEqual(len(gene_seqs), 2)


    # HNRNPU sequence
    hnrnpu_chrom = 'chr1'
    hnrnpu_tss = [245026396, 245027827]
    hnrnpu_mid = int(np.mean(hnrnpu_tss))
    hnrnpu_start = hnrnpu_mid - self.seq_len//2   # 244961576
    hnrnpu_end = hnrnpu_mid + self.seq_len//2   # 245092648

    self.assertEqual(gene_seqs[0].chrom, hnrnpu_chrom)
    self.assertEqual(gene_seqs[0].start, hnrnpu_start)
    self.assertEqual(gene_seqs[0].end, hnrnpu_end)
    self.assertEqual(len(gene_seqs[0].tss_list), 2)


    # CTCF sequence
    ctcf_chrom = 'chr16'
    ctcf_tss = [67596310]
    ctcf_mid = int(np.mean(ctcf_tss))
    ctcf_start = ctcf_mid - self.seq_len//2
    ctcf_end = ctcf_mid + self.seq_len//2

    self.assertEqual(gene_seqs[1].chrom, ctcf_chrom)
    self.assertEqual(gene_seqs[1].start, ctcf_start)
    self.assertEqual(gene_seqs[1].end, ctcf_end)
    self.assertEqual(len(gene_seqs[1].tss_list), 1)


  def test_targets(self):
    tss_targets = self.gene_data.tss_targets

    self.assertEqual(tss_targets.shape[0], 3)
    self.assertEqual(tss_targets.shape[1], 2)

    # HNRNPU TSS1
    self.assertAlmostEqual(tss_targets[0,0], 15, places=0)
    self.assertAlmostEqual(tss_targets[0,1], 8, places=0)

    # HNRNPU TSS2
    self.assertAlmostEqual(tss_targets[1,0], 149, places=0)
    self.assertAlmostEqual(tss_targets[1,1], 143, places=0)

    # CTCF TSS3
    self.assertAlmostEqual(tss_targets[2,0], 70, places=0)
    self.assertAlmostEqual(tss_targets[2,1], 40, places=0)



class TestTestGenes(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    fasta_file = '%s/assembly/hg19.fa' % os.environ['HG19']
    gtf_file = 'data/genes.gtf'
    h5_file = 'data/genes.h5'
    target_wigs_file = 'data/target_wigs.txt'
    cls.seq_len = 131072
    cls.pool_width = 128

    cmd = 'basenji_hdf5_genes.py -l %d -w %d -t %s --w5 %s %s %s' % (cls.seq_len, cls.pool_width, target_wigs_file, fasta_file, gtf_file, h5_file)
    subprocess.call(cmd, shell=True)

    cls.gene_data = GeneData(h5_file)

  def test_test_genes(cls):
    params_file = 'data/params.txt'
    model_file = 'data/model_best.tf'
    h5_file = 'data/genes.h5'

    cmd = 'basenji_test_genes.py --heat --table --tss -v %s %s %s' % (params_file, model_file, h5_file)
    subprocess.call(cmd, shell=True)

    print('GOOD')


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  unittest.main()