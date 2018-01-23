#!/usr/bin/env python
from optparse import OptionParser

import os
import pdb
import subprocess
import unittest

import numpy as np

from basenji.genedata import GeneData

class TestSED(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    fasta_file = '%s/assembly/hg19.fa' % os.environ['HG19']
    gtf_file = 'data/genes.gtf'
    cls.h5_file = 'data/genes.h5'
    target_wigs_file = 'data/target_wigs.txt'
    cls.seq_len = 131072
    cls.pool_width = 128

    cmd = 'basenji_hdf5_genes.py -l %d -w %d -t %s --w5 %s %s %s' % (cls.seq_len, cls.pool_width, target_wigs_file, fasta_file, gtf_file, cls.h5_file)
    subprocess.call(cmd, shell=True)

    cls.gene_data = GeneData(cls.h5_file)


  def test_sed(self):
    params_file = 'data/params.txt'
    model_file = 'data/model_best.tf'
    variants_file = 'data/variants.vcf'
    targets_file = 'data/target_wigs_index.txt'

    cmd = 'basenji_sed.py -a -o sed --rc -t %s %s %s %s %s' % \
        (targets_file, params_file, model_file, self.h5_file, variants_file)
    subprocess.call(cmd, shell=True)

    # check variants
    variant_scores = {}
    for line in open(variants_file):
        if not line.startswith('#'):
            a = line.split('\t')
            variant_scores[a[2]] = 0

    sed_in = open('sed/sed_gene.txt')
    sed_in.readline()
    for line in sed_in:
        a = line.split()
        variant_scores[a[0]] += 1
    sed_in.close()

    for rsid in variant_scores:
        self.assertGreater(variant_scores[rsid], 0)


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  unittest.main()