#!/usr/bin/env python
from optparse import OptionParser
import glob
import os
import pdb
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import h5py

from basenji.emerald import EmeraldVCF
from basenji.sad5 import ChrSAD5

'''
basenji_fetch_vcf.py

Fetch and synthesize scores, and LD-associated scores, for variants
in the given VCF file.
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <sad_h5_path> <vcf_file>'
    parser = OptionParser(usage)
    parser.add_option('-c', dest='chrom_h5',
            default=False, action='store_true',
            help='HDF5 files split by chromosome [Default: %default]')
    parser.add_option('-f', dest='full_tables',
            default=False, action='store_true',
            help='Print full tables describing all linked variants [Default: %default]')
    parser.add_option('-p', dest='population',
            default='EUR', help='Population code')
    parser.add_option('-o', dest='out_dir',
            default='fetch_vcf')
    (options,args) = parser.parse_args()

    if len(args) != 2:
        parser.error('Must provide SAD HDF5 path and VCF file')
    else:
        sad_h5_path = args[0]
        vcf_file = args[1]

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    ##################################################
    # precursors

    print('Preparing data...', end='', flush=True)
    sad5 = ChrSAD5(sad_h5_path, options.population)
    print('DONE.', flush=True)


    ##################################################
    # parse VCF

    ldscores_out = open('%s/ldscores.txt' % options.out_dir, 'w')
    if options.full_tables:
        full_dir = '%s/full' % options.out_dir
        if not os.path.isdir(full_dir):
            os.mkdir(full_dir)

    for line in open(vcf_file):
        if not line.startswith('#'):
            t0 = time.time()
            a = line.split()
            chrm = a[0]
            pos = int(a[1])
            rsid = a[2]

            # retrieve scores for variants in LD
            snp_ldscores, snp_ld_df, snps_scores = sad5.retrieve_snp(rsid, chrm, pos)

            # print LD scores
            for ti in range(sad5.num_targets):
                cols = (rsid, snp_ldscores[ti], sad5.target_ids[ti], sad5.target_labels[ti])
                print('%-16s  %7.3f  %20s  %s' % cols, file=ldscores_out)

            if options.full_tables:
                # print all LD variant scores
                full_ld_out = open('%s/%s.txt' % (full_dir, rsid), 'w')
                for si in range(snp_ld_df.shape[0]):
                    snp_ld_series = snp_ld_df.iloc[si]
                    snp_scores = snps_scores[si]
                    for ti in range(sad5.num_targets):
                        snp_score_ti = snp_scores[ti]
                        snp_ldscore_ti = snp_ld_series.r * snp_score_ti
                        cols = (snp_ld_series.snp, snp_ldscore_ti, snp_score_ti, snp_ld_series.r,
                                sad5.target_ids[ti], sad5.target_labels[ti])
                        print('%-16s  %7.3f  %7.3f  %6.1f  %20s  %s' % cols, file=full_ld_out)
                full_ld_out.close()

            print(rsid, '%.1fs'%(time.time()-t0))

    ldscores_out.close()



################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
