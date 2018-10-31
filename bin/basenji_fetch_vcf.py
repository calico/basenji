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

from basenji.sad5 import SAD5

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
    parser.add_option('-p', dest='pop_vcf_stem',
            default='%s/popgen/1000G/phase3/eur/1000G.EUR.QC'%os.environ['HG19'])
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
    sad5 = IndexSAD5(sad_h5_path, options.pop_vcf_stem)
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


class IndexSAD5:
    def __init__(self, sad_h5_path, pop_vcf_stem):
        self.pop_vcf_stem = pop_vcf_stem
        self.max_ld_distance = 1000000
        self.open_chr_sad5(sad_h5_path)
        self.index_snps()
        self.target_info()

    def index_snps(self):
        """Hash RSID's to HDF5 index."""
        self.snp_indexes = {}

        # for each chromosome
        for ci in self.chr_sad5:

            # hash SNP ids to indexes
            snps = self.chr_sad5[ci].snps()
            for i, snp_id in enumerate(snps):
                snp_id = snp_id.decode('UTF-8')
                self.snp_indexes[snp_id] = i

            # clean up
            del snps

    def retrieve_ld(self, snp_id, chrm, pos, ld_threshold=0.1):
        """Retrieve SNPs in LD with the given SNP."""

        # determine search region
        ld_region_start = max(0, pos - self.max_ld_distance)
        ld_region_end = pos + self.max_ld_distance

        # construct emerald command
        cmd  = 'emeraLD'
        cmd += ' -i %s.%s.vcf.gz' % (self.pop_vcf_stem, chrm)
        cmd += ' --rsid %s' % snp_id
        cmd += ' --region %s:%d-%d' % (chrm, ld_region_start, ld_region_end)
        cmd += ' --threshold %f' % ld_threshold
        cmd += ' --no-phase'
        cmd += ' --extra'
        cmd += ' --stdout'
        cmd += ' 2> /dev/null'

        ld_snps = [snp_id]
        ld_r = [1.0]
        ld_pos = [pos]

        # parse returned lines
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        for line in proc.stdout:
            line = line.decode('UTF-8')
            if not line.startswith('#'):
                a = line.split()
                ld_pos.append(int(a[4]))
                ld_snps.append(a[5])
                ld_r.append(float(a[7]))
        proc.communicate()

        # sort by position
        sort_indexes = np.argsort(ld_pos)
        ld_snps = np.array(ld_snps)[sort_indexes]
        ld_r = np.array(ld_r)[sort_indexes]

        # construct data frame
        ld_df = pd.DataFrame()
        ld_df['snp'] = ld_snps
        ld_df['r'] = ld_r
        return ld_df

    def open_chr_sad5(self, sad_h5_path):
        self.chr_sad5 = {}

        for sad_h5_file in glob.glob('%s/*/sad.h5' % sad_h5_path):
            sad5 = SAD5(sad_h5_file)
            chrm = sad_h5_file.split('/')[-2]
            if chrm.startswith('chr'):
                chrm = chrm[3:]
            self.chr_sad5[chrm] = sad5

    def retrieve_snp(self, snp_id, chrm, pos):
        if chrm.startswith('chr'):
            chrm = chrm[3:]

        if snp_id in self.snp_indexes:
            snp_i = self.snp_indexes[snp_id]

            # retrieve LD variants
            ld_df = self.retrieve_ld(snp_id, chrm, pos)

            # does emerald return the snp itself with 1.0?

            # retrieve scores for LD snps
            ld_snp_indexes = np.zeros(ld_df.shape[0], dtype='uint32')
            for si, ld_snp_id in enumerate(ld_df.snp):
                ld_snp_indexes[si] = self.snp_indexes[ld_snp_id]
            snps_scores = self.chr_sad5[chrm][ld_snp_indexes]

            # (1xN)(NxT) = (1xT)
            ld_r1 = np.reshape(ld_df.r.values, (1,-1))
            snp_ldscores = np.squeeze(np.matmul(ld_r1, snps_scores))

            return snp_ldscores, ld_df, snps_scores
        else:
            return [], [], None


    def target_info(self):
        # easy access to target information
        chrm = list(self.chr_sad5.keys())[0]
        self.target_ids = self.chr_sad5[chrm].target_ids
        self.target_labels = self.chr_sad5[chrm].target_labels
        self.num_targets = len(self.target_ids)


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
