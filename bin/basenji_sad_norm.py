#!/usr/bin/env python
from optparse import OptionParser
import glob
from multiprocessing import Array, Pool
import os
import pdb

import h5py
import numpy as np
from scipy.stats import cauchy

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns

from basenji.plots import jointplot
from basenji.sad5 import SAD5

'''
basenji_sad_norm.py

Compute normalization parameters across a split chromosome dataset.
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] arg'
    parser = OptionParser(usage)
    parser.add_option('-o', dest='out_dir',
            default='sad_norm')
    parser.add_option('-s', dest='sample',
            default=100000, type='int',
            help='Number of SNPs to sample for fit [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 1:
        parser.error('Must provide SAD HDF5 path')
    else:
        sad_h5_path = args[0]

    # retrieve chromosome SAD HDF5 files
    chr_sad_h5_files = sorted(glob.glob('%s/*/sad.h5' % sad_h5_path))
    assert(len(chr_sad_h5_files) > 0)

    # clean out any existing fits
    # count SNPs across chromosomes
    num_snps = 0
    for chr_sad_h5_file in chr_sad_h5_files:
        chr_sad_h5 = h5py.File(chr_sad_h5_file, 'r+')

        # delete fit params
        if 'target_cauchy_fit_loc' in chr_sad_h5.keys():
            del chr_sad_h5['target_cauchy_fit_loc']
            del chr_sad_h5['target_cauchy_fit_scale']

        # delete norm params
        if 'target_cauchy_norm_loc' in chr_sad_h5.keys():
            del chr_sad_h5['target_cauchy_norm_loc']
            del chr_sad_h5['target_cauchy_norm_scale']

        # count SNPs
        num_snps += chr_sad_h5['SAD'].shape[0]
        num_targets = chr_sad_h5['SAD'].shape[-1]

        chr_sad_h5.close()


    # sample SNPs across chromosomes
    sad = sample_sad(chr_sad_h5_files, options.sample, num_snps, num_targets)


    # initialize fit parameters
    target_cauchy_fit_loc = np.zeros(num_targets)
    target_cauchy_fit_scale = np.zeros(num_targets)

    # fit parameters
    for ti in range(num_targets):
        print('Fitting t%d' % ti, flush=True)
        cp = cauchy.fit(sad[:,ti])
        target_cauchy_fit_loc[ti] = cp[0]
        target_cauchy_fit_scale[ti] = cp[1]
    del sad

    # write across chromosomes
    for chr_sad_h5_file in chr_sad_h5_files:
        chr_sad_h5 = h5py.File(chr_sad_h5_file, 'r+')
        chr_sad_h5.create_dataset('target_cauchy_fit_loc',
                                  data=target_cauchy_fit_loc)
        chr_sad_h5.create_dataset('target_cauchy_fit_scale',
                                  data=target_cauchy_fit_scale)
        chr_sad_h5.close()

    # compute normalization parameters
    for chr_sad_h5_file in chr_sad_h5_files:
        chr_sad5 = SAD5(chr_sad_h5_file)


    # QC fit table
    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)
    fit_out = open('%s/fits.txt' % options.out_dir, 'w')
    for ti in range(num_targets):
        print('%-4d  %7.1e  %7.1e' % (ti, target_cauchy_fit_loc[ti], target_cauchy_fit_scale[ti]), file=fit_out)
    fit_out.close()

    # QC quantiles
    quantile_dir = '%s/quantiles' % options.out_dir
    if not os.path.isdir(quantile_dir):
        os.mkdir(quantile_dir)
    sad_qc = sample_sad(chr_sad_h5_files, 2048, num_snps, num_targets)
    for ti in np.linspace(0, num_targets-1, 64, dtype='int'):
        # compute cauchy and argsort quantiles
        cauchy_q = cauchy.cdf(sad_qc[:,ti], loc=target_cauchy_fit_loc[ti], scale=target_cauchy_fit_scale[ti])
        sort_i = np.argsort(sad_qc[:,ti])

        quantile_pdf = '%s/t%d.pdf' % (quantile_dir, ti)

        jointplot(np.linspace(0,1,len(sort_i)), cauchy_q[sort_i], quantile_pdf,
                  square=True, cor=None, x_label='Empirical', y_label='Cauchy')
        # plt.figure()
        # g = sns.jointplot(np.linspace(0,1,len(sort_i)), cauchy_q[sort_i], joint_kws={'alpha':0.5, 's':10})
        # plt.savefig()
        # plt.close()

    # QC plots
    norm_dir = '%s/norm' % options.out_dir
    if not os.path.isdir(norm_dir):
        os.mkdir(norm_dir)
    chr_sad5 = SAD5(chr_sad_h5_files[0])
    qc_sample = 2048
    if qc_sample < chr_sad5.num_snps:
        ri = sorted(np.random.choice(np.arange(chr_sad5.num_snps), size=qc_sample, replace=False))
    else:
        ri = np.arange(chr_sad5.num_snps)
    qc_sad_raw = chr_sad5.sad_matrix[ri]
    qc_sad_norm = chr_sad5[ri]
    for ti in np.linspace(0, num_targets-1, 32, dtype='int'):
        plt.figure()
        sns.jointplot(qc_sad_raw[:,ti], qc_sad_norm[:,ti], joint_kws={'alpha':0.5, 's':10})
        plt.savefig('%s/t%d.pdf' % (norm_dir, ti))
        plt.close()


def sample_sad(chr_sad_h5_files, sample, num_snps, num_targets):
    # sample SNPs uniformly across chromosomes
    if sample < num_snps:
        ri = np.random.choice(np.arange(num_snps), size=sample, replace=False)
        ri.sort()
    else:
        ri = np.arange(num_snps)

    # read SAD across chromosomes
    sad = np.zeros((len(ri), num_targets), dtype='float32')
    chr_start = 0
    si = 0
    for chr_sad_h5_file in chr_sad_h5_files:
        chr_sad_h5 = h5py.File(chr_sad_h5_file, 'r')

        # determine chr interval
        chr_end = chr_start + chr_sad_h5['SAD'].shape[0]

        # filter/transform random indexes for chromosome
        chr_ri_mask = (chr_start <= ri) & (ri < chr_end)
        chr_ri = ri[chr_ri_mask] - chr_start
        chr_snps = len(chr_ri)

        # read chr SNPs
        sad[si:si+chr_snps,:] = chr_sad_h5['SAD'][chr_ri,:]
        chr_sad_h5.close()

        # advance indexes
        si += chr_snps
        chr_start = chr_end

    return sad


def fit_cauchy(sad, ti):
    print('Fitting t%d' % ti)
    return cauchy.fit(sad[:,ti])

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
