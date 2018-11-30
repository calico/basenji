#!/usr/bin/env python
from optparse import OptionParser
import glob
import pdb
import os

import h5py
import numpy as np
from scipy.stats import cauchy

from basenji.emerald import EmeraldVCF

'''
sad5.py

Interfaces to normalized and population-adjusted SAD scores.
'''

class SAD5:
    def __init__(self, sad_h5_file, sad_key='SAD', recompute_norm=False):
        self.sad_h5_file = sad_h5_file
        self.sad_h5_open = h5py.File(self.sad_h5_file, 'r')

        self.sad_matrix = self.sad_h5_open[sad_key]
        self.num_snps, self.num_targets = self.sad_matrix.shape

        self.target_ids = np.array([tl.decode('UTF-8') for tl in self.sad_h5_open['target_ids']])
        self.target_labels = np.array([tl.decode('UTF-8') for tl in self.sad_h5_open['target_labels']])

        # read SAD percentile indexes into memory
        self.pct_sad = np.array(self.sad_h5_open['SAD_pct'])

        # read percentiles
        self.percentiles = np.around(self.sad_h5_open['percentiles'], 3)
        self.percentiles = np.append(self.percentiles, self.percentiles[-1])

        # fit, if not present
        if recompute_norm or not 'target_cauchy_fit_loc' in self.sad_h5_open:
            self.fit_cauchy()

        # make target-specific fit cauchy's
        target_cauchy_fit_params = zip(self.sad_h5_open['target_cauchy_fit_loc'],
                                       self.sad_h5_open['target_cauchy_fit_scale'])
        self.target_cauchy_fit = [cauchy(*cp) for cp in target_cauchy_fit_params]

        # choose normalizing values, if not present
        if recompute_norm or not 'target_cauchy_norm_loc' in self.sad_h5_open:
            self.norm_cauchy()

        # make target-specific normalizing cauchy's
        target_cauchy_norm_params = zip(self.sad_h5_open['target_cauchy_norm_loc'],
                                        self.sad_h5_open['target_cauchy_norm_scale'])
        self.target_cauchy_norm = [cauchy(*cp) for cp in target_cauchy_norm_params]


    def __getitem__(self, si_ti):
        """Return normalized scores for the given SNP and target indexes."""

        cdf_buf = 1e-5
        if isinstance(si_ti, slice):
            print('SAD5 slice is not implemented.', file=sys.stderr)
            exit(1)

        # single target
        elif isinstance(si_ti, tuple):
            si, ti = si_ti
            sad_st = self.sad_matrix[si,ti].astype('float32')
            sad_st_q = self.target_cauchy_fit[ti].cdf(sad_st)
            if sad_st_q > 0.5:
                sad_st_q -= cdf_buf
            else:
                sad_st_q += cdf_buf
            sad_norm = self.target_cauchy_norm[ti].ppf(sad_st_q)

        elif isinstance(si_ti, (list,np.ndarray)):
            si = si_ti
            sad_s = self.sad_matrix[si,:].astype('float32')
            sad_norm = np.zeros(sad_s.shape)
            for ti in range(self.num_targets):
                sad_s_q = self.target_cauchy_fit[ti].cdf(sad_s[:,ti])
                sad_s_q = np.where(sad_s_q > 0.5, sad_s_q-cdf_buf, sad_s_q+cdf_buf)
                sad_norm[:,ti] = self.target_cauchy_norm[ti].ppf(sad_s_q)

        # single SNP, multiple targets
        else:
            si = si_ti
            sad_s = self.sad_matrix[si,:].astype('float32')
            sad_norm = np.zeros(sad_s.shape)
            for ti in range(self.num_targets):
                sad_s_q = self.target_cauchy_fit[ti].cdf(sad_s[ti])
                sad_s_q = np.where(sad_s_q > 0.5, sad_s_q-cdf_buf, sad_s_q+cdf_buf)
                sad_norm[ti] = self.target_cauchy_norm[ti].ppf(sad_s_q)

        return sad_norm


    def fit_cauchy(self, sample=320000):
        """Fit target-specific Cauchy distributions, and save to HDF5"""

        # sample SNPs
        if sample < self.num_snps:
            ri = sorted(np.random.choice(np.arange(self.num_snps), size=sample, replace=False))
        else:
            ri = np.arange(self.num_snps)

        # read SNPs
        sad = self.sad_matrix[ri,:].astype('float32')

        # initialize fit parameters
        target_cauchy_fit_loc = np.zeros(self.num_targets)
        target_cauchy_fit_scale = np.zeros(self.num_targets)

        # fit parameters
        for ti in range(self.num_targets):
            print('Fitting t%d' % ti, flush=True)
            cp = cauchy.fit(sad[:,ti])
            target_cauchy_fit_loc[ti] = cp[0]
            target_cauchy_fit_scale[ti] = cp[1]

        # write to HDF5
        self.sad_h5_open.close()
        self.sad_h5_open = h5py.File(self.sad_h5_file, 'r+')
        self.sad_h5_open.create_dataset('target_cauchy_fit_loc', data=target_cauchy_fit_loc)
        self.sad_h5_open.create_dataset('target_cauchy_fit_scale', data=target_cauchy_fit_scale)
        self.sad_h5_open.close()
        self.sad_h5_open = h5py.File(self.sad_h5_file, 'r')


    def norm_cauchy(self, target_sets=['CAGE']):
        """Compute normalizing Cauchy distribution parameters within
            target sets, and save to HDF5."""

        # read fits
        if not 'target_cauchy_fit_loc' in self.sad_h5_open:
            raise Something
        target_cauchy_fit_loc = self.sad_h5_open['target_cauchy_fit_loc'][:]
        target_cauchy_fit_scale = self.sad_h5_open['target_cauchy_fit_scale'][:]

        # initialize target hash
        target_hash = {'-':[]}
        for ts in target_sets:
            target_hash[ts] = []

        # hash targets by set
        for ti, target_label in enumerate(self.target_labels):
            target_label_set = '-'
            for ts in target_sets:
                if target_label.startswith(ts):
                    target_label_set = ts
            target_hash[target_label_set].append(ti)

        # initialize norm parameters
        target_cauchy_norm_loc = np.zeros(self.num_targets)
        target_cauchy_norm_scale = np.zeros(self.num_targets)

        # for each target set
        for ts, target_set_indexes in target_hash.items():
            # compute medians
            target_set_loc = np.median(target_cauchy_fit_loc[target_set_indexes])
            target_set_scale = np.median(target_cauchy_fit_scale[target_set_indexes])

            # save
            target_cauchy_norm_loc[target_set_indexes] = target_set_loc
            target_cauchy_norm_scale[target_set_indexes] = target_set_scale

        # write to HDF5
        self.sad_h5_open.close()
        self.sad_h5_open = h5py.File(self.sad_h5_file, 'r+')
        self.sad_h5_open.create_dataset('target_cauchy_norm_loc', data=target_cauchy_norm_loc)
        self.sad_h5_open.create_dataset('target_cauchy_norm_scale', data=target_cauchy_norm_scale)
        self.sad_h5_open.close()
        self.sad_h5_open = h5py.File(self.sad_h5_file, 'r')

    def pos(self, snp_i):
        return self.sad_h5_open['pos'][snp_i]

    def sad_pct(self, sad):
        # compute percentile indexes
        sadq = []
        for ti in range(len(sad)):
            sadq.append(int(np.searchsorted(self.pct_sad[ti], sad[ti])))

        # return percentiles
        return self.percentiles[sadq]

    def snps(self):
        return np.array(self.sad_h5_open['snp'])


class ChrSAD5:
    def __init__(self, sad_h5_path, population='EUR', index_chr=False):
        self.index_chr = index_chr
        self.set_population(population)
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
                if self.index_chr:
                    self.snp_indexes[snp_id] = (ci,i)
                else:
                    self.snp_indexes[snp_id] = i

            # clean up
            del snps

    def open_chr_sad5(self, sad_h5_path):
        self.chr_sad5 = {}

        # TEMP
        # for sad_h5_file in glob.glob('%s/*/sad.h5' % sad_h5_path):
        for sad_h5_file in glob.glob('%s/chr1/sad.h5' % sad_h5_path):
            sad5 = SAD5(sad_h5_file)
            chrm = sad_h5_file.split('/')[-2]
            if chrm.startswith('chr'):
                chrm = chrm[3:]
            self.chr_sad5[chrm] = sad5

    def retrieve_snp(self, snp_id, chrm, pos, ld_t=0.1):
        if chrm.startswith('chr'):
            chrm = chrm[3:]

        if snp_id in self.snp_indexes:
            snp_i = self.snp_indexes[snp_id]

            # retrieve LD variants
            ld_df = self.emerald_vcf.query_ld(snp_id, chrm, pos, ld_t=ld_t)

            # retrieve scores for LD snps
            ld_snp_indexes = np.zeros(ld_df.shape[0], dtype='uint32')
            for si, ld_snp_id in enumerate(ld_df.snp):
                if self.index_chr:
                    _, ld_snp_i = self.snp_indexes[ld_snp_id]
                else:
                    ld_snp_i = self.snp_indexes[ld_snp_id]
                ld_snp_indexes[si] = ld_snp_i
            snps_scores = self.chr_sad5[chrm][ld_snp_indexes]

            # (1xN)(NxT) = (1xT)
            ld_r1 = np.reshape(ld_df.r.values, (1,-1))
            snp_ldscores = np.squeeze(np.matmul(ld_r1, snps_scores))

            return snp_ldscores, ld_df, snps_scores
        else:
            return [], [], None

    def set_population(self, population):
        self.pop_vcf_stem = '%s/popgen/1000G/phase3/%s/1000G.%s.QC' % (os.environ['HG19'], population.lower(), population.upper())
        if glob.glob('%s*' % self.pop_vcf_stem):
            self.emerald_vcf = EmeraldVCF(self.pop_vcf_stem)
        else:
            raise ValueError('Population %s not found' % population)

    def snp_chr_index(self, snp_id):
        if not self.index_chr:
            raise RuntimeError('SNPs not indexed to retrieve chromosome')
        else:
            return self.snp_indexes.get(snp_id, (None,None))

    def snp_index(self, snp_id):
        if self.index_chr:
            chrm, snp_i = self.snp_indexes[snp_id]
        else:
            snp_i = self.snp_indexes.get(snp_id, None)
        return snp_i

    def snp_pos(self, snp_i, chrm):
        return self.chr_sad5[chrm].pos(snp_i)

    def target_info(self):
        # easy access to target information
        chrm = list(self.chr_sad5.keys())[0]
        self.target_ids = self.chr_sad5[chrm].target_ids
        self.target_labels = self.chr_sad5[chrm].target_labels
        self.num_targets = len(self.target_ids)
