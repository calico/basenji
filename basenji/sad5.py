#!/usr/bin/env python
from optparse import OptionParser
import pdb

import h5py
import numpy as np
from scipy.stats import cauchy

'''
sad5.py

SAD HDF5 interface.
'''

class SAD5:
    def __init__(self, sad_h5_file, sad_key='SAD', recompute_norm=False):
        self.sad_h5_file = sad_h5_file
        self.sad_h5_open = h5py.File(self.sad_h5_file, 'r')

        self.sad_matrix = self.sad_h5_open[sad_key]
        self.num_snps, self.num_targets = self.sad_matrix.shape

        self.target_ids = [tl.decode('UTF-8') for tl in self.sad_h5_open['target_ids']]
        self.target_labels = [tl.decode('UTF-8') for tl in self.sad_h5_open['target_labels']]

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


    def snps(self):
        return np.array(self.sad_h5_open['snp'])
