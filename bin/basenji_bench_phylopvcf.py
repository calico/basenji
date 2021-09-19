#!/usr/bin/env python
from optparse import OptionParser
import joblib
import os
import pdb
import sys

import h5py
import numpy as np
import pysam
import pyBigWig
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import KFold

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from basenji import dna_io

'''
basenji_bench_phylop.py
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <sad_file> <vcf_file>'
    parser = OptionParser(usage)
    parser.add_option('-d', dest='n_components',
            default=None, type='int',
            help='PCA n_components [Default: %default]')
    parser.add_option('-e', dest='num_estimators',
            default=100, type='int',
            help='Number of random forest estimators [Default: %default]')
    parser.add_option('-i', dest='iterations',
            default=1, type='int',
            help='Cross-validation iterations [Default: %default]')
    parser.add_option('--msl', dest='msl',
            default=1, type='int',
            help='Random forest min_samples_leaf [Default: %default]')
    parser.add_option('-o', dest='out_dir',
            default='regr_out')
    parser.add_option('-p', dest='parallel_threads',
            default=1, type='int',
            help='Parallel threads passed to scikit-learn n_jobs [Default: %default]')
    parser.add_option('-r', dest='random_seed',
            default=44, type='int')
    (options,args) = parser.parse_args()

    if len(args) != 2:
        parser.error('Must provide ISM scores and PhyloP VCF file.')
    else:
        sad_file = args[0]
        phylop_vcf_file = args[1]

    np.random.seed(options.random_seed)

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    ################################################################
    # read mutation scores

    with h5py.File(sad_file, 'r') as h5o:
        mut_sad = h5o['SAD'][:].astype('float32')
    num_muts, num_targets = mut_sad.shape

    ################################################################
    # read mutation phylop

    mut_phylop = []
    for line in open(phylop_vcf_file):
        if not line.startswith('#'):
            a = line.split()
            phylop = float(a[-1].replace('PP=',''))
            mut_phylop.append(phylop)

    # transform PhyloP
    mut_phylop = np.array(mut_phylop, dtype='float32')
    mut_phylop = np.nan_to_num(mut_phylop)
    mut_phylop = np.clip(mut_phylop, -1.5, 5)

    # verify?

    ################################################################
    # regression

    # regressor
    r2s, pcors = randfor_cv(mut_sad, mut_phylop,
        iterations=options.iterations,
        n_estimators=options.num_estimators,
        msl=options.msl,
        random_state=options.random_seed,
        n_jobs=options.parallel_threads)

    # save
    np.save('%s/r2.npy' % options.out_dir, r2s)
    np.save('%s/pcor.npy' % options.out_dir, pcors)

    # print stats
    iterations = len(r2s)
    stats_out = open('%s/stats.txt' % options.out_dir, 'w')
    print('R2 %.4f (%.4f)' % (r2s.mean(), r2s.std()/np.sqrt(iterations)), file=stats_out)
    print('pR %.4f (%.4f)' % (pcors.mean(), pcors.std()/np.sqrt(iterations)), file=stats_out)
    stats_out.close()


def randfor_cv(Xs, ys, folds=8, iterations=1, n_estimators=50, msl=1,
               max_features='log2', random_state=44, n_jobs=8):
    """Compute random forest regression accuracy statistics, shuffling at the sequence level."""
    r2s = []
    pcors = []

    for i in range(iterations):
        rs_iter = random_state + i

        kf = KFold(n_splits=folds, shuffle=True, random_state=rs_iter)

        for train_index, test_index in kf.split(Xs):
            X_train = Xs[train_index]
            y_train = ys[train_index]
            X_test = Xs[test_index]
            y_test = ys[test_index]
                        
            # fit model
            if random_state is None:
                rs_rf = None
            else:
                rs_rf = rs_iter+test_index[0]
            model = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features,
                                          max_depth=64, min_samples_leaf=msl, min_samples_split=2,
                                          random_state=rs_rf, n_jobs=n_jobs)
            model.fit(X_train, y_train)
            
            # predict test set
            preds = model.predict(X_test)

            # compute R2
            r2s.append(explained_variance_score(y_test, preds))

            # compute pearsonr
            pcors.append(pearsonr(y_test, preds)[0])

    r2s = np.array(r2s)
    pcors = np.array(pcors)

    return r2s, pcors


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
