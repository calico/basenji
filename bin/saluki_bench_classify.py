#!/usr/bin/env python
from optparse import OptionParser
import joblib
import os
import pdb

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from basenji.dna_io import dna_1hot
from basenji_bench_classify import fold_roc, plot_roc, randfor_full, randfor_roc, ridge_roc

'''
saluki_bench_classify.py
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <ssdp_file> <ssdn_file>'
    parser = OptionParser(usage)
    parser.add_option('-a', dest='abs_value',
            default=False, action='store_true')
    parser.add_option('-i', dest='iterations',
            default=1, type='int',
            help='Cross-validation iterations [Default: %default]')
    parser.add_option('--msl', dest='min_samples_leaf',
            default=1, type='int',
            help='Random forest min_samples_leaf [Default: %default]')
    parser.add_option('-o', dest='out_dir',
            default='class_out')
    parser.add_option('-p', dest='parallel_threads',
            default=1, type='int',
            help='Parallel threads passed to scikit-learn n_jobs [Default: %default]')
    parser.add_option('-r', dest='random_seed',
            default=None, type='int')
    parser.add_option('-s', dest='save_preds',
            default=False, action='store_true',
            help='Save predictions across iterations [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 2:
        parser.error('Must provide positive and negative variant predictions.')
    else:
        ssdp_file = args[0]
        ssdn_file = args[1]

    np.random.seed(options.random_seed)

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    # read positive/negative variants
    Xp = read_ssd(ssdp_file)
    Xn = read_ssd(ssdn_file)
    if options.abs_value:
        Xp = np.abs(Xp)
        Xn = np.abs(Xn)

    # combine
    X = np.concatenate([Xp, Xn], axis=0)
    y = np.array([True]*Xp.shape[0] + [False]*Xn.shape[0], dtype='bool')

    # train classifier
    if X.shape[1] == 1:
        aurocs, fpr_folds, tpr_folds, fpr_mean, tpr_mean = fold_roc(X, y, folds=8)
    else:
        # aurocs, fpr_folds, tpr_folds, fpr_mean, tpr_mean, preds = ridge_roc(X, y, folds=8,
        #           alpha=1, random_state=options.random_seed)
        aurocs, fpr_folds, tpr_folds, fpr_mean, tpr_mean, preds = randfor_roc(X, y, folds=8,
                iterations=options.iterations, random_state=options.random_seed,
                n_jobs=options.parallel_threads, min_samples_leaf=options.min_samples_leaf)

        # save preds
        if options.save_preds:
            np.save('%s/preds.npy' % options.out_dir, preds)

        # save full model
        model = randfor_full(X, y)
        joblib.dump(model, '%s/model.pkl' % options.out_dir)

    # save
    np.save('%s/aurocs.npy' % options.out_dir, aurocs)
    np.save('%s/fpr_mean.npy' % options.out_dir, fpr_mean)
    np.save('%s/tpr_mean.npy' % options.out_dir, tpr_mean)

    # print stats
    stats_out = open('%s/stats.txt' % options.out_dir, 'w')
    auroc_stdev = np.std(aurocs) / np.sqrt(len(aurocs))
    print('AUROC: %.4f (%.4f)' % (np.mean(aurocs), auroc_stdev), file=stats_out)
    stats_out.close()

    # plot roc
    plot_roc(fpr_folds, tpr_folds, options.out_dir)


def read_ssd(ssd_file):
    df = pd.read_csv(ssd_file, sep='\t')
    # ssd = np.array(df.SSD, dtype='float64')
    ssd = np.array(df.iloc[:,2:], dtype='float64')
    # return np.expand_dims(ssd, axis=-1)
    return ssd


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
