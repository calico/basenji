#!/usr/bin/env python
from optparse import OptionParser
import joblib
import os
import pdb

import h5py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from basenji.dna_io import dna_1hot

'''
basenji_sad_classify.py
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <sadp_file> <sadn_file>'
    parser = OptionParser(usage)
    parser.add_option('-a', dest='abs_value',
            default=False, action='store_true')
    parser.add_option('-i', dest='iterations',
            default=1, type='int',
            help='Cross-validation iterations [Default: %default]')
    parser.add_option('-l', dest='log',
            default=False, action='store_true')
    parser.add_option('-m', dest='model_pkl',
            help='Dimension reduction model')
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
        sadp_file = args[0]
        sadn_file = args[1]

    np.random.seed(options.random_seed)

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    # read dimension reduction model
    if options.model_pkl:
        model = joblib.load(options.model_pkl)

    # read positive/negative variants
    Xp = read_sad(sadp_file)
    Xn = read_sad(sadn_file)
    if options.log:
        Xp = np.arcsinh(Xp)
        Xn = np.arcsinh(Xn)
    if options.abs_value:
        Xp = np.abs(Xp)
        Xn = np.abs(Xn)
    if options.model_pkl:
        Xp = model.transform(Xp)
        Xn = model.transform(Xn)

    # combine
    X = np.concatenate([Xp, Xn], axis=0)
    y = np.array([True]*Xp.shape[0] + [False]*Xn.shape[0], dtype='bool')

    # train classifier
    if X.shape[1] == 1:
        aurocs, fpr_folds, tpr_folds, fpr_mean, tpr_mean = fold_roc(X, y, folds=8)
    else:
        # aurocs, fpr_folds, tpr_folds, fpr_full, tpr_full = ridge_roc(X, y, folds=8, alpha=10000)
        aurocs, fpr_folds, tpr_folds, fpr_mean, tpr_mean, preds = randfor_roc(X, y, folds=8,
                iterations=options.iterations, random_state=options.random_seed,
                n_jobs=options.parallel_threads)

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


def fold_roc(X, y, folds=8, random_state=44):
    """Compute ROC for a single value, sans model."""
    aurocs = []
    fpr_folds = []
    tpr_folds = []

    fpr_mean = np.linspace(0, 1, 256)
    tpr_mean = []

    # preds_full = np.zeros(y.shape)

    kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)

    for train_index, test_index in kf.split(X):
        # predict test set (as is)
        preds = X[test_index,:]

        # save
        # preds_full[test_index] = preds.squeeze()

        # compute ROC curve
        fpr, tpr, _ = roc_curve(y[test_index], preds)
        fpr_folds.append(fpr)
        tpr_folds.append(tpr)

        interp_tpr = np.interp(fpr_mean, fpr, tpr)
        interp_tpr[0] = 0.0
        tpr_mean.append(interp_tpr)

        # compute AUROC
        aurocs.append(roc_auc_score(y[test_index], preds))

    # fpr_full, tpr_full, _ = roc_curve(y, preds_full)
    tpr_mean = np.array(tpr_mean).mean(axis=0)

    return np.array(aurocs), np.array(fpr_folds), np.array(tpr_folds), fpr_mean, tpr_mean


def plot_roc(fprs, tprs, out_dir):
    plt.figure(figsize=(4,4))

    for fi in range(len(fprs)):
        plt.plot(fprs[fi], tprs[fi], alpha=0.25)

    ax = plt.gca()
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')

    sns.despine()
    plt.tight_layout()

    plt.savefig('%s/roc.pdf' % out_dir)
    plt.close()


def randfor_full(X, y, random_state=None, n_jobs=1):
    """Compute a single random forest on the full data."""
    model = RandomForestClassifier(n_estimators=100, max_features='log2', max_depth=64,
                                   min_samples_leaf=1, min_samples_split=2,
                                   random_state=random_state, n_jobs=n_jobs)
    model.fit(X, y)
    return model


def randfor_roc(X, y, folds=8, iterations=1, random_state=None, n_jobs=1):
    """Compute ROC using a random forest."""
    aurocs = []
    fpr_folds = []
    tpr_folds = []
    fpr_fulls = []
    tpr_fulls = []
    preds_return = []

    fpr_mean = np.linspace(0, 1, 256)
    tpr_mean = []

    for i in range(iterations):
        rs_iter = random_state + i
        preds_full = np.zeros(y.shape)

        kf = KFold(n_splits=folds, shuffle=True, random_state=rs_iter)

        for train_index, test_index in kf.split(X):
            # fit model
            if random_state is None:
                rs_rf = None
            else:
                rs_rf = rs_iter+test_index[0]
            model = RandomForestClassifier(n_estimators=100, max_features='log2', max_depth=64,
                                           min_samples_leaf=1, min_samples_split=2,
                                           random_state=rs_rf, n_jobs=n_jobs)
            model.fit(X[train_index,:], y[train_index])

            # predict test set
            preds = model.predict_proba(X[test_index,:])[:,1]

            # save
            preds_full[test_index] = preds.squeeze()

            # compute ROC curve
            fpr, tpr, _ = roc_curve(y[test_index], preds)
            fpr_folds.append(fpr)
            tpr_folds.append(tpr)

            interp_tpr = np.interp(fpr_mean, fpr, tpr)
            interp_tpr[0] = 0.0
            tpr_mean.append(interp_tpr)

            # compute AUROC
            aurocs.append(roc_auc_score(y[test_index], preds))

        fpr_full, tpr_full, _ = roc_curve(y, preds_full)
        fpr_fulls.append(fpr_full)
        tpr_fulls.append(tpr_full)
        preds_return.append(preds_full)

    aurocs = np.array(aurocs)
    tpr_mean = np.array(tpr_mean).mean(axis=0)
    preds_return = np.array(preds_return).T

    return aurocs, fpr_folds, tpr_folds, fpr_mean, tpr_mean, preds_return


def read_sad(sad_file):
    with h5py.File(sad_file) as sad_open:
        sad = np.array(sad_open['SAD'], dtype='float64')
    return sad


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
