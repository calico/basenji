#!/usr/bin/env python
from optparse import OptionParser
import joblib
import os
import pdb

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
    usage = 'usage: %prog [options] <scores_file>'
    parser = OptionParser(usage)
    parser.add_option('-d', dest='n_components',
            default=None, type='int',
            help='PCA n_components [Default: %default]')
    parser.add_option('-e', dest='num_estimators',
            default=100, type='int',
            help='Number of random forest estimators [Default: %default]')
    parser.add_option('-g', dest='genome',
            default='ce11', help='PhyloP and FASTA genome [Default: %default]')
    parser.add_option('-i', dest='iterations',
            default=1, type='int',
            help='Cross-validation iterations [Default: %default]')
    parser.add_option('-o', dest='out_dir',
            default='regr_out')
    parser.add_option('-p', dest='parallel_threads',
            default=1, type='int',
            help='Parallel threads passed to scikit-learn n_jobs [Default: %default]')
    parser.add_option('-r', dest='random_seed',
            default=44, type='int')
    parser.add_option('--stat', dest='sad_stat',
            default='sum',
            help='HDF5 key stat to consider. [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 1:
        parser.error('Must provide ISM scores and PhyloP bigwig file.')
    else:
        scores_file = args[0]

    np.random.seed(options.random_seed)
    options.genome = options.genome.lower()

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    ################################################################
    # read ISM scores

    with h5py.File(scores_file, 'r') as h5o:
        score_chrs = [chrm.decode('UTF-8') for chrm in h5o['chr']]
        score_starts = h5o['start'][:]
        score_ends = h5o['end'][:]
        score_strands = [strand.decode('UTF-8') for strand in h5o['strand']]
        score_seqs = h5o['seqs'][:]
        nt_scores = h5o[options.sad_stat][:].astype('float32')
    num_seqs, mut_len, _, num_targets = nt_scores.shape

    # reference transform
    nt_scores_ref = np.reshape(nt_scores[score_seqs], (num_seqs, mut_len, num_targets))

    # min/max transform
    nt_scores_min = nt_scores.min(axis=-2)
    nt_scores_max = nt_scores.max(axis=-2)
    pos_mask = (nt_scores_ref > 0)
    nt_scores_refm = nt_scores_ref.copy()
    nt_scores_refm[pos_mask] -= nt_scores_min[pos_mask]
    nt_scores_refm[~pos_mask] -= nt_scores_max[~pos_mask]

    ################################################################
    # read phylop bigwig annotations

    genome_path = os.environ[options.genome.upper()]
    fasta_file = '%s/assembly/%s.fa' % (genome_path, options.genome)
    if options.genome == 'ce11':
        phylop_file = '%s/phylop/ce11.phyloP26way.bw' % genome_path
    else:
        print('Genome PhyloP not found', file=sys.stderr)
        exit(1)

    seqs_phylop = []
    seqs_phylop_dna1 = []

    fasta_open = pysam.FastaFile(fasta_file)
    phylop_open = pyBigWig.open(phylop_file, 'r')

    for si in range(num_seqs):
        phylop_chr = score_chrs[si]
        if not phylop_chr.startswith('chr'):
            phylop_chr = 'chr%s' % phylop_chr

        # read values
        seq_phylop = phylop_open.values(phylop_chr, score_starts[si], score_ends[si],
            numpy=True).astype('float32')

        # read DNA
        seq_phylop_dna = fasta_open.fetch(score_chrs[si], score_starts[si], score_ends[si])
        seq_phylop_dna1 = dna_io.dna_1hot(seq_phylop_dna)

        # reverse complement
        if score_strands[si] == '-':            
            seq_phylop = seq_phylop[::-1]
            seq_phylop_dna1 = dna_io.hot1_rc(seq_phylop_dna1)

        # save
        seqs_phylop.append(seq_phylop)
        seqs_phylop_dna1.append(seq_phylop_dna1)

    # transform PhyloP
    seqs_phylop = np.array(seqs_phylop)
    seqs_phylop = np.nan_to_num(seqs_phylop)
    seqs_phylop = np.clip(seqs_phylop, -1.5, 5)

    # verify DNA
    seqs_phylop_dna1 = np.array(seqs_phylop_dna1)
    for si in range(num_seqs):
        seq_diff = np.logical_xor(score_seqs[si], seqs_phylop_dna1[si])
        nts_diff = seq_diff.sum() // 2
        if nts_diff != 0:
            pdb.set_trace()

    ################################################################
    # regression

    # add positions
    seqs_pos = np.arange(mut_len)
    seqs_pos = np.tile(seqs_pos, num_seqs)
    seqs_pos = np.reshape(seqs_pos, (num_seqs,-1,1))
    
    # flatten everything
    # seqs_phylop_flat = seqs_phylop.flatten()
    # seqs_pos_flat = seqs_pos.flatten()
    # nt_scores_refm_flat = nt_scores_refm.reshape((-1,num_targets))
    # num_pos = nt_scores_refm_flat.shape[0]

    # form matrix
    # X_scores = nt_scores_refm_flat
    # if options.n_components is not None:
    #     options.n_components = min(options.n_components, num_targets)
    #     X_scores = PCA(options.n_components).fit_transform(nt_scores_refm_flat)
    # X_pos = seqs_pos.reshape(num_pos,1)
    # X = np.concatenate([X_scores,X_pos], axis=1)

    X = np.concatenate([nt_scores_refm,seqs_pos], axis=-1)

    # regressor
    r2s, pcors = randfor_cv(X, seqs_phylop,
        iterations=options.iterations,
        n_estimators=options.num_estimators,
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


def randfor_cv(Xs, ys, folds=8, iterations=1, n_estimators=50,
               max_features='log2', random_state=44, n_jobs=8):
    """Compute random forest regression accuracy statistics, shuffling at the sequence level."""
    r2s = []
    pcors = []

    for i in range(iterations):
        rs_iter = random_state + i

        kf = KFold(n_splits=folds, shuffle=True, random_state=rs_iter)

        for train_index, test_index in kf.split(Xs):
            num_seqs, num_pos, num_feat = Xs.shape
            X_train = Xs[train_index].reshape((-1,num_feat))
            y_train = ys[train_index].flatten()
            X_test = Xs[test_index].reshape((-1,num_feat))
            y_test = ys[test_index].flatten()
                        
            # fit model
            if random_state is None:
                rs_rf = None
            else:
                rs_rf = rs_iter+test_index[0]
            model = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features,
                                          max_depth=64, min_samples_leaf=1, min_samples_split=2,
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
