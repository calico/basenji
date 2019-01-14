#!/usr/bin/env python
from __future__ import print_function
from optparse import OptionParser

import gc
import joblib
import pdb
import os
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, PCA
# import zarr

'''
basenji_annot_chr.py

Format Basenji SAD output for LD score analysis.
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <sad_dir>'
    parser = OptionParser(usage)
    parser.add_option('-c', dest='chroms',
            default=None, type='int',
            help='Limit to first N chromosomes')
    parser.add_option('-d', dest='decomposition_dim',
            default=None, type='int')
    parser.add_option('-m', dest='model',
            default='pca', help='Matrix factorization model [Default: %default]')
    parser.add_option('-o', dest='out_dir', default='.')
    parser.add_option('-p', dest='plink_stem',
            default='%s/popgen/1000G_EUR_Phase3_plink/1000G.EUR.QC' % os.environ['HG19'])
    parser.add_option('-r', dest='restart',
            default=False, action='store_true',
            help='Restart an interrupted job [Default: %default]')
    parser.add_option('-s', dest='sad_stat',
            default='SAD', help='SAD statistic [Default: %default]')
    parser.add_option('-t', dest='targets_file',
            default=None, help='Read only the specified targets')
    parser.add_option('-x', dest='memory_max',
            default='2B', type='str',
            help='Decomposition matrix size maximum [Default: %default]')
    parser.add_option('-u', dest='unsigned',
            default=False, action='store_true',
            help='Write unsigned annotations using .annot suffix [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 1:
        parser.error('Must provide SAD directory')
    else:
        sad_dir = args[0]

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    if options.targets_file is not None:
        targets_df = pd.read_table(options.targets_file, index_col=0)
    else:
        targets_df = None

    if options.memory_max[-1] == 'M':
        options.memory_max = int(1000000*float(options.memory_max[:-1]))
    elif options.memory_max[-1] == 'B':
        options.memory_max = int(1000000000*float(options.memory_max[:-1]))
    else:
        print('Cannot parse %s' % options.memory_max)
        exit(1)

    ############################################
    # fit dimension reduction model

    if options.decomposition_dim is not None:
        model_pkl_file = '%s/model.pkl' % options.out_dir
        if options.restart and os.path.isfile(model_pkl_file):
            model = joblib.load(model_pkl_file)

        else:
            X_sad = sample_sad(sad_dir, options.memory_max,
                               options.sad_stat, targets_df)

            if options.unsigned:
                X_sad = np.abs(X_sad)

            X_sad = X_sad.astype('float32')

            if options.model.lower() == 'pca':
                print('Computing PCA to %d' % options.decomposition_dim)
                model = PCA(n_components=options.decomposition_dim,
                            svd_solver='randomized')

            elif options.model.lower() == 'nmf':
                print('Computing NMF to %d' % options.decomposition_dim)
                model = NMF(n_components=options.decomposition_dim, init='nndsvda')

            else:
                print('Unrecognized matrix factorization model "%s"' % options.model,
                      file=sys.stderr)
                exit(1)

            model.fit(X_sad)
            joblib.dump(model, model_pkl_file)
            del X_sad


    ############################################
    # read BIM / write annotation

    if options.unsigned:
        annot_ext = 'annot'
    else:
        annot_ext = 'sannot'

    max_chr = 22
    if options.chroms is not None:
        max_chr = options.chroms

    for ci in range(1,max_chr+1):
        print('Reading chr%d BIM' % ci, flush=True)
        annot_file = '%s/sad.%s.%s' % (options.out_dir, ci, annot_ext)

        if not options.restart or not os.path.isfile(annot_file+'.gz'):
            chrom_bim = '%s.%d.bim' % (options.plink_stem, ci)

            # read SAD
            sad_chr_file = '%s/chr%d/sad.h5' % (sad_dir, ci)
            if os.path.isfile(sad_chr_file):
                sad_chr_h5 = h5py.File(sad_chr_file, 'r')

                if targets_df is None:
                    X_sad = sad_chr_h5[options.sad_stat][:,:]
                    annot_labels = [tid.decode('UTF-8') for tid in sad_chr_h5['target_ids']]
                else:
                    X_sad = sad_chr_h5[options.sad_stat][:,targets_df.index]
                    annot_labels = list(targets_df.identifier.values)

                # up convert
                X_sad = X_sad.astype('float32')

                ref_sad = [nt.decode('UTF-8') for nt in sad_chr_h5['ref']]

                # read SAD zarr
                # sad_chr_file = '%s/chr%d/sad_table.zarr' % (sad_dir, ci)
                # sad_chr_zarr = zarr.open_group(sad_chr_file, 'r')
                # X_sad = np.array(sad_chr_zarr['SAD'])

                if options.unsigned:
                    X_sad = np.abs(X_sad)

                # reduce dimension
                if options.decomposition_dim is not None:
                    X_dim = model.transform(X_sad)
                    del X_sad
                    X_sad = X_dim

                    # re-label the annotations
                    annot_labels = ['sad-%d' % (i+1) for i in range(X_sad.shape[1])]

                # initialize output file
                annot_out = open(annot_file, 'w')
                header_cols = ['CHR', 'BP', 'SNP', 'CM', 'A1', 'A2'] + annot_labels
                print('\t'.join(header_cols), file=annot_out)

                # process .bim
                si = 0
                for line in open(chrom_bim):
                    a = line.split()
                    rsid = a[1]
                    cm = a[2]
                    pos = a[3]
                    a1 = a[4]
                    a2 = a[5]

                    if ref_sad[si] == a2:
                        # take prediction
                        sad_si = X_sad[si]

                    elif ref_sad[si] == a1:
                        # flip prediction
                        sad_si = -X_sad[si]

                    else:
                        print('ERROR: %s SAD reference %s does not match Plink A1 %s or A2 %s' % \
                                (rsid, ref_sad[si], a1, a2), file=sys.stderr)
                        exit(1)

                    if options.unsigned:
                        sad_si = np.abs(sad_si)

                    cols = [str(ci), pos, rsid, cm, a1, a2]
                    cols += ['%.4f'%x for x in sad_si]
                    print('\t'.join(cols), file=annot_out)

                    si += 1

                annot_out.close()

                # compress
                subprocess.call('gzip -f %s' % annot_file, shell=True)

                # clean memory
                del X_sad
                gc.collect()


def sample_sad(sad_dir, sad_max, sad_stat='SAD', targets_df=None):
    # count SNPs and targets
    num_snps = 0
    for ci in range(1,23):
        sad_chr_file = '%s/chr%d/sad.h5' % (sad_dir, ci)
        if os.path.isfile(sad_chr_file):
            sad_chr_h5 = h5py.File(sad_chr_file, 'r')
            num_snps += sad_chr_h5[sad_stat].shape[0]
            if targets_df is None:
                num_targets = sad_chr_h5[sad_stat].shape[1]
            else:
                num_targets = len(targets_df.index)

    # compute sample %
    sample_p = sad_max / (num_snps*num_targets)

    # sample SNPs
    X_sad = None
    for ci in range(1,23):
        sad_chr_file = '%s/chr%d/sad.h5' % (sad_dir, ci)
        if os.path.isfile(sad_chr_file):
            sad_chr_h5 = h5py.File(sad_chr_file, 'r')

            # choose indexes
            chr_snps = sad_chr_h5[sad_stat].shape[0]
            sample_snps = int(chr_snps * sample_p)
            if sample_snps >= chr_snps:
                si = np.arange(chr_snps)
            else:
                si = np.random.choice(chr_snps, size=sample_snps, replace=False)
                si.sort()

            # read from HDF5
            if targets_df is None:
                X_sad_ci = sad_chr_h5[sad_stat][si,:]
            else:
                X_sad_ci = sad_chr_h5[sad_stat][si,:][:,targets_df.index]

            # save/concat
            if X_sad is None:
                X_sad = X_sad_ci
            else:
                X_sad = np.concatenate([X_sad, X_sad_ci], axis=0)

    return X_sad


def read_enough_sad(sad_dir, sad_max, sad_stat='SAD', targets_df=None):
    X_sad = None

    for ci in reversed(range(1,23)):
        sad_chr_file = '%s/chr%d/sad.h5' % (sad_dir, ci)
        if os.path.isfile(sad_chr_file):
            sad_chr_h5 = h5py.File(sad_chr_file, 'r')

            if targets_df is None:
                X_sad_ci = sad_chr_h5[sad_stat][:,:]
            else:
                X_sad_ci = sad_chr_h5[sad_stat][:,targets_df.index]

            if X_sad is None:
                X_sad = X_sad_ci
            else:
                X_sad = np.concatenate([X_sad, X_sad_ci], axis=0)

            X_len = X_sad.shape[0]*X_sad.shape[1]
            if X_len > sad_max:
                break

    # sample down below max
    variants_max = sad_max // X_sad.shape[1]
    if variants_max < X_sad.shape[0]:
        ri = np.random.choice(np.arange(X_sad.shape[0]),
                              size=variants_max, replace=False)
        X_sad = X_sad[ri]

    return X_sad


def read_enough_sad_zarr(sad_dir, sad_max):
    X_sad = None

    for ci in reversed(range(1,23)):
        sad_chr_file = '%s/chr%d/sad_table.zarr' % (sad_dir, ci)
        sad_chr_zarr = zarr.open_group(sad_chr_file, 'r')

        print('Reading SAD chr%d' % ci)
        if X_sad is None:
            X_sad = np.array(sad_chr_zarr['SAD'])
        else:
            X_sad = np.concatenate([X_sad, np.array(sad_chr_zarr['SAD'])], axis=0)

        X_len = X_sad.shape[0]*X_sad.shape[1]
        if X_len > sad_max:
            break

    return X_sad


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
