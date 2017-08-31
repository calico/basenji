#!/usr/bin/env python
# Copyright 2017 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from __future__ import print_function
from optparse import OptionParser
import os
import random
import sys
import time

import h5py
import joblib
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyBigWig
from scipy.stats import spearmanr, poisson
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import tensorflow as tf

import basenji

'''
basenji_test.py

Test the accuracy of a trained model.

Notes
 -This probably needs work for the pooled large sequence version. I tried to
  update the "full" comparison, but it's not tested. The notion of peak calls
  will need to completely change; we probably want to predict in each bin.
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <params_file> <model_file> <test_hdf5_file>'
    parser = OptionParser(usage)
    parser.add_option('--ai', dest='accuracy_indexes', help='Comma-separated list of target indexes to make accuracy plots comparing true versus predicted values')
    parser.add_option('-d', dest='down_sample', default=1, type='int', help='Down sample test computation by taking uniformly spaced positions [Default: %default]')
    parser.add_option('-g', dest='genome_file', default='%s/assembly/human.hg19.genome'%os.environ['HG19'], help='Chromosome length information [Default: %default]')
    parser.add_option('--mc', dest='mc_n', default=0, type='int', help='Monte carlo test iterations [Default: %default]')
    parser.add_option('-o', dest='out_dir', default='test_out', help='Output directory for test statistics [Default: %default]')
    parser.add_option('--rc', dest='rc', default=False, action='store_true', help='Average the forward and reverse complement predictions when testing [Default: %default]')
    parser.add_option('-s', dest='scent_file', help='Dimension reduction model file')
    parser.add_option('--save', dest='save', default=False, action='store_true')
    parser.add_option('--shifts', dest='shifts', default='0', help='Ensemble prediction shifts [Default: %default]')
    parser.add_option('-t', dest='track_bed', help='BED file describing regions so we can output BigWig tracks')
    parser.add_option('--ti', dest='track_indexes', help='Comma-separated list of target indexes to output BigWig tracks')
    parser.add_option('-v', dest='valid', default=False, action='store_true', help='Process the validation set [Default: %default]')
    parser.add_option('-w', dest='pool_width', default=1, type='int', help='Max pool width for regressing nucleotide predictions to predict peak calls [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 3:
    	parser.error('Must provide parameters, model, and test data HDF5')
    else:
        params_file = args[0]
        model_file = args[1]
        test_hdf5_file = args[2]

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    options.shifts = [int(shift) for shift in options.shifts.split(',')]

    #######################################################
    # load data
    #######################################################
    data_open = h5py.File(test_hdf5_file)

    if not options.valid:
        test_seqs = data_open['test_in']
        test_targets = data_open['test_out']
        test_na = None
        if 'test_na' in data_open:
            test_na = data_open['test_na']

    else:
        test_seqs = data_open['valid_in']
        test_targets = data_open['valid_out']
        test_na = None
        if 'test_na' in data_open:
            test_na = data_open['valid_na']

    target_labels = [tl.decode('UTF-8') for tl in data_open['target_labels']]

    #######################################################
    # model parameters and placeholders

    job = basenji.dna_io.read_job_params(params_file)

    job['batch_length'] = test_seqs.shape[1]
    job['seq_depth'] = test_seqs.shape[2]
    job['num_targets'] = test_targets.shape[2]
    job['target_pool'] = int(np.array(data_open.get('pool_width', 1)))

    t0 = time.time()
    dr = basenji.seqnn.SeqNN()
    dr.build(job)
    print('Model building time %ds' % (time.time()-t0))

    # adjust for fourier
    job['fourier'] = 'train_out_imag' in data_open
    if job['fourier']:
        test_targets_imag = data_open['test_out_imag']
        if options.valid:
            test_targets_imag = data_open['valid_out_imag']

    # adjust for factors
    if options.scent_file is not None:
        t0 = time.time()
        test_targets_full = data_open['test_out_full']
        model = joblib.load(options.scent_file)

    #######################################################
    # test

    # initialize batcher
    if job['fourier']:
        batcher_test = basenji.batcher.BatcherF(test_seqs, test_targets, test_targets_imag, test_na, dr.batch_size, dr.target_pool)
    else:
        batcher_test = basenji.batcher.Batcher(test_seqs, test_targets, test_na, dr.batch_size, dr.target_pool)

    # initialize saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # load variables into session
        saver.restore(sess, model_file)

        # test
        t0 = time.time()
        # test_loss, test_r2, test_cor, test_preds = dr.test(sess, batcher_test, rc_avg=options.rc, return_preds=True, down_sample=options.down_sample)
        # test_acc = dr.test(sess, batcher_test, rc_avg=options.rc, down_sample=options.down_sample)
        test_acc = dr.test(sess, batcher_test, rc=options.rc, shifts=options.shifts, mc_n=options.mc_n)

        if options.save:
            np.save('%s/preds.npy' % options.out_dir, test_acc.preds)
            np.save('%s/targets.npy' % options.out_dir, test_acc.targets)

        test_preds = test_acc.preds
        print('SeqNN test: %ds' % (time.time()-t0))

        # compute stats
        test_r2 = test_acc.r2()
        test_log_r2 = test_acc.r2(log=True)
        test_cor = test_acc.spearmanr()

        # print
        print('Test Loss:      %7.5f' % test_acc.loss)
        print('Test R2:        %7.5f' % test_r2.mean())
        print('Test log R2:    %7.5f' % test_log_r2.mean())
        print('Test SpearmanR: %7.5f' % test_cor.mean())

        acc_out = open('%s/acc.txt' % options.out_dir, 'w')
        for ti in range(len(test_r2)):
            print('%4d  %7.5f  %.5f  %.5f  %.5f  %s' % (ti, test_acc.target_losses[ti], test_r2[ti], test_log_r2[ti], test_cor[ti], target_labels[ti]), file=acc_out)
        acc_out.close()

        # clean up
        del test_acc

        # if test targets are reconstructed, measure versus the truth
        if options.scent_file is not None:
            compute_full_accuracy(dr, model, test_preds, test_targets_full, options.out_dir, options.down_sample)

    #######################################################
    # peak call accuracy

    # sample every few bins to decrease correlations
    ds_indexes_preds = np.arange(0, test_preds.shape[1], 8)
    ds_indexes_targets = ds_indexes_preds + (dr.batch_buffer // dr.target_pool)

    aurocs = []
    auprcs = []

    peaks_out = open('%s/peaks.txt' % options.out_dir, 'w')
    for ti in range(test_targets.shape[2]):
        if options.scent_file is not None:
            test_targets_ti = test_targets_full[:,:,ti]
        else:
            test_targets_ti = test_targets[:,:,ti]

        # subset and flatten
        test_targets_ti_flat = test_targets_ti[:,ds_indexes_targets].flatten().astype('float32')
        test_preds_ti_flat = test_preds[:,ds_indexes_preds,ti].flatten().astype('float32')

        # call peaks
        test_targets_ti_lambda = np.mean(test_targets_ti_flat)
        test_targets_pvals = 1 - poisson.cdf(np.round(test_targets_ti_flat)-1, mu=test_targets_ti_lambda)
        test_targets_qvals = np.array(ben_hoch(test_targets_pvals))
        test_targets_peaks = test_targets_qvals < 0.01

        if test_targets_peaks.sum() == 0:
            aurocs.append(0.5)
            auprcs.append(0)

        else:
            # compute prediction accuracy
            aurocs.append(roc_auc_score(test_targets_peaks, test_preds_ti_flat))
            auprcs.append(average_precision_score(test_targets_peaks, test_preds_ti_flat))

        print('%4d  %6d  %.5f  %.5f' % (ti,test_targets_peaks.sum(), aurocs[-1], auprcs[-1]), file=peaks_out)

    peaks_out.close()

    print('Test AUROC:     %7.5f' % np.mean(aurocs))
    print('Test AUPRC:     %7.5f' % np.mean(auprcs))


    #######################################################
    # BigWig tracks

    # NOTE: THESE ASSUME THERE WAS NO DOWN-SAMPLING ABOVE

    # print bigwig tracks for visualization
    if options.track_bed:
        if options.genome_file is None:
            parser.error('Must provide genome file in order to print valid BigWigs')

        if not os.path.isdir('%s/tracks' % options.out_dir):
            os.mkdir('%s/tracks' % options.out_dir)

        track_indexes = range(test_preds.shape[2])
        if options.track_indexes:
            track_indexes = [int(ti) for ti in options.track_indexes.split(',')]

        bed_set = 'test'
        if options.valid:
            bed_set = 'valid'

        for ti in track_indexes:
            if options.scent_file is not None:
                test_targets_ti = test_targets_full[:,:,ti]
            else:
                test_targets_ti = test_targets[:,:,ti]

            # make true targets bigwig
            bw_file = '%s/tracks/t%d_true.bw' % (options.out_dir,ti)
            bigwig_write(bw_file, test_targets_ti, options.track_bed, options.genome_file, bed_set=bed_set)

            # make predictions bigwig
            bw_file = '%s/tracks/t%d_preds.bw' % (options.out_dir,ti)
            bigwig_write(bw_file, test_preds[:,:,ti], options.track_bed, options.genome_file, dr.batch_buffer, bed_set=bed_set)

        # make NA bigwig
        bw_file = '%s/tracks/na.bw' % options.out_dir
        bigwig_write(bw_file, test_na, options.track_bed, options.genome_file, bed_set=bed_set)


    #######################################################
    # accuracy plots

    if options.accuracy_indexes is not None:
        accuracy_indexes = [int(ti) for ti in options.accuracy_indexes.split(',')]

        if not os.path.isdir('%s/scatter' % options.out_dir):
            os.mkdir('%s/scatter' % options.out_dir)

        if not os.path.isdir('%s/violin' % options.out_dir):
            os.mkdir('%s/violin' % options.out_dir)

        if not os.path.isdir('%s/roc' % options.out_dir):
            os.mkdir('%s/roc' % options.out_dir)

        if not os.path.isdir('%s/pr' % options.out_dir):
            os.mkdir('%s/pr' % options.out_dir)

        for ti in accuracy_indexes:
            if options.scent_file is not None:
                test_targets_ti = test_targets_full[:,:,ti]
            else:
                test_targets_ti = test_targets[:,:,ti]

            ############################################
            # scatter

            # sample every few bins (adjust to plot the # points I want)
            ds_indexes_preds = np.arange(0, test_preds.shape[1], 8)
            ds_indexes_targets = ds_indexes_preds + (dr.batch_buffer // dr.target_pool)

            # subset and flatten
            test_targets_ti_flat = test_targets_ti[:,ds_indexes_targets].flatten().astype('float32')
            test_preds_ti_flat = test_preds[:,ds_indexes_preds,ti].flatten().astype('float32')

            # take log2
            test_targets_ti_log = np.log2(test_targets_ti_flat+1)
            test_preds_ti_log = np.log2(test_preds_ti_flat+1)

            # plot log2
            sns.set(font_scale=1.2, style='ticks')
            out_pdf = '%s/scatter/t%d.pdf' % (options.out_dir,ti)
            basenji.plots.regplot(test_targets_ti_log, test_preds_ti_log, out_pdf, poly_order=1, alpha=0.3, sample=500, figsize=(6,6), x_label='log2 Experiment', y_label='log2 Prediction', table=True)

            ############################################
            # violin

            # call peaks
            test_targets_ti_lambda = np.mean(test_targets_ti_flat)
            test_targets_pvals = 1 - poisson.cdf(np.round(test_targets_ti_flat)-1, mu=test_targets_ti_lambda)
            test_targets_qvals = np.array(ben_hoch(test_targets_pvals))
            test_targets_peaks = test_targets_qvals < 0.01
            test_targets_peaks_str = np.where(test_targets_peaks, 'Peak', 'Background')

            # violin plot
            sns.set(font_scale=1.3, style='ticks')
            plt.figure()
            df = pd.DataFrame({'log2 Prediction':np.log2(test_preds_ti_flat+1), 'Experimental coverage status':test_targets_peaks_str})
            ax = sns.violinplot(x='Experimental coverage status', y='log2 Prediction', data=df)
            ax.grid(True, linestyle=':')
            plt.savefig('%s/violin/t%d.pdf' % (options.out_dir,ti))
            plt.close()

            # ROC
            plt.figure()
            fpr, tpr, _ = roc_curve(test_targets_peaks, test_preds_ti_flat)
            auroc = roc_auc_score(test_targets_peaks, test_preds_ti_flat)
            plt.plot([0,1], [0,1], c='black', linewidth=1, linestyle='--', alpha=0.7)
            plt.plot(fpr, tpr, c='black')
            ax = plt.gca()
            ax.set_xlabel('False positive rate')
            ax.set_ylabel('True positive rate')
            ax.text(0.99, 0.02, 'AUROC %.3f'%auroc, horizontalalignment='right') # , fontsize=14)
            ax.grid(True, linestyle=':')
            plt.savefig('%s/roc/t%d.pdf' % (options.out_dir,ti))
            plt.close()

            # PR
            plt.figure()
            prec, recall, _ = precision_recall_curve(test_targets_peaks, test_preds_ti_flat)
            auprc = average_precision_score(test_targets_peaks, test_preds_ti_flat)
            plt.axhline(y=test_targets_peaks.mean(), c='black', linewidth=1, linestyle='--', alpha=0.7)
            plt.plot(recall, prec, c='black')
            ax = plt.gca()
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.text(0.99, 0.95, 'AUPRC %.3f'%auprc, horizontalalignment='right') # , fontsize=14)
            ax.grid(True, linestyle=':')
            plt.savefig('%s/pr/t%d.pdf' % (options.out_dir,ti))
            plt.close()


    data_open.close()


def ben_hoch(p_values):
    ''' Convert the given p-values to q-values using Benjamini-Hochberg FDR. '''
    m = len(p_values)

    # attach original indexes to p-values
    p_k = [(p_values[k],k) for k in range(m)]

    # sort by p-value
    p_k.sort()

    # compute q-value and attach original index to front
    k_q = [(p_k[i][1],p_k[i][0]*m//(i+1)) for i in range(m)]

    # re-sort by original index
    k_q.sort()

    # drop original indexes
    q_values = [k_q[k][1] for k in range(m)]

    return q_values


def bigwig_open(bw_file, genome_file):
    ''' Open the bigwig file for writing and write the header. '''

    bw_out = pyBigWig.open(bw_file, 'w')

    chrom_sizes = []
    for line in open(genome_file):
        a = line.split()
        chrom_sizes.append((a[0],int(a[1])))

    bw_out.addHeader(chrom_sizes)

    return bw_out


def bigwig_write(bw_file, signal_ti, track_bed, genome_file, buffer=0, bed_set='test'):
    ''' Write a signal track to a BigWig file over the regions
         specified by track_bed.

    Args
     bw_file:     BigWig filename
     signal_ti:   Sequences X Length array for some target
     track_bed:   BED file specifying sequence coordinates
     genome_file: Chromosome lengths file
     buffer:      Length skipped on each side of the region.
    '''

    bw_out = bigwig_open(bw_file, genome_file)

    si = 0
    bw_entries = []

    # set entries
    for line in open(track_bed):
        a = line.split()
        if a[3] == bed_set:
            chrom = a[0]
            start = int(a[1])
            end = int(a[2])

            preds_pool = (end - start - 2*buffer) // signal_ti.shape[1]

            bw_start = start + buffer
            for li in range(signal_ti.shape[1]):
                bw_end = bw_start + preds_pool
                bw_entries.append((chrom,bw_start,bw_end,signal_ti[si,li]))
                bw_start = bw_end

            si += 1

    # sort entries
    bw_entries.sort()

    # add entries
    for line in open(genome_file):
        chrom = line.split()[0]

        bw_entries_chroms = [be[0] for be in bw_entries if be[0] == chrom]
        bw_entries_starts = [be[1] for be in bw_entries if be[0] == chrom]
        bw_entries_ends = [be[2] for be in bw_entries if be[0] == chrom]
        bw_entries_values = [float(be[3]) for be in bw_entries if be[0] == chrom]

        if len(bw_entries_chroms) > 0:
            bw_out.addEntries(bw_entries_chroms, bw_entries_starts, ends=bw_entries_ends, values=bw_entries_values)

    bw_out.close()


def compute_full_accuracy(dr, model, test_preds, test_targets_full, out_dir, down_sample):
    ''' Compute accuracy on the saved full target set, as opposed to a
         reconstructed version via dim reduction and/or fourier. '''

    full_targets = test_targets_full.shape[2]

    # determine non-buffer region
    buf_start = dr.batch_buffer // dr.target_pool
    buf_end = (dr.batch_length - dr.batch_buffer) // dr.target_pool
    buf_len = buf_end - buf_start

    # uniformly sample indexes
    ds_indexes = np.arange(0, buf_len, options.down_sample)

    # filter down full test targets
    test_targets_full_ds = test_targets_full[:,buf_start+ds_indexes,:]
    test_na_ds = test_na[:,buf_start+ds_indexes]

    # inverse transform in length batches
    t0 = time.time()
    test_preds_full = np.zeros((test_preds.shape[0], test_preds.shape[1], full_targets), dtype='float16')
    for li in range(test_preds.shape[1]):
        test_preds_full[:,li,:] = model.inverse_transform(test_preds[:,li,:])
    print('PCA transform: %ds' % (time.time()-t0))

    print(test_preds_full.shape)
    print(test_targets_full_ds.shape)

    # compute R2 by target
    t0 = time.time()
    test_r2_full = np.zeros(full_targets)
    for ti in range(full_targets):
        # flatten
        # preds_ti = test_preds_full[:,:,ti].flatten()
        # targets_ti = test_targets_full_ds[:,:,ti].flatten()
        preds_ti = test_preds_full[np.logical_not(test_na_ds),ti]
        targets_ti = test_targets_full_ds[np.logical_not(test_na_ds),ti]

        # compute R2
        tmean = targets_ti.mean(dtype='float64')
        tvar = (targets_ti-tmean).var(dtype='float64')
        pvar = (targets_ti-preds_ti).var(dtype='float64')
        test_r2_full[ti] = 1.0 - pvar/tvar
    print('Compute full R2: %d' % (time.time()-t0))

    print('Test full R2: %7.5f' % test_r2_full.mean())

    acc_out = open('%s/acc_full.txt' % out_dir, 'w')
    for ti in range(len(test_r2_full)):
        print('%4d  %.4f' % (ti,test_r2_full[ti]), file=acc_out)
    acc_out.close()


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
