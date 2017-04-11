#!/usr/bin/env python
from optparse import OptionParser
import os
import random
import re

import h5py
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
import tensorflow as tf

import basenji
import fdr

################################################################################
# basenji_test_reps.py
#
################################################################################

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <params_file> <model_file> <data_file> <samples_file>'
    parser = OptionParser(usage)
    parser.add_option('-d', dest='down_sample', default=1, type='int', help='Down sample test computation by taking uniformly spaced positions [Default: %default]')
    parser.add_option('-o', dest='out_dir', default='test_out', help='Output directory for test statistics [Default: %default]')
    parser.add_option('--rc', dest='rc', default=False, action='store_true', help='Average the forward and reverse complement predictions when testing [Default: %default]')
    parser.add_option('-p', dest='plot_pct', default=1.0, type='float', help='Proportion of plots to make [Default: %default]')
    parser.add_option('-s', dest='scatter_plots', default=False, action='store_true', help='Make scatter plots [Default: %default]')
    parser.add_option('-v', dest='valid', default=False, action='store_true', help='Process the validation set [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 4:
    	parser.error('Must provide parameters, model, test data HDF5, and samples file')
    else:
        params_file = args[0]
        model_file = args[1]
        data_file = args[2]
        samples_file = args[3]

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    random.seed(1)

    #######################################################
    # load data
    #######################################################
    data_open = h5py.File(data_file)

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

    target_labels_long = []
    for line in open(samples_file):
        a = line.split('\t')
        a[-1] = a[-1].rstrip()
        target_labels_long.append(a[2])


    #######################################################
    # model parameters and placeholders
    #######################################################
    job = basenji.dna_io.read_job_params(params_file)

    job['batch_length'] = test_seqs.shape[1]
    job['seq_depth'] = test_seqs.shape[2]
    job['num_targets'] = test_targets.shape[2]
    job['target_pool'] = int(np.array(data_open.get('pool_width', 1)))

    dr = basenji.rnn.RNN()
    dr.build(job)

    # adjust for fourier
    job['fourier'] = 'train_out_imag' in data_open
    if job['fourier']:
        test_targets_imag = data_open['test_out_imag']
        if options.valid:
            test_targets_imag = data_open['valid_out_imag']


    #######################################################
    # predict w/ model

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
        test_loss, test_r2, test_cor, test_preds = dr.test(sess, batcher_test, rc_avg=options.rc, return_preds=True, down_sample=options.down_sample)


    #######################################################
    # compute replicate correlations

    # determine replicates
    replicate_lists = infer_replicates(target_labels_long)
    replicate_labels = sorted(replicate_lists.keys())

    reps_out = open('%s/replicates.txt' % options.out_dir, 'w')
    for label in replicate_labels:
        if len(replicate_lists[label]) > 1:
            ti1 = replicate_lists[label][0]
            ti2 = replicate_lists[label][1]
            print(ti1, ti2, label, file=reps_out)
    reps_out.close()

    sns.set(font_scale=1.3, style='ticks')

    table_out = open('%s/table.txt' % options.out_dir, 'w')

    # sample every few bins (adjust to plot the # points I want)
    ds_indexes_preds = np.arange(0, test_preds.shape[1], 8)
    ds_indexes_targets = ds_indexes_preds + (dr.batch_buffer // dr.target_pool)

    cor_reps = []
    cor_preds = []

    li = 0
    for label in replicate_labels:
        if len(replicate_lists[label]) > 1:
            ti1 = replicate_lists[label][0]
            ti2 = replicate_lists[label][1]

            test_targets_rep1 = test_targets[:,ds_indexes_targets,ti1].flatten().astype('float32')
            test_targets_rep2 = test_targets[:,ds_indexes_targets,ti2].flatten().astype('float32')

            test_preds_rep1 = test_preds[:,ds_indexes_preds,ti1].flatten().astype('float32')
            test_preds_rep2 = test_preds[:,ds_indexes_preds,ti2].flatten().astype('float32')

            # take logs
            test_targets_rep1 = np.log2(test_targets_rep1+1)
            test_targets_rep2 = np.log2(test_targets_rep2+1)
            test_preds_rep1 = np.log2(test_preds_rep1+1)
            test_preds_rep2 = np.log2(test_preds_rep2+1)

            # decide whether to plot it
            plot_label = options.scatter_plots and random.random() < options.plot_pct

            #####################################
            # replicate

            # compute replicate correlation
            #scor, _ = spearmanr(test_targets_rep1, test_targets_rep2)
            corr, _ = pearsonr(test_targets_rep1, test_targets_rep2)
            cor_reps.append(corr)

            # scatter plot rep vs rep
            if plot_label:
                out_pdf = '%s/reps_s%d.pdf' % (options.out_dir,li)
                regplot(test_targets_rep1, test_targets_rep2, out_pdf, poly_order=1, alpha=0.3, x_label='log2 Replicate 1', y_label='log2 Replicate 2')

            #####################################
            # prediction

            # save prediction correlation
            # cor_preds.append((test_cor[ti1]+test_cor[ti2])/2)
            corr1, _ = pearsonr(test_targets_rep1, test_preds_rep1)
            corr2, _ = pearsonr(test_targets_rep2, test_preds_rep2)
            cor_preds.append((corr1+corr2)/2)

            if plot_label:
                # scatter plot rep vs pred
                out_pdf = '%s/preds_s%d_rep1.pdf' % (options.out_dir,li)
                regplot(test_targets_rep1, test_preds_rep1, out_pdf, poly_order=1, alpha=0.3, x_label='log2 Replicate 1', y_label='log2 Prediction 1')

                # scatter plot rep vs pred
                out_pdf = '%s/preds_s%d_rep2.pdf' % (options.out_dir,li)
                regplot(test_targets_rep2, test_preds_rep2, out_pdf, poly_order=1, alpha=0.3, x_label='log2 Replicate 2', y_label='log2 Prediction 2')

            #####################################
            # table

            print('%4d  %4d  %4d  %7.4f  %7.4f  %s' % (li, ti1, ti2, cor_reps[-1], cor_preds[-1], label), file=table_out)

            # update counter
            li += 1

    table_out.close()

    cor_reps = np.array(cor_reps)
    cor_preds = np.array(cor_preds)


    #######################################################
    # scatter plot replicate versus prediction correlation

    out_pdf = '%s/correlation.pdf' % options.out_dir
    jointplot(cor_reps, cor_preds, out_pdf, x_label='Replicates', y_label='Predictions')

    data_open.close()


def infer_replicates(target_labels_long):
    ''' Infer replicate experiments based on their long form labels.

    In:
        target_labels_long [str]: list of long form target labels
    Out:
        replicate_lists {exp_label -> [target indexes]}
    '''
    replicate_lists = {}

    rep_re = []
    rep_re.append(re.compile('rep\d+'))
    rep_re.append(re.compile('donor\d+'))

    for ti in range(len(target_labels_long)):
        label = target_labels_long[ti]

        for ri in range(len(rep_re)):
            rep_m = rep_re[ri].search(label)
            if rep_m:
                rep_str = rep_m.group(0)
                label = label.replace(rep_str,'')

        replicate_lists.setdefault(label,[]).append(ti)

    return replicate_lists


def jointplot(vals1, vals2, out_pdf, alpha=0.5, x_label=None, y_label=None):
    plt.figure((3,3))

    g = sns.jointplot(vals1, vals2, alpha=alpha, color='black', stat_func=None)

    ax = g.ax_joint
    vmin, vmax = basenji.plots.scatter_lims(vals1, vals2)
    ax.plot([vmin,vmax], [vmin,vmax], linestyle='--', color='black')

    ax.set_xlim(vmin,vmax)
    ax.set_xlabel(x_label)
    ax.set_ylim(vmin,vmax)
    ax.set_ylabel(y_label)

    lim_eps = .02 * (vmax - vmin)
    ax.text(vmax-lim_eps, vmin+lim_eps, 'mean PearsonR %.3f'%vals1.mean(), horizontalalignment='right', fontsize=8)
    ax.text(vmin+lim_eps, vmax-3*lim_eps, 'mean PearsonR %.3f'%vals2.mean(), horizontalalignment='left', fontsize=8)

    # ax.grid(True, linestyle=':')
    # plt.tight_layout(w_pad=0, h_pad=0)

    plt.savefig(out_pdf)
    plt.close()


def regplot(vals1, vals2, out_pdf, poly_order=1, alpha=0.5, x_label=None, y_label=None):
    plt.figure(figsize=(6,6))

    # g = sns.jointplot(vals1, vals2, alpha=0.5, color='black', stat_func=spearmanr)
    gold = sns.color_palette('husl',8)[1]
    ax = sns.regplot(vals1, vals2, color='black', order=poly_order, scatter_kws={'color':'black', 's':4, 'alpha':alpha}, line_kws={'color':gold})

    vmin, vmax = basenji.plots.scatter_lims(vals1, vals2)
    # ax.plot([vmin,vmax], [vmin,vmax], linestyle='--', color='black')

    ax.set_xlim(vmin,vmax)
    if x_label is not None:
        ax.set_xlabel(x_label)
    ax.set_ylim(vmin,vmax)
    if y_label is not None:
        ax.set_ylabel(y_label)

    # corr, _ = spearmanr(vals1, vals2)
    corr, _ = pearsonr(vals1, vals2)
    lim_eps = (vmax-vmin) * .02
    ax.text(vmin+lim_eps, vmax-3*lim_eps, 'PearsonR: %.3f'%corr, horizontalalignment='left', fontsize=12)

    ax.grid(True, linestyle=':')

    # plt.tight_layout(w_pad=0, h_pad=0)

    plt.savefig(out_pdf)
    plt.close()


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
