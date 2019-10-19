#!/usr/bin/env python
from optparse import OptionParser

import glob
import pdb
import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

'''
Name

Description...
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <cmd_dir1> ...'
    parser = OptionParser(usage)
    parser.add_option('-o', dest='out_dir', default='cmp_out')
    (options,args) = parser.parse_args()

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    ################################################
    # read training
    
    train_dfs = {}
    test_dfs = {}

    for cmp_dir in args:
        cmp_label = os.path.split(cmp_dir)[-1]
    
        train_dfs[cmp_label] = []
        for train_file in glob.glob('%s/*/train.out' % cmp_dir):
            df_train = read_training(train_file)
            train_dfs[cmp_label].append(df_train)

        test_dfs[cmp_label] = []
        for test_file in glob.glob('%s/*/test_out/acc.txt' % cmp_dir):
            df_test = pd.read_csv(test_file, sep='\t')
            test_dfs[cmp_label].append(df_test)


    ################################################
    # training curves

    sns.set(font_scale=1.2, style='ticks')

    plot_curve(train_dfs, 'loss', options.out_dir)
    plot_curve(train_dfs, 'r', options.out_dir)

    ################################################
    # read test

    cmp_labels = sorted(test_dfs.keys())

    test_r_list = []
    labels_list = []

    for i in range(len(test_dfs)):
        label_i = cmp_labels[i]

        test_i_r = [test_df.pearsonr.mean() for test_df in test_dfs[label_i]]
        test_r_list += test_i_r
        labels_list += [label_i]*len(test_i_r)

        for j in range(i+1, len(test_dfs)):
            label_j = cmp_labels[j]

            test_j_r = [test_df.pearsonr.mean() for test_df in test_dfs[label_j]]

            mw_z, mw_p = mannwhitneyu(test_i_r, test_j_r)

            print(label_i, label_j)
            print('%.4f  %.4f  %.5f' % (np.mean(test_i_r), np.mean(test_j_r), mw_p))
            print('')

    test_dist_df = pd.DataFrame({
        'label':labels_list,
        'pearsonr':test_r_list
        })
    plt.figure()
    sns.swarmplot(x='label', y='pearsonr', data=test_dist_df, size=8)
    plt.tight_layout()
    plt.savefig('%s/swarm.pdf' % options.out_dir)
    plt.close()

    plt.figure()
    sns.violinplot(x='label', y='pearsonr', data=test_dist_df, inner='stick')
    plt.tight_layout()
    plt.savefig('%s/violin.pdf' % options.out_dir)
    plt.close()


def plot_curve(cmp_dfs, stat, out_dir):
    colors = sns.color_palette("Paired", n_colors=2*len(cmp_dfs))

    train_stat = 'train_%s' % stat
    valid_stat = 'valid_%s' % stat

    plt.figure()

    for ci, cmp_label in enumerate(cmp_dfs.keys()):
        for rep_i in range(len(cmp_dfs[cmp_label])):
            df_train = cmp_dfs[cmp_label][rep_i]

            if rep_i == 0:
                train_label = '%s train' % cmp_label
                valid_label = '%s valid' % cmp_label
            else:
                train_label = None
                valid_label = None

            plt.plot(np.arange(df_train.shape[0]), df_train[train_stat],
                     label=train_label,
                     color=colors[2*ci], alpha=0.7)
            plt.plot(np.arange(df_train.shape[0]), df_train[valid_stat],
                     label=valid_label,
                     color=colors[2*ci+1], alpha=0.7)

    ax = plt.gca()
    ax.set_xlabel('Epoch')
    ax.set_ylabel(stat)
    plt.legend()
    sns.despine()
    plt.tight_layout()
    plt.savefig('%s/curve_%s.pdf' % (out_dir,stat))
    plt.close()


def read_training(train_file):
    train_loss = []
    train_r = []
    valid_loss = []
    valid_r = []

    for line in open(train_file):
        if line.find('val_loss:') != -1:
            estats = epoch_stats(line)

            train_loss.append(estats['loss'])
            train_r.append(estats['pearsonr'])
            valid_loss.append(estats['val_loss'])
            valid_r.append(estats['val_pearsonr'])

    df_train = pd.DataFrame({
        'train_loss':train_loss,
        'train_r':train_r,
        'valid_loss':valid_loss,
        'valid_r':valid_r
        })

    return df_train


def epoch_stats(epoch_line):
    estats = {}
    for kv in epoch_line.split(' - '):
        if kv.find(':') != -1:
            k, v = kv.split(':')
            estats[k.strip()] = float(v)
    return estats

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
