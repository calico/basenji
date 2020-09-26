#!/usr/bin/env python
from optparse import OptionParser
import glob
import pdb
import os
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.stats import combine_pvalues, pearsonr, ttest_rel, wilcoxon
from tabulate import tabulate

import matplotlib.pyplot as plt
import seaborn as sns


'''
basenji_bench_gtex_cmp.py

Compare multiple variant score sets on the GTEx fine mapped eQTL benchmark.
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <bench1_dir> <bench2_dir> ...'
    parser = OptionParser(usage)
    parser.add_option('-a', '--alt', dest='alternative',
            default='two-sided', help='Statistical test alternative [Default: %default]')
    parser.add_option('--hue', dest='plot_hue',
            default=False, action='store_true',
            help='Scatter plot variant number as hue [Default: %default]')
    parser.add_option('-l', dest='labels')
    parser.add_option('-o', dest='out_dir',
            default='compare_scores')
    parser.add_option('--stats', dest='stats',
            default='SAD')
    parser.add_option('-v', dest='min_variants',
            default=0, type='int',
            help='Minimum variants to include tissue [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) == 0:
        parser.error('Must provide classification output directories')
    else:
        bench_dirs = args

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    num_benches = len(bench_dirs)
    sad_stats = [stat.lower() for stat in options.stats.split(',')]
    if len(sad_stats) == 1:
        sad_stats = sad_stats*num_benches

    sns.set(font_scale=1.2, style='ticks')

    if options.labels is None:
        options.labels = [os.path.split(bd)[1] for bd in bench_dirs]
    else:
        options.labels = options.labels.split(',')
    assert(len(options.labels) == num_benches)

    # initialize data frame lists
    df_tissues = []
    df_variants = []
    df_label1 = []
    df_label2 = []
    df_auroc1 = []
    df_auroc2 = []
    df_mwp = []
    df_tp = []

    # determine tissues
    tissue_bench_dirs0 = glob.glob('%s/*_class-%s' % (bench_dirs[0], sad_stats[0]))
    if len(tissue_bench_dirs0) == 0:
        # TEMP during transition
        tissue_bench_dirs0 = glob.glob('%s/*_class' % bench_dirs[0])
    # tissues = [tbd.split('/')[-1].replace('_class','') for tbd in tissue_bench_dirs0]
    tissue_class_dirs = [tbd.split('/')[-1] for tbd in tissue_bench_dirs0]
    tissues = [tcd[:tcd.find('_class')] for tcd in tissue_class_dirs]

    for tissue in tissues:
        tissue_out_dir = '%s/%s' % (options.out_dir, tissue)
        if not os.path.isdir(tissue_out_dir):
            os.mkdir(tissue_out_dir)

        # count variants
        with h5py.File('%s/%s_pos/sad.h5' % (bench_dirs[0],tissue), 'r') as tissue_sad_h5:
            sad_stat_up = sad_stats[0].upper()
            num_variants = tissue_sad_h5[sad_stat_up].shape[0]

        # read TPRs and FPRs
        bench_tpr_mean = []
        bench_fpr_mean = []
        bench_aurocs = []
        for i in range(num_benches):
            tissue_class_dir_i = '%s/%s_class-%s' % (bench_dirs[i],tissue,sad_stats[i])
            if not os.path.isdir(tissue_class_dir_i):
                # TEMP during transition
                tissue_class_dir_i = '%s/%s_class' % (bench_dirs[i],tissue)

            tpr_mean = np.load('%s/tpr_mean.npy' % tissue_class_dir_i)
            fpr_mean = np.load('%s/fpr_mean.npy' % tissue_class_dir_i)
            aurocs = np.load('%s/aurocs.npy' % tissue_class_dir_i)
            bench_tpr_mean.append(tpr_mean)
            bench_fpr_mean.append(fpr_mean)
            bench_aurocs.append(aurocs)

        # mean ROC plot
        plt.figure(figsize=(6,6))
        for i in range(num_benches):
            label_i = '%s AUROC %.4f' % (options.labels[i], bench_aurocs[i].mean())
            plt.plot(bench_fpr_mean[i], bench_tpr_mean[i], alpha=0.5, label=label_i)
        plt.legend()
        ax = plt.gca()
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        sns.despine()
        plt.tight_layout()                                                  
        plt.savefig('%s/roc_full.pdf' % tissue_out_dir)
        plt.close()


        # scatter plot versions' fold AUROCss
        for i in range(num_benches):
            for j in range(i+1, num_benches):
                if len(bench_aurocs[i]) == len(bench_aurocs[j]):
                    plt.figure(figsize=(6,6))
                    sns.scatterplot(bench_aurocs[i], bench_aurocs[j],
                                    color='black', linewidth=0, alpha=0.5)
                    ax = plt.gca()

                    vmin = min(bench_aurocs[i].min(), bench_aurocs[j].min())
                    vmax = max(bench_aurocs[i].max(), bench_aurocs[j].max())
                    ax.plot([vmin,vmax], [vmin,vmax], linestyle='--', color='gold')
                    ax.set_xlabel('%s fold AUROC' % options.labels[i])
                    ax.set_ylabel('%s fold AUROC' % options.labels[j])
                    sns.despine()
                    plt.tight_layout()
                    plt.savefig('%s/auroc_%s_%s.pdf' % (tissue_out_dir, options.labels[i], options.labels[j]))
                    plt.close()

                # append lists
                df_tissues.append(tissue)
                df_variants.append(num_variants)
                df_label1.append(options.labels[i])
                df_label2.append(options.labels[j])
                df_auroc1.append(bench_aurocs[i].mean())
                df_auroc2.append(bench_aurocs[j].mean())
                if len(bench_aurocs[i]) == len(bench_aurocs[j]):
                    df_mwp.append(wilcoxon(bench_aurocs[i], bench_aurocs[j],
                                           alternative=options.alternative)[1])
                    df_tp.append(ttest_alt(bench_aurocs[i], bench_aurocs[j],
                                           alternative=options.alternative)[1])
                else:
                    df_mwp.append(0)
                    df_tp.append(0)

    df_cmp = pd.DataFrame({
        'tissue':df_tissues,
        'variants':df_variants,
        'label1':df_label1,
        'label2':df_label2,
        'auroc1':df_auroc1,
        'auroc2':df_auroc2,
        'wilcoxon':df_mwp,
        'ttest':df_tp
        })

    # filter variants
    df_cmp = df_cmp[df_cmp.variants >= options.min_variants]

    # print table
    df_cmp.sort_values('variants', inplace=True)
    df_cmp.to_csv('%s/table_cmp.tsv' % options.out_dir, sep='\t')
    table_cmp = tabulate(df_cmp, headers='keys', tablefmt='github')
    border = table_cmp.split('\n')[1].replace('|','-')
    print(border)
    print(table_cmp)
    print(border)

    # scatter plot pairs
    for i in range(num_benches):
        for j in range(i+1, num_benches):
            mask_ij = (df_cmp.label1 == options.labels[i]) & (df_cmp.label2 == options.labels[j])
            df_cmp_ij = df_cmp[mask_ij]

            hue_var = None
            if options.plot_hue:
                hue_var = 'variants'

            plt.figure(figsize=(6,6))
            sns.scatterplot('auroc1', 'auroc2', data=df_cmp_ij,
                            hue=hue_var, linewidth=0, alpha=0.8)
            ax = plt.gca()

            vmin = min(df_cmp_ij.auroc1.min(), df_cmp_ij.auroc2.min())
            vmax = max(df_cmp_ij.auroc1.max(), df_cmp_ij.auroc2.max())
            ax.plot([vmin,vmax], [vmin,vmax], linestyle='--', color='black')

            eps = 0.05
            ax.text(1-eps, eps, 'Mean %.3f'%df_cmp_ij.auroc1.mean(),
                horizontalalignment='right', transform=ax.transAxes)
            ax.text(eps, 1-eps, 'Mean %.3f'%df_cmp_ij.auroc2.mean(),
                verticalalignment='top', transform=ax.transAxes)

            ax.set_xlabel('%s AUROC' % options.labels[i])
            ax.set_ylabel('%s AUROC' % options.labels[j])
            sns.despine()
            plt.tight_layout()
            plt.savefig('%s/auroc_%s_%s.pdf' % (options.out_dir, options.labels[i], options.labels[j]))
            plt.close()

            # wilcoxon_p = combine_pvalues(df_cmp_ij.wilcoxon)[1]
            # ttest_p = combine_pvalues(df_cmp_ij.ttest)[1]
            wilcoxon_p = wilcoxon(df_cmp_ij.auroc1, df_cmp_ij.auroc2,
                                  alternative=options.alternative)[1]
            ttest_p = ttest_alt(df_cmp_ij.auroc1, df_cmp_ij.auroc2,
                                alternative=options.alternative)[1]
            print('')
            print('%s AUROC: %.4f' % (options.labels[i], df_cmp_ij.auroc1.mean()))
            print('%s AUROC: %.4f' % (options.labels[j], df_cmp_ij.auroc2.mean()))
            print('Wilcoxon p: %.3g' % wilcoxon_p)
            print('T-test p:   %.3g' % ttest_p)

    
def read_auroc(stats_file):
    auroc = None

    for line in open(stats_file):
        a = line.split()
        if a[0] == 'AUROC:':
            auroc = float(a[1])
            break

    if auroc is None:
        print('AUROC not found in %s' % stats_file, file=sys.stderr)
        exit(1)

    return auroc


def ttest_alt(a, b, alternative='two-sided'):
    tt, tp = ttest_rel(a, b)

    if alternative == 'greater':
        if tt > 0:
            tp = 1 - (1-tp)/2
        else:
            tp /= 2
    elif alternative == 'less':
        if tt <= 0:
            tp /= 2
        else:
            tp = 1 - (1-tp)/2

    return tt, tp


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
