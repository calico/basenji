#!/usr/bin/env python
from optparse import OptionParser
from collections import OrderedDict
import copy
import os
import subprocess
import sys
import tempfile

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
from sklearn.preprocessing import scale
import tensorflow as tf

import basenji

'''
basenji_test_genes.py

Compare predicted to measured CAGE gene expression estimates.
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <params_file> <model_file> <genes_hdf5_file>'
    parser = OptionParser(usage)
    parser.add_option('-b', dest='batch_size', default=None, type='int', help='Batch size [Default: %default]')
    parser.add_option('-i', dest='ignore_bed', help='Ignore genes overlapping regions in this BED file')
    parser.add_option('-l', dest='load_preds', help='Load transcript_preds from file')
    parser.add_option('-o', dest='out_dir', default='genes_out', help='Output directory for tables and plots [Default: %default]')
    parser.add_option('-s', dest='plot_scatter', default=False, action='store_true', help='Make time-consuming accuracy scatter plots [Default: %default]')
    parser.add_option('-t', dest='target_indexes', default=None, help='File or Comma-separated list of target indexes to scatter plot true versus predicted values')
    (options,args) = parser.parse_args()

    if len(args) != 3:
        parser.error('Must provide parameters and model files, and genes HDF5 file')
    else:
        params_file = args[0]
        model_file = args[1]
        genes_hdf5_file = args[2]

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    #################################################################
    # reads in genes HDF5

    print('Reading from gene HDF', flush=True)

    genes_hdf5_in = h5py.File(genes_hdf5_file)

    #######################################
    # read in sequences and descriptions

    seq_chrom = [chrom.decode('UTF-8') for chrom in genes_hdf5_in['seq_chrom']]
    seq_start = list(genes_hdf5_in['seq_start'])
    seq_end = list(genes_hdf5_in['seq_end'])
    seq_coords = list(zip(seq_chrom,seq_start,seq_end))

    seqs_1hot = genes_hdf5_in['seqs_1hot']

    #######################################
    # read in transcripts and map to sequences

    transcripts = [tx.decode('UTF-8') for tx in genes_hdf5_in['transcripts']]
    transcript_index = list(genes_hdf5_in['transcript_index'])
    transcript_pos = list(genes_hdf5_in['transcript_pos'])

    transcript_map = OrderedDict()
    for ti in range(len(transcripts)):
        transcript_map[transcripts[ti]] = (transcript_index[ti], transcript_pos[ti])

    # map transcript indexes to gene indexes
    gene_indexes = {}
    gene_list = []
    transcript_gene_indexes = []
    for gid in genes_hdf5_in['genes']:
        gid = gid.decode('UTF-8')
        if gid not in gene_indexes:
            gene_indexes[gid] = len(gene_indexes)
            gene_list.append(gid)
        transcript_gene_indexes.append(gene_indexes[gid])

    transcript_targets = genes_hdf5_in['transcript_targets']

    #######################################
    # targets

    target_labels = [tl.decode('UTF-8') for tl in genes_hdf5_in['target_labels']]

    if options.target_indexes is None:
        options.target_indexes = []
    elif options.target_indexes == 'all':
        options.target_indexes = range(transcript_targets.shape[1])
    elif os.path.isfile(options.target_indexes):
        target_indexes_file = options.target_indexes
        options.target_indexes = []
        for line in open(target_indexes_file):
            options.target_indexes.append(int(line.split()[0]))
    else:
        options.target_indexes = [int(ti) for ti in options.target_indexes.split(',')]

    options.target_indexes = np.array(options.target_indexes)

    print(' Done', flush=True)


    #################################################################
    # ignore genes overlapping trained BED regions

    if options.ignore_bed:
        seqs_1hot, transcript_map, transcript_targets = ignore_trained_regions(options.ignore_bed, seq_coords, seqs_1hot, transcript_map, transcript_targets)


    #################################################################
    # transcript predictions

    if options.load_preds is not None:
        # load from file
        transcript_preds = np.load(options.load_preds)

    else:

        #######################################################
        # setup model

        print('Constructing model', flush=True)

        job = basenji.dna_io.read_job_params(params_file)

        job['batch_length'] = seqs_1hot.shape[1]
        job['seq_depth'] = seqs_1hot.shape[2]
        job['target_pool'] = int(np.array(genes_hdf5_in['pool_width']))
        job['num_targets'] = transcript_targets.shape[1]

        # build model
        dr = basenji.rnn.RNN()
        dr.build(job)

        if options.batch_size is not None:
            dr.batch_size = options.batch_size

        print(' Done', flush=True)


        #######################################################
        # predict transcripts

        print('Computing gene predictions', flush=True)

        # initialize batcher
        batcher = basenji.batcher.Batcher(seqs_1hot, batch_size=dr.batch_size)

        # initialie saver
        saver = tf.train.Saver()

        with tf.Session() as sess:
            # load variables into session
            saver.restore(sess, model_file)

            # predict
            transcript_preds = dr.predict_genes(sess, batcher, transcript_map)

            # dr. predict_genes_bigwig(sess, batcher, seq_coords, options.out_dir, '%s/assembly/human.hg19.ml.genome'%os.environ['HG19'], [1471])
            # transcript_preds = dr.predict_genes_coords(sess, batcher, transcript_map, seq_coords)

        # save to file
        np.save('%s/preds' % options.out_dir, transcript_preds)

        print(' Done', flush=True)


    #################################################################
    # convert to genes

    gene_targets = map_transcripts_genes(transcript_targets, transcript_map, transcript_gene_indexes)
    gene_preds = map_transcripts_genes(transcript_preds, transcript_map, transcript_gene_indexes)


    #################################################################
    # correlation statistics

    cor_table(transcript_targets, transcript_preds, target_labels, '%s/transcript_cors.txt' % options.out_dir)
    cor_table(gene_targets, gene_preds, target_labels, '%s/gene_cors.txt' % options.out_dir)


    #################################################################
    # gene statistics

    gene_table(transcript_targets, transcript_preds, list(transcript_map.keys()), target_labels, options.target_indexes, '%s/transcript'%options.out_dir, options.plot_scatter)

    gene_table(gene_targets, gene_preds, gene_list, target_labels, options.target_indexes, '%s/gene'%options.out_dir, options.plot_scatter)


    #################################################################
    # normalize predictions across targets

    # focus on requested targets
    gene_preds_targets = gene_preds[:,options.target_indexes]
    gene_targets_targets = gene_targets[:,options.target_indexes]

    # take log
    gene_preds_log = np.log2(gene_preds_targets+1)
    gene_targets_log = np.log2(gene_targets_targets+1)

    # identify outliers
    gene_preds_tmean = gene_preds_log.mean(axis=0, dtype='float32')
    gene_targets_tmean = gene_targets_log.mean(axis=0, dtype='float32')

    # highlight outliers
    gene_preds_tmmean = gene_preds_tmean.mean()
    gene_targets_tmmean = gene_targets_tmean.mean()
    for ti in range(len(gene_targets_tmean)):
        if gene_targets_tmean[ti] > 2*gene_targets_tmmean or gene_targets_tmean[ti] < .5*gene_targets_tmmean:
            print('%d outlies %.3f versus %.3f (%.4f)' % (ti, gene_targets_tmean[ti], gene_targets_tmmean, gene_targets_tmean[ti] / gene_targets_tmmean))

    # quantile normalize
    gene_preds_qn = quantile_normalize(gene_preds_log, quantile_stat='mean')
    gene_targets_qn = quantile_normalize(gene_targets_log, quantile_stat='mean')

    #################################################################
    # plot genes by targets clustermap

    sns.set(font_scale=1.2, style='ticks')
    plot_genes = 1000

    # choose a set of variable genes
    gene_vars = gene_preds_qn.var(axis=1)
    indexes_var = np.argsort(gene_vars)[::-1][:plot_genes]

    # choose a set of random genes
    indexes_rand = np.random.choice(np.arange(gene_preds_qn.shape[0]), plot_genes, replace=False)

    # variable gene predictions
    clustermap(gene_preds_qn[indexes_var,:], '%s/gene_heat_var.pdf' % options.out_dir)
    clustermap(gene_preds_qn[indexes_var,:], '%s/gene_heat_var_color.pdf' % options.out_dir, color='viridis')

    # random gene predictions
    clustermap(gene_preds_qn[indexes_rand,:], '%s/gene_heat_rand.pdf' % options.out_dir)

    # variable gene targets
    clustermap(gene_targets_qn[indexes_var,:], '%s/gene_theat_var.pdf' % options.out_dir)
    clustermap(gene_targets_qn[indexes_var,:], '%s/gene_theat_var_color.pdf' % options.out_dir, color='viridis')

    # random gene targets
    clustermap(gene_targets_qn[indexes_rand,:], '%s/gene_theat_rand.pdf' % options.out_dir)

    #################################################################
    # clean up

    genes_hdf5_in.close()


def clustermap(gene_values, out_pdf, color=None):
    plt.figure()
    g = sns.clustermap(gene_values, metric='euclidean', cmap=color, xticklabels=False, yticklabels=False)
    g.ax_heatmap.set_xlabel('Targets')
    g.ax_heatmap.set_ylabel('Genes')
    plt.savefig(out_pdf)
    plt.close()


def cor_table(gene_targets, gene_preds, target_labels, out_file):
    ''' Print a table of target correlations. '''
    table_out = open(out_file, 'w')

    for ti in range(gene_targets.shape[1]):
        gti = gene_targets[:,ti]
        gpi = gene_preds[:,ti]
        scor, _ = spearmanr(gti, gpi)
        pcor, _ = pearsonr(np.log2(gti+1), np.log2(gpi+1))
        cols = (ti, scor, pcor, target_labels[ti])
        print('%-4d  %7.3f  %7.3f  %s' % cols, file=table_out)

    table_out.close()


def ignore_trained_regions(ignore_bed, seq_coords, seqs_1hot, transcript_map, transcript_targets, mid_pct=0.5):
    ''' Filter the sequence and transcript data structures to ignore the sequences
         in a training set BED file.

    In
     ignore_bed: BED file of regions to ignore
     seq_coords: list of (chrom,start,end) sequence coordinates
     seqs_1hot:
     transcript_map:
     transcript_targets:
     mid_pct:

    Out
     seqs_1hot
     transcript_map
     transcript_targets
    '''

    # write sequence coordinates to file
    seqs_bed_temp = tempfile.NamedTemporaryFile()
    seqs_bed_out = open(seqs_bed_temp.name, 'w')
    for chrom, start, end in seq_coords:
        span = end-start
        mid = (start+end)/2
        mid_start = mid - mid_pct*span // 2
        mid_end = mid + mid_pct*span // 2
        print('%s\t%d\t%d' % (chrom,mid_start,mid_end), file=seqs_bed_out)
    seqs_bed_out.close()

    # intersect with the BED file
    p = subprocess.Popen('bedtools intersect -c -a %s -b %s' % (seqs_bed_temp.name,ignore_bed), shell=True, stdout=subprocess.PIPE)

    # track indexes that overlap
    seqs_keep = []
    for line in p.stdout:
        a = line.split()
        seqs_keep.append(int(a[-1]) == 0)
    seqs_keep = np.array(seqs_keep)

    # update sequence data structs
    seqs_1hot = seqs_1hot[seqs_keep,:,:]

    # update transcript_map
    transcripts_keep = []
    transcript_map_new = OrderedDict()
    for transcript in transcript_map:
        tx_i, tx_pos = transcript_map[transcript]

        # collect ignored transcript bools
        transcripts_keep.append(seqs_keep[tx_i])

        # keep it
        if seqs_keep[tx_i]:
            # update the sequence index to consider previous kept sequences
            txn_i = seqs_keep[:tx_i].sum()

            # let's say it's 0 - False, 1 - True, 2 - True, 3 - False
            # 1 would may to 0
            # 2 would map to 1
            # all good!

            # update the map
            transcript_map_new[transcript] = (txn_i, tx_pos)

    transcript_map = transcript_map_new

    # convert to array
    transcripts_keep = np.array(transcripts_keep)

    # update transcript_targets
    transcript_targets = transcript_targets[transcripts_keep,:]

    return seqs_1hot, transcript_map, transcript_targets


def gene_table(gene_targets, gene_preds, gene_list, target_labels, target_indexes, out_prefix, plot_scatter):
    ''' Print a gene-based statistics table and scatter plot for the given target indexes. '''

    num_genes = gene_targets.shape[0]

    table_out = open('%s_table.txt' % out_prefix, 'w')

    for ti in target_indexes:
        gti = np.log2(gene_targets[:,ti]+1)
        gpi = np.log2(gene_preds[:,ti]+1)

        # plot scatter
        if plot_scatter:
            sns.set(font_scale=1.2, style='ticks')
            out_pdf = '%s_scatter%d.pdf' % (out_prefix, ti)
            ri = np.random.choice(range(num_genes), 2000, replace=False)
            basenji.plots.regplot(gti[ri].astype('float32'), gpi[ri].astype('float32'), out_pdf, poly_order=3, alpha=0.3, x_label='log2 Experiment', y_label='log2 Prediction')

        # print table lines
        tx_i = 0
        for gid in gene_list:
            # print transcript line
            cols = (gid, gti[tx_i], gpi[tx_i], ti, target_labels[ti])
            print('%-20s  %.3f  %.3f  %4d  %20s' % cols, file=table_out)
            tx_i += 1

    table_out.close()


def map_transcripts_genes(transcript_targets, transcript_map, transcript_gene_indexes):
    ''' Map a transcript X target array to a gene X target array '''

    # map sequence -> pos -> genes
    sequence_pos_genes = []

    # map sequence -> pos -> targets
    sequence_pos_targets = []

    genes = set()
    txi = 0
    for transcript in transcript_map:
        # transcript sits at sequence si at pos
        si, pos = transcript_map[transcript]

        # transcript to gene index
        gi = transcript_gene_indexes[txi]
        genes.add(gi)

        # extend sequence lists to tsi
        while len(sequence_pos_genes) <= si:
            sequence_pos_genes.append({})
            sequence_pos_targets.append({})

        # add gene to sequence/position set
        sequence_pos_genes[si].setdefault(pos,set()).add(gi)

        # save targets to sequence/position
        sequence_pos_targets[si][pos] = transcript_targets[txi,:]

        txi += 1

    # accumulate targets from sequence/positions for each gene
    gene_targets = np.zeros((len(genes), transcript_targets.shape[1]))
    for si in range(len(sequence_pos_genes)):
        for pos in sequence_pos_genes[si]:
            for gi in sequence_pos_genes[si][pos]:
                gene_targets[gi,:] += sequence_pos_targets[si][pos]

    return gene_targets


def quantile_normalize(gene_expr, quantile_stat='median'):
    ''' Quantile normalize across targets. '''

    # make a copy
    gene_expr_qn = copy.copy(gene_expr)

    # sort values within each column
    for ti in range(gene_expr.shape[1]):
        gene_expr_qn[:,ti].sort()

    # compute the mean/median in each row
    if quantile_stat == 'median':
        sorted_index_stats = np.median(gene_expr_qn, axis=1)
    elif quantile_stat == 'mean':
        sorted_index_stats = np.mean(gene_expr_qn, axis=1)
    else:
        print('Unrecognized quantile statistic %s' % quantile_stat, file=sys.stderr)
        exit()

    # set new values
    for ti in range(gene_expr.shape[1]):
        sorted_indexes = np.argsort(gene_expr[:,ti])
        for gi in range(gene_expr.shape[0]):
            gene_expr_qn[sorted_indexes[gi],ti] = sorted_index_stats[gi]

    return gene_expr_qn


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
