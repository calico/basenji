#!/usr/bin/env python
from optparse import OptionParser
from collections import OrderedDict
import os
import sys
import subprocess

import h5py
import numpy as np
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
    parser.add_option('-o', dest='out_dir', default='genes_out', help='Output directory for tables and plots [Default: %default]')
    parser.add_option('-t', dest='target_indexes', help='Comma-separated list of target indexes to scatter plot true versus predicted values')
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

    print('Reading from gene HDF')
    sys.stdout.flush()

    genes_hdf5_in = h5py.File(genes_hdf5_file)

    seg_chrom = [chrom.decode('UTF-8') for chrom in genes_hdf5_in['seg_chrom']]
    seg_start = np.array(genes_hdf5_in['seg_start'])
    seg_end = np.array(genes_hdf5_in['seg_end'])
    seqs_segments = list(zip(seg_chrom,seg_start,seg_end))

    seqs_1hot = genes_hdf5_in['seqs_1hot']

    transcripts = [tx.decode('UTF-8') for tx in genes_hdf5_in['transcripts']]
    transcript_index = np.array(genes_hdf5_in['transcript_index'])
    transcript_pos = np.array(genes_hdf5_in['transcript_pos'])

    transcript_map = OrderedDict()
    for ti in range(len(transcripts)):
        transcript_map[transcripts[ti]] = (transcript_index[ti], transcript_pos[ti])

    transcript_targets = genes_hdf5_in['transcript_targets']

    target_labels = [tl.decode('UTF-8') for tl in genes_hdf5_in['target_labels']]

    print(' Done')
    sys.stdout.flush()

    #################################################################
    # ignore genes overlapping trained BED regions

    if options.ignore_bed:
        seqs_segments, seqs_1hot, transcript_map, transcript_targets = ignore_trained_regions(options.ignore_bed, seqs_segments, seqs_1hot, transcript_map, transcript_targets, options.out_dir)


    #################################################################
    # setup model

    print('Constructing model')
    sys.stdout.flush()

    job = basenji.dna_io.read_job_params(params_file)

    job['batch_length'] = seqs_1hot.shape[1]
    job['seq_depth'] = seqs_1hot.shape[2]

    if 'num_targets' not in job:
        print("Must specify number of targets (num_targets) in the parameters file. I know, it's annoying. Sorry.", file=sys.stderr)
        exit(1)

    # build model
    dr = basenji.rnn.RNN()
    dr.build(job)

    if options.batch_size is not None:
        dr.batch_size = options.batch_size

    print(' Done')
    sys.stdout.flush()


    #################################################################
    # predict

    print('Computing gene predictions')
    sys.stdout.flush()

    # initialize batcher
    batcher = basenji.batcher.Batcher(seqs_1hot, batch_size=dr.batch_size)

    # initialie saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # load variables into session
        saver.restore(sess, model_file)

        # predict
        transcript_preds = dr.predict_genes(sess, batcher, transcript_map)

    print(' Done')
    sys.stdout.flush()


    #################################################################
    # print and plot

    if options.target_indexes is None:
        options.target_indexes = range(transcript_preds.shape[1])
    else:
        options.target_indexes = options.target_indexes.split(',')

    table_out = open('%s/table.txt' % options.out_dir, 'w')
    for ti in options.target_indexes:
        # plot scatter
        out_pdf = '%s/t%d.pdf' % (options.out_dir, ti)
        basenji.plots.jointplot(transcript_targets[:,ti], transcript_preds[:,ti], out_pdf)

        # print table lines
        for tx_i in range(len(transcripts)):
            # print transcript line
            cols = [transcripts[tx_i], transcript_targets[tx_i,ti], transcript_preds[tx_i,ti], ti, target_labels[ti]]
            print('%-20s  %.3f  %.3f  %4d  %20s' % cols, file=table_out)

    table_out.close()


    #################################################################
    # clean up

    genes_hdf5_in.close()


def ignore_trained_regions(ignore_bed, seqs_segments, seqs_1hot, transcript_map, transcript_targets, out_dir, mid_pct=0.5):
    ''' Filter the sequence and transcript data structures to ignore the sequences
         in a training set BED file.

    In

    Out
     seqs_segments
     seqs_1hot
     transcript_map
     transcript_targets
    '''

    # write segment coordinates to file
    seqs_bed_file = '%s/seqs.bed' % out_dir
    seqs_bed_out = open(seqs_bed_file, 'w')
    for chrom, start, end in seqs_segments:
        span = end-start
        mid = (start+end)/2
        mid_start = mid - mid_pct*span/2
        mid_end = mid + mid_pct*span/2
        print('%s\t%d\t%d' % (chrom,mid_start,mid_end), file=seqs_bed_out)
    seqs_bed_out.close()

    # intersect with the BED file
    p = subprocess.Popen('bedtools intersect -wo -a %s -b %s' % (seqs_bed_file,ignore_bed), shell=True, stdout=subprocess.PIPE)

    # track indexes that overlap
    seqs_keep = []
    for line in p.stdout:
        a = line.split()
        seqs_keep.append(int(a[-1]) == 0)
    seqs_keep = np.array(seqs_keep)

    # update sequence data structs
    seqs_segments = seqs_segments[seqs_keep]
    seqs_1hot = seqs_1hot[seqs_keep]

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

    # update transcript_targets
    transcript_targets = transcript_targets[np.logical_not(transcripts_ignore)]

    return seqs_segments, seqs_1hot, transcript_map, transcript_targets


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
