#!/usr/bin/env python
from optparse import OptionParser
from collections import OrderedDict
import os
import subprocess
import sys
import tempfile

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

    seq_chrom = [chrom.decode('UTF-8') for chrom in genes_hdf5_in['seq_chrom']]
    seq_start = list(genes_hdf5_in['seq_start'])
    seq_end = list(genes_hdf5_in['seq_end'])
    seq_coords = list(zip(seq_chrom,seq_start,seq_end))

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
        seqs_1hot, transcript_map, transcript_targets = ignore_trained_regions(options.ignore_bed, seq_coords, seqs_1hot, transcript_map, transcript_targets)


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
        options.target_indexes = [int(ti) for ti in options.target_indexes.split(',')]

    table_out = open('%s/table.txt' % options.out_dir, 'w')
    for ti in options.target_indexes:
        # plot scatter
        out_pdf = '%s/t%d.pdf' % (options.out_dir, ti)
        basenji.plots.jointplot(transcript_targets[:,ti], transcript_preds[:,ti], out_pdf)

        # print table lines
        tx_i = 0
        for transcript in transcript_map:
            # print transcript line
            cols = (transcript, transcript_targets[tx_i,ti], transcript_preds[tx_i,ti], ti, target_labels[ti])
            print('%-20s  %.3f  %.3f  %4d  %20s' % cols, file=table_out)
            tx_i += 1

    table_out.close()


    #################################################################
    # clean up

    genes_hdf5_in.close()


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


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
