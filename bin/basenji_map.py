#!/usr/bin/env python
from optparse import OptionParser
import gc
import os
import sys
import time

import h5py
import numpy as np
import pyBigWig
import tensorflow as tf

import basenji

from basenji_test import bigwig_open

'''
basenji_map.py

Visualize a sequence's prediction's gradients as a map of influence across
the genomic region.

Notes:
  -I'm providing the sequence as a FASTA file for now, but I may want to
   provide a BED-style region so that I can print the output as a bigwig.
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <params_file> <model_file> <genes_hdf5_file>'
    parser = OptionParser(usage)
    parser.add_option('-g', dest='genome_file', default='%s/assembly/human.hg19.genome'%os.environ['HG19'], help='Chromosome lengths file [Default: %default]')
    parser.add_option('-l', dest='transcript_list', help='Process only transcript ids in the given file')
    parser.add_option('-o', dest='out_dir', default='grad_map', help='Output directory [Default: %default]')
    parser.add_option('-t', dest='target_indexes', default=None, help='Target indexes to plot')
    (options,args) = parser.parse_args()

    if len(args) != 3:
    	parser.error('Must provide parameters, model, and genomic position')
    else:
        params_file = args[0]
        model_file = args[1]
        genes_hdf5_file = args[2]

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)


    #################################################################
    # reads in genes HDF5

    genes_hdf5_in = h5py.File(genes_hdf5_file)

    seq_coords, seqs_1hot, seq_transcripts, transcript_targets, transcript_genes, target_labels = read_hdf5(genes_hdf5_in)

    # subset transcripts
    transcripts_subset = set()
    if options.transcript_list:
        for line in open(options.transcript_list):
            transcripts_subset.add(line.rstrip())

        seq_mask = np.zeros(seqs_1hot.shape[0], dtype='bool')
        for si in range(len(seq_mask)):
            # check this sequence's transcripts for matches
            seq_si_mask = [tx_id in transcripts_subset for tx_id, tx_pos in seq_transcripts[si]]

            # if some transcripts match
            if np.sum(seq_si_mask) > 0:
                # keep the sequence
                seq_mask[si] = True

                # filter the transcript list
                seq_transcripts[si] = [seq_transcripts[si][sti] for sti in range(len(seq_si_mask)) if seq_si_mask[sti]]

        # filter the sequence data structures
        seq_coords = [seq_coords[si] for si in range(len(seq_coords)) if seq_mask[si]]
        seqs_1hot = seqs_1hot[seq_mask,:,:]
        seq_transcripts = [seq_transcripts[si] for si in range(len(seq_transcripts)) if seq_mask[si]]

        # unused
        # transcript_targets
        # transcript_genes

        print('Filtered to %d sequences' % seqs_1hot.shape[0])

    #######################################################
    # model parameters and placeholders

    job = basenji.dna_io.read_job_params(params_file)

    job['batch_length'] = seqs_1hot.shape[1]
    job['seq_depth'] = seqs_1hot.shape[2]
    job['target_pool'] = int(np.array(genes_hdf5_in['pool_width']))
    job['save_reprs'] = True

    if 'num_targets' not in job:
        print("Must specify number of targets (num_targets) in the parameters file. I know, it's annoying. Sorry.", file=sys.stderr)
        exit(1)

    # build model
    model = basenji.rnn.RNN()
    model.build(job)

    # determine final pooling layer
    post_pooling_layer = len(model.cnn_pool)-1

    #######################################################
    # acquire gradients

    # set target indexes
    if options.target_indexes is not None:
        options.target_indexes = [int(ti) for ti in options.target_indexes.split(',')]
    else:
        options.target_indexes = list(range(job['num_targets']))

    # initialize saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # load variables into session
        saver.restore(sess, model_file)

        si = 0
        while si < seqs_1hot.shape[0]:
            # initialize batcher
            # batcher = basenji.batcher.Batcher(seqs_1hot[si:si+model.batch_size], batch_size=model.batch_size, pool_width=model.target_pool)
            batcher = basenji.batcher.Batcher(seqs_1hot[si:si+1], batch_size=model.batch_size, pool_width=model.target_pool)

            # determine transcript positions
            transcript_positions = set()
            # for bi in range(model.batch_size):   # TEMP
            for bi in range(1):
                if si+bi < len(seq_transcripts):
                    for transcript, tx_pos in seq_transcripts[si+bi]:
                        transcript_positions.add(tx_pos)
            transcript_positions = sorted(list(transcript_positions))

            # get layer representations
            t0 = time.time()
            print('Computing gradients.', end='', flush=True)
            batch_grads = model.gradients_pos(sess, batcher, transcript_positions, options.target_indexes, post_pooling_layer)
            print(' Done in %ds.' % (time.time()-t0), flush=True)

            # only layer
            batch_grads = batch_grads[0]

            # (B sequences) x (P pooled seq len) x (F filters) x (G gene positions) x (T targets)
            print('batch_grads', batch_grads.shape)

            # sum across filters
            batch_grads_sum = batch_grads.sum(axis=2)

            # (B sequences) x (P pooled seq len) x (G gene positions) x (T targets)
            pooled_length = batch_grads_sum.shape[1]

            # write bigwigs
            t0 = time.time()
            print('Writing BigWigs.', end='', flush=True)
            # for bi in range(model.batch_size):   # TEMP
            for bi in range(1):
                sbi = si+bi
                if sbi < len(seq_transcripts):
                    positions_written = set()
                    for transcript, tx_pos in seq_transcripts[sbi]:
                        # has this transcript position been written?
                        if tx_pos not in positions_written:
                            # which gene position is this tx_pos?
                            gi = 0
                            while transcript_positions[gi] != tx_pos:
                                gi += 1

                            # for each target
                            for tii in range(len(options.target_indexes)):
                                ti = options.target_indexes[tii]

                                # bw_file, options.genome_file, seq_coords[sbi], batch_grads_sum[bi]

                                bw_file = '%s/%s_t%d.bw' % (options.out_dir, transcript, ti)
                                bw_open = bigwig_open(bw_file, options.genome_file)

                                seq_chrom, seq_start, seq_end = seq_coords[sbi]
                                bw_chroms = [seq_chrom]*pooled_length
                                bw_starts = [int(seq_start + li*model.target_pool) for li in range(pooled_length)]
                                bw_ends = [int(bws + model.target_pool) for bws in bw_starts]

                                bw_values = [float(bgs) for bgs in batch_grads_sum[bi,:,gi,tii]]
                                bw_open.addEntries(bw_chroms, bw_starts, ends=bw_ends, values=bw_values)

                                bw_open.close()

                                positions_written.add(tx_pos)
            print(' Done in %ds.' % (time.time()-t0), flush=True)
            gc.collect()

            # advance through sequences
            # si += model.batch_size
            si += 1


def read_hdf5(genes_hdf5_in):
    #######################################
    # seq_coords

    seq_chrom = [chrom.decode('UTF-8') for chrom in genes_hdf5_in['seq_chrom']]
    seq_start = list(genes_hdf5_in['seq_start'])
    seq_end = list(genes_hdf5_in['seq_end'])
    seq_coords = list(zip(seq_chrom,seq_start,seq_end))

    #######################################
    # seqs_1hot

    seqs_1hot = genes_hdf5_in['seqs_1hot']
    print('genes seqs_1hot', seqs_1hot.shape)

    #######################################
    # transcript_map

    transcripts = [tx.decode('UTF-8') for tx in genes_hdf5_in['transcripts']]
    transcript_index = list(genes_hdf5_in['transcript_index'])
    transcript_pos = list(genes_hdf5_in['transcript_pos'])

    transcript_map = {}
    for ti in range(len(transcripts)):
        transcript_map[transcripts[ti]] = (transcript_index[ti], transcript_pos[ti])

    #######################################
    # transcript_genes

    genes = [gid.decode('UTF-8') for gid in genes_hdf5_in['genes']]

    transcript_genes = {}
    for ti in range(len(transcripts)):
        transcript_genes[transcripts[ti]] = genes[ti]

    #######################################
    # transcript_targets / target_labels

    if 'transcript_targets' in genes_hdf5_in:
        transcript_targets = genes_hdf5_in['transcript_targets']
        target_labels = [tl.decode('UTF-8') for tl in genes_hdf5_in['target_labels']]
    else:
        transcript_targets = None
        target_labels = None

    #######################################
    # seq_transcripts

    seq_transcripts = []
    for si in range(len(seq_coords)):
        seq_transcripts.append([])

    for transcript in transcript_map:
        tx_index, tx_pos = transcript_map[transcript]
        seq_transcripts[tx_index].append((transcript,tx_pos))

    return seq_coords, seqs_1hot, seq_transcripts, transcript_targets, transcript_genes, target_labels

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
