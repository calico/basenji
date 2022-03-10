from __future__ import print_function

from optparse import OptionParser

import gc
import json
import os
import pdb
import pickle
from queue import Queue
import random
import sys
from threading import Thread

import h5py
import numpy as np
import pandas as pd
import pysam
import tensorflow as tf

if tf.__version__[0] == '1':
    tf.compat.v1.enable_eager_execution()

from basenji import bed
from basenji import dna_io
from basenji import seqnn
from basenji import stream


'''
basenji_saliency_bed.py: Compute the gradient * input for sequences in the input bed file.
'''


def main():
    usage = 'usage: %prog [options] <params_file> <model_file> <bed_file>'
    parser = OptionParser(usage)
    parser.add_option('-d', dest='mut_down',
                      default=0, type='int',
                      help='Nucleotides downstream of center sequence to mutate [Default: %default]')
    parser.add_option('-f', dest='genome_fasta',
                      default=None,
                      help='Genome FASTA for sequences [Default: %default]')
    parser.add_option('-l', dest='mut_len',
                      default=0, type='int',
                      help='Length of center sequence to mutate [Default: %default]')
    parser.add_option('-o', dest='out_dir',
                      default='sat_mut', help='Output directory [Default: %default]')
    parser.add_option('--plots', dest='plots',
                      default=False, action='store_true',
                      help='Make heatmap plots [Default: %default]')
    parser.add_option('-p', dest='processes',
                      default=None, type='int',
                      help='Number of processes, passed by multi script')
    parser.add_option('--rc', dest='rc',
                      default=False, action='store_true',
                      help='Ensemble forward and reverse complement predictions [Default: %default]')
    parser.add_option('--shifts', dest='shifts',
                      default='0',
                      help='Ensemble prediction shifts [Default: %default]')
    parser.add_option('--stats', dest='sad_stats',
                      default='sum',
                      help='Comma-separated list of stats to save. [Default: %default]')
    parser.add_option('-t', dest='targets_file',
                      default=None, type='str',
                      help='File specifying target indexes and labels in table format')
    parser.add_option('-u', dest='mut_up',
                      default=0, type='int',
                      help='Nucleotides upstream of center sequence to mutate [Default: %default]')
    (options, args) = parser.parse_args()

    if len(args) == 3:
        # single worker
        params_file = args[0]
        model_file = args[1]
        bed_file = args[2]

    elif len(args) == 4:
        # master script
        options_pkl_file = args[0]
        params_file = args[1]
        model_file = args[2]
        bed_file = args[3]

        # load options
        options_pkl = open(options_pkl_file, 'rb')
        options = pickle.load(options_pkl)
        options_pkl.close()

    elif len(args) == 5:
        # multi worker
        options_pkl_file = args[0]
        params_file = args[1]
        model_file = args[2]
        bed_file = args[3]
        worker_index = int(args[4])

        # load options
        options_pkl = open(options_pkl_file, 'rb')
        options = pickle.load(options_pkl)
        options_pkl.close()

        # update output directory
        options.out_dir = '%s/job%d' % (options.out_dir, worker_index)

    else:
        parser.error('Must provide parameter and model files and BED file')

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    options.shifts = [int(shift) for shift in options.shifts.split(',')]
    options.sad_stats = [sad_stat.lower() for sad_stat in options.sad_stats.split(',')]

    if options.mut_up > 0 or options.mut_down > 0:
        options.mut_len = options.mut_up + options.mut_down
    else:
        assert (options.mut_len > 0)
        options.mut_up = options.mut_len // 2
        options.mut_down = options.mut_len - options.mut_up

    #################################################################
    # read parameters and targets

    # read model parameters
    with open(params_file) as params_open:
        params = json.load(params_open)
    params_model = params['model']
    params_train = params['train']

    # read targets
    if options.targets_file is None:
        target_slice = None
    else:
        targets_df = pd.read_table(options.targets_file, index_col=0)
        target_slice = targets_df.index

    #################################################################
    # setup model

    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(model_file)
    seqnn_model.build_slice(target_slice)
    seqnn_model.build_ensemble(options.rc, options.shifts)

    num_targets = seqnn_model.num_targets()

    #################################################################
    # sequence dataset

    # read sequences from BED
    seqs_dna, seqs_coords = bed.make_bed_seqs(
        bed_file, options.genome_fasta, params_model['seq_length'], stranded=True)

    # filter for worker SNPs
    if options.processes is not None:
        worker_bounds = np.linspace(0, len(seqs_dna), options.processes + 1, dtype='int')
        seqs_dna = seqs_dna[worker_bounds[worker_index]:worker_bounds[worker_index + 1]]
        seqs_coords = seqs_coords[worker_bounds[worker_index]:worker_bounds[worker_index + 1]]

    num_seqs = len(seqs_dna)

    # determine mutation region limits
    seq_mid = params_model['seq_length'] // 2
    mut_start = seq_mid - options.mut_up
    mut_end = mut_start + options.mut_len

    # make sequence generator
    # This is used to generate sequences for ISM
    # seqs_gen = satmut_gen(seqs_dna, mut_start, mut_end)

    #################################################################
    # setup output

    scores_h5_file = '%s/scores.h5' % options.out_dir
    if os.path.isfile(scores_h5_file):
        os.remove(scores_h5_file)
    scores_h5 = h5py.File(scores_h5_file, 'w')
    scores_h5.create_dataset('seqs', dtype='bool',
                             shape=(num_seqs, options.mut_len, 4))
    for sad_stat in options.sad_stats:
        scores_h5.create_dataset(sad_stat, dtype='float16',
                                 shape=(num_seqs, options.mut_len, 4, num_targets))

    # store mutagenesis sequence coordinates
    scores_chr = []
    scores_start = []
    scores_end = []
    scores_strand = []
    for seq_chr, seq_start, seq_end, seq_strand in seqs_coords:
        scores_chr.append(seq_chr)
        scores_strand.append(seq_strand)
        if seq_strand == '+':
            score_start = seq_start + mut_start
            score_end = score_start + options.mut_len
        else:
            score_end = seq_end - mut_start
            score_start = score_end - options.mut_len
        scores_start.append(score_start)
        scores_end.append(score_end)

    scores_h5.create_dataset('chr', data=np.array(scores_chr, dtype='S'))
    scores_h5.create_dataset('start', data=np.array(scores_start))
    scores_h5.create_dataset('end', data=np.array(scores_end))
    scores_h5.create_dataset('strand', data=np.array(scores_strand, dtype='S'))

    # compute gradients
    for si in range(num_seqs):
        print('Computing gradient w.r.t. input for sequence number %d' % si, flush=True)

        seq_dna = seqs_dna[si]
        seq_1hot_mut = dna_io.dna_1hot(seq_dna)
        seq_1hot_mut = np.expand_dims(seq_1hot_mut, axis=0)

        print("seq_1hot_mut.shape")
        print(seq_1hot_mut.shape)

        input_seq = tf.Variable(seq_1hot_mut, dtype=tf.float32)

        with tf.GradientTape() as tape:
            # record actions
            preds = seqnn_model.model(input_seq)

            if 'sum' in options.sad_stats:
                preds_sum = tf.math.reduce_sum(preds, axis=1)
                # compute the Jacobian
                grads = tape.jacobian(preds_sum, input_seq)
                grads = np.squeeze(grads)
                # slice "grads" to only keep mut_start:mut_end
                # TODO: This slice should happen earlier - reducing jacobian computation time.
                grads = grads[:, mut_start:mut_end, :]
                # grads.shape = (num_targets, mut_len, 4)
                # Compute gradient * input
                grads_x_inp = grads * seq_1hot_mut[:, mut_start:mut_end, :]
                # Note: This operation is the same as iterating over each target and computing grad x input
                # grad_x_inp.shape
                print(grads_x_inp.shape)

            if 'center' in options.sad_stats:
                raise NotImplementedError

            if 'scd' in options.sad_stats:
                raise NotImplementedError

        # write to HDF5
        scores_h5['seqs'][si, :, :] = seq_1hot_mut[:, mut_start:mut_end, :]

        for sad_stat in options.sad_stats:
            # initialize scores
            # seq_scores = np.zeros((mut_len, 4, num_targets), dtype='float32')

            # rearrange the "grads" array dimensions to be consistent with the ISM-style seq_scores
            seq_scores = np.transpose(grads_x_inp, axes=[1, 2, 0])
            print(seq_scores.shape)

            # summary stat
            if sad_stat == 'sum':
                # write to HDF5
                scores_h5[sad_stat][si, :, :, :] = seq_scores.astype('float16')
            elif sad_stat == 'center':
                raise NotImplementedError
            elif sad_stat == 'scd':
                raise NotImplementedError
            else:
                print('Unrecognized summary statistic "%s"' % options_sad_stat)
                exit(1)

    # save scores.h5
    scores_h5.close()

if __name__ == '__main__':
    main()
