#!/usr/bin/env python
from optparse import OptionParser
import os
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

import basenji

################################################################################
# basenji_map.py
#
# Visualize a sequence's prediction's gradients as a map of influence across
# the genomic region.
#
# Notes:
#  -I'm providing the sequence as a FASTA file for now, but I may want to
#   provide a BED-style region so that I can print the output as a bigwig.
################################################################################

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <params_file> <model_file> <fasta_file>'
    parser = OptionParser(usage)
    parser.add_option('-l', dest='layers', default=None, help='Comma-separated list of layers to plot')
    parser.add_option('-o', dest='out_dir', default='grad_map', help='Output directory [Default: %default]')
    parser.add_option('-t', dest='target_indexes', default=None, help='Target indexes to plot')
    (options,args) = parser.parse_args()

    if len(args) != 3:
    	parser.error('Must provide parameters, model, and genomic position')
    else:
        params_file = args[0]
        model_file = args[1]
        fasta_file = args[2]

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    if options.layers is not None:
        options.layers = [int(li) for li in options.layers.split(',')]

    if options.target_indexes is not None:
        options.target_indexes = [int(ti) for ti in options.target_indexes.split(',')]

    #######################################################
    # one hot code sequence
    #######################################################
    seq = ''
    for line in open(fasta_file):
        if line[0] != '>':
            seq += line.rstrip()

    seq_1hot = basenji.dna_io.dna_1hot(seq)
    seqs_1hot = np.expand_dims(seq_1hot, axis=0)

    #######################################################
    # model parameters and placeholders
    #######################################################
    job = basenji.dna_io.read_job_params(params_file)

    job['batch_length'] = seqs_1hot.shape[1]
    job['seq_depth'] = seqs_1hot.shape[2]
    job['save_reprs'] = True

    if 'num_targets' not in job or 'target_pool' not in job:
        print("Must specify number of targets (num_targets) and target pooling (target_pool) in the parameters file. I know, it's annoying. Sorry.", file=sys.stderr)
        exit(1)

    model = basenji.rnn.RNN()
    model.build(job)

    #######################################################
    # test
    #######################################################
    # initialize batcher
    batcher = basenji.batcher.Batcher(seqs_1hot, batch_size=model.batch_size, pool_width=model.target_pool)

    # initialize saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # load variables into session
        saver.restore(sess, model_file)

        # get layer representations
        grads = model.gradients(sess, batcher, options.target_indexes, options.layers)
        print('grads', grads.shape)

        # plot as heatmap
        plt.figure()
        sns.heatmap(grads[0,:,:])
        plt.savefig('%s/heat.pdf' % options.out_dir)
        plt.close()

        # plot norm across filters as lineplot
        grad_norms = np.linalg.norm(grads, axis=2)
        plt.figure()
        plt.plot(range(grad_norms.shape[1]), grad_norms[0])
        plt.savefig('%s/norms.pdf' % options.out_dir)


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
