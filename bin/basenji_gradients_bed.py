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
basenji_jacobian_bed.py: Compute the gradient * input for sequences in the input bed file.
'''

class ComputeGradients():

    def __init__(self, params_file, model_file, bed_file, options):

        self.bed_file = bed_file
        self.model_file = model_file
        # load model parameters
        with open(params_file) as params_open:
            params = json.load(params_open)
        self.params_model = params['model']
        self.params_train = params['train']

        self.options = options

    def setup_model(self):
        # read targets
        if self.options.targets_file is None:
            target_slice = None
        else:
            targets_df = pd.read_table(self.options.targets_file, index_col=0)
            target_slice = targets_df.index

        seqnn_model = seqnn.SeqNN(self.params_model)
        seqnn_model.restore(self.model_file)
        seqnn_model.build_slice(target_slice)
        seqnn_model.build_ensemble(self.options.rc, self.options.shifts)

        num_targets = seqnn_model.num_targets()
        setattr(self, 'seqnn_model', seqnn_model)
        setattr(self, 'num_targets', num_targets)

    def read_seqs(self):
        # read sequences from BED
        seqs_dna, seqs_coords = bed.make_bed_seqs(
            self.bed_file, self.options.genome_fasta, self.params_model['seq_length'], stranded=True)

        num_seqs = len(seqs_dna)

        # determine mutation region limits
        seq_mid = self.params_model['seq_length'] // 2
        mut_start = seq_mid - options.mut_up
        mut_end = mut_start + options.mut_len

        setattr(self, 'num_seqs', num_seqs)
        setattr(self, 'mut_start', mut_start)
        setattr(self, 'mut_end', mut_end)

        return seqs_dna, seqs_coords

    def setup_output(self, seqs_coords):
        # note: do we want sequence properties to be class attributes?
        # setup output
        scores_h5_file = '%s/scores.h5' % self.options.out_dir
        if os.path.isfile(scores_h5_file):
            os.remove(scores_h5_file)
        scores_h5 = h5py.File(scores_h5_file, 'w')
        scores_h5.create_dataset('seqs', dtype='bool',
                                 shape=(self.num_seqs, self.options.mut_len, 4))
        for sad_stat in options.sad_stats:
            scores_h5.create_dataset(sad_stat, dtype='float16',
                                     shape=(self.num_seqs, self.options.mut_len, 4, self.num_targets))

        # store sequence coordinates
        scores_chr = []
        scores_start = []
        scores_end = []
        scores_strand = []
        for seq_chr, seq_start, seq_end, seq_strand in seqs_coords:
            scores_chr.append(seq_chr)
            scores_strand.append(seq_strand)
            if seq_strand == '+':
                score_start = seq_start + self.mut_start
                score_end = score_start + self.options.mut_len
            else:
                score_end = seq_end - self.mut_start
                score_start = score_end - self.options.mut_len
            scores_start.append(score_start)
            scores_end.append(score_end)

        scores_h5.create_dataset('chr', data=np.array(scores_chr, dtype='S'))
        scores_h5.create_dataset('start', data=np.array(scores_start))
        scores_h5.create_dataset('end', data=np.array(scores_end))
        scores_h5.create_dataset('strand', data=np.array(scores_strand, dtype='S'))

        return scores_h5

    def compute_gradients(self, seqs_dna, scores_h5):
        """
        Computing the gradient w.r.t the sequence
        """

        # compute gradients
        for si in range(self.num_seqs):
            print('Computing gradient w.r.t. input for sequence number %d' % si, flush=True)

            seq_dna = seqs_dna[si]
            seq_1hot_mut = dna_io.dna_1hot(seq_dna)
            seq_1hot_mut = np.expand_dims(seq_1hot_mut, axis=0)

            print("seq_1hot_mut.shape")
            print(seq_1hot_mut.shape)

            input_seq = tf.Variable(seq_1hot_mut, dtype=tf.float32)
            # additional variables
            input_left_flank = tf.stop_gradient(tf.Variable(seq_1hot_mut[:, 0:self.mut_start, :], dtype=tf.float32))
            input_right_flank = tf.stop_gradient(tf.Variable(seq_1hot_mut[:, self.mut_end:, :], dtype=tf.float32))
            input_seq_wind = tf.Variable(seq_1hot_mut[:, self.mut_start:self.mut_end, :], dtype=tf.float32)

            with tf.GradientTape(persistent=True) as tape:
                # Must delete tape since I have set the persistent flag = True
                # If persistent flag = False, then all stored values are discarded after a single gradient computation.
                # Here, we want to do multiple gradient computations (for each target wrt input)
                # input_seq = tf.concat([input_left_flank, input_seq_wind, input_right_flank], axis=1)
                preds = self.seqnn_model.model(tf.concat([input_left_flank, input_seq_wind, input_right_flank], axis=1),
                                          training=False)
                if 'sum' in options_sad_stats:
                    preds_sum = tf.math.reduce_sum(preds, axis=1)
                    output_variables = [preds_sum[:, target_idx] for target_idx in range(self.num_targets)]

            # compute the gradient
            grads_per_outvar = []
            for y_i in output_variables:
                print(y_i)
                curr_grad = tape.gradient(y_i, input_seq_wind)
                grads_per_outvar.append(curr_grad)
            grads = tf.concat(grads_per_outvar, axis=0)
            grads_x_inp = grads * seq_1hot_mut[:, self.mut_start:self.mut_end, :]
            print(grads_x_inp.shape)
            del tape

            scores_h5['seqs'][si, :, :] = seq_1hot_mut[:, self.mut_start:self.mut_end, :]

            for sad_stat in self.options.sad_stats:
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
                    print('Unrecognized summary statistic "%s"' % self.options.sad_stats)
                    exit(1)
        # save scores.h5
        scores_h5.close()

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
        params_file = args[0]
        model_file = args[1]
        bed_file = args[2]
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

    # Compute gradients
    compute_grads = ComputeGradients(params_file=params_file, model_file=model_file, bed_file=bed_file, options=options)
    compute_grads.setup_model()
    seqs_dna, seqs_coords = compute_grads.read_seqs()
    scores_h5 = compute_grads.setup_output(seqs_coords=seqs_coords)
    compute_grads.compute_gradients(seqs_dna=seqs_dna, scores_h5=scores_h5)

if __name__ == '__main__':
    main()