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
from tensorflow.keras import mixed_precision

if tf.__version__[0] == '1':
    tf.compat.v1.enable_eager_execution()

from basenji import bed
from basenji import dna_io
from basenji import seqnn
from basenji import stream


'''
basenji_hessian_bed.py: Compute the hessian matrix for sequences in the input bed file.
'''

class ComputeHessian():

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
        mut_start = seq_mid - self.options.mut_up
        mut_end = mut_start + self.options.mut_len

        setattr(self, 'num_seqs', num_seqs)
        setattr(self, 'mut_start', mut_start)
        setattr(self, 'mut_end', mut_end)

        return seqs_dna, seqs_coords

    def compute_gradients(self, seq_1hot_mut):

        input_left_flank = tf.Variable(seq_1hot_mut[:, 0:self.mut_start, :], dtype=tf.float32)
        input_right_flank = tf.Variable(seq_1hot_mut[:, self.mut_end:, :], dtype=tf.float32)
        input_seq_wind = tf.Variable(seq_1hot_mut[:, self.mut_start:self.mut_end, :], dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape2:
            with tf.GradientTape() as tape1:
                preds = self.seqnn_model.model(tf.concat([input_left_flank, input_seq_wind, input_right_flank], axis=1),
                                               training=False)
                preds_sum = tf.math.reduce_sum(preds, axis=1)[:, 0]  # working with a single target for now.
            dy_dx = tape1.gradient(preds_sum, input_seq_wind)
            grads_x_inp = dy_dx * seq_1hot_mut[:, self.mut_start:self.mut_end, :]
            # Reduce the grads_x_input into a 1-D array using tf operations.
            grads_x_inp = tf.reshape(grads_x_inp, (self.options.mut_len, 4))
            indices = tf.stack([range(self.options.mut_len), np.nonzero(grads_x_inp)[1]], axis=-1)
            # note: np.nonzero returns a tuple of arrays (one for each dimension)
            grads_1d = tf.gather_nd(grads_x_inp, indices=indices)
            print(grads_1d.shape)

        # Compute the jacobian w.r.t the gradients (equivalent to the Hessian w.r.t to the input)
        d2y_dx2 = tape2.jacobian(grads_1d, input_seq_wind, experimental_use_pfor=False)
        del tape2

        hess = []
        for row in d2y_dx2:
            tmp_var = row * input_seq_wind
            # reshape
            tmp_var = tf.reshape(tmp_var, (self.options.mut_len, 4))
            indices = tf.stack([range(self.options.mut_len), np.nonzero(tmp_var)[1]], axis=-1)
            d2y_dx2_1d = tf.gather_nd(tmp_var, indices=indices)
            hess.append(d2y_dx2_1d)
        hessian = np.array(hess)
        return hessian

    def process_seqs(self, seqs_dna, out_dir):
        """
        Computing the gradient w.r.t the sequence
        """
        for si in range(self.num_seqs):
            seq_dna = seqs_dna[si]
            seq_1hot_mut = dna_io.dna_1hot(seq_dna)
            seq_1hot_mut = np.expand_dims(seq_1hot_mut, axis=0)
            print("seq_1hot_mut.shape")
            print(seq_1hot_mut.shape)
            print('Computing gradient w.r.t. input for sequence number %d' % si, flush=True)
            # compute the hessian matrix
            hessian_mat = self.compute_gradients(seq_1hot_mut)
            np.savetxt(out_dir + '/hessian.' + str(si) + '.txt', hessian_mat, delimiter='\t')


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
    parser.add_option('--mixed', dest='policy',
                      default=False, action='store_true',
                      help='Use a mixed float16 keras policy')
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

    if options.policy:
        print('using the tf mixed float policy')
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        # This should set the policy for all tf layers and computations.

    # Compute gradients
    compute_grads = ComputeHessian(params_file=params_file, model_file=model_file, bed_file=bed_file, options=options)
    # In setup model, options.policy is passed to SeqNN to override a 32 bit computation.
    compute_grads.setup_model()
    seqs_dna, seqs_coords = compute_grads.read_seqs()
    compute_grads.process_seqs(seqs_dna=seqs_dna, out_dir=options.out_dir)


if __name__ == '__main__':
    main()
