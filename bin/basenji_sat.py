#!/usr/bin/env python
from optparse import OptionParser
import os
import random
import subprocess
import sys
import tempfile
import time

import h5py
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import seaborn as sns
import tensorflow as tf

import basenji.dna_io
from seq_logo import seq_logo

################################################################################
# basenji_sat.py
#
# Perform an in silico saturated mutagenesis of the given test sequences
# using the given model.
#
# Note:
#  -Currently written assuming we have targets associated with each input sequence.
################################################################################

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <params_file> <model_file> <input_file>'
    parser = OptionParser(usage)
    parser.add_option('-a', dest='activity_enrich', default=1, type='float', help='Enrich the sample for the top proportion sorted by acitvity in the target cells requested. [Default: %default]')
    parser.add_option('-b', dest='batch_size', default=None, type='int', help='Batch size')
    parser.add_option('-l', dest='satmut_len', default=200, type='int', help='Length of centered sequence to mutate [Default: %default]')
    parser.add_option('-m', dest='min_limit', default=0.005, type='float', help='Minimum heatmap limit [Default: %default]')
    parser.add_option('-o', dest='out_dir', default='heat', help='Output directory [Default: %default]')
    parser.add_option('-r', dest='rng_seed', default=1, type='float', help='Random number generator seed [Default: %default]')
    parser.add_option('-s', dest='sample', default=None, type='int', help='Sample sequences from the test set [Default:%default]')
    parser.add_option('-t', dest='targets', default='0', help='Comma-separated list of target indexes to plot (or -1 for all) [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 3:
        parser.error('Must provide parameters and model files and input sequences (as a FASTA file or test data in an HDF file')
    else:
        params_file = args[0]
        model_file = args[1]
        test_file = args[2]

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    random.seed(options.rng_seed)

    #################################################################
    # parse input file
    #################################################################
    seqs, seqs_1hot, targets = parse_input(test_file, options.sample)

    # decide which targets to obtain
    if options.targets == '-1':
        target_indexes = range(targets.shape[2])
    else:
        target_indexes = [int(ti) for ti in options.targets.split(',')]

    # enrich for active sequences
    if targets is not None:
        seqs, seqs_1hot, targets = enrich_activity(seqs, seqs_1hot, targets, options.activity_enrich, target_indexes)

    seqs_n = seqs_1hot.shape[0]


    #################################################################
    # setup model
    #################################################################
    job = basenji.dna_io.read_job_params(params_file)

    job['batch_length'] = seqs_1hot.shape[1]
    job['seq_depth'] = seqs_1hot.shape[2]

    if targets is None:
        if 'num_targets' not in job or 'target_pool' not in job:
            print('Must provide num_targets and target_pool in parameters file', file=sys.stderr)
            exit(1)
    else:
        job['num_targets'] = targets.shape[2]
        job['target_pool'] = job['batch_length'] // targets.shape[1]

    t0 = time.time()
    dr = basenji.rnn.RNN()
    dr.build(job)
    print('Model building time %f' % (time.time()-t0), flush=True)

    if options.batch_size is not None:
        dr.batch_size = options.batch_size

    # initialize saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # load variables into session
        saver.restore(sess, model_file)

        for si in range(seqs_n):
            print('Mutating sequence %d / %d' % (si+1,seqs_n), flush=True)

            # write sequence
            fasta_out = open('%s/seq%d.fa' % (options.out_dir,si), 'w')
            end_len = (len(seqs[si]) - options.satmut_len) // 2
            print('>seq%d\n%s' % (si,seqs[si][end_len:-end_len]), file=fasta_out)
            fasta_out.close()

            #################################################################
            # predict modifications

            # supplement with saturated mutagenesis
            sat_seqs_1hot = satmut_seqs(seqs_1hot[si:si+1], options.satmut_len)

            # initialize batcher
            batcher_sat = basenji.batcher.Batcher(sat_seqs_1hot, batch_size=dr.batch_size)

            # predict
            sat_preds = dr.predict(sess, batcher_sat, rc_avg=True, target_indexes=target_indexes)

            #################################################################
            # compute delta, loss, and gain matrices

            # compute the matrix of prediction deltas: (4 x L_sm x T) array
            sat_delta = delta_matrix(seqs_1hot[si], sat_preds, options.satmut_len)

            # sat_loss, sat_gain = loss_gain(sat_delta, sat_preds[si], options.satmut_len)
            sat_loss = sat_delta.min(axis=0)
            sat_gain = sat_delta.max(axis=0)

            ##############################################
            # plot

            for ti in range(len(target_indexes)):
                # setup plot
                sns.set(style='white', font_scale=1)
                spp = subplot_params(sat_delta.shape[1])
                plt.figure(figsize=(20,4))
                ax_pred = plt.subplot2grid((4,spp['heat_cols']), (0,spp['pred_start']), colspan=spp['pred_span'])
                ax_logo = plt.subplot2grid((4,spp['heat_cols']), (1,spp['logo_start']), colspan=spp['logo_span'])
                ax_sad = plt.subplot2grid((4,spp['heat_cols']), (2,spp['sad_start']), colspan=spp['sad_span'])
                ax_heat = plt.subplot2grid((4,spp['heat_cols']), (3,0), colspan=spp['heat_cols'])

                # plot predictions
                plot_predictions(ax_pred, sat_preds[0,:,ti], options.satmut_len, dr.batch_length, dr.batch_buffer)

                # plot sequence weblogo
                plot_weblogo(ax_logo, seqs[si], sat_loss[:,ti], options.min_limit)

                # plot SAD
                plot_sad(ax_sad, sat_loss[:,ti], sat_gain[:,ti])

                # plot heat map
                plot_heat(ax_heat, sat_delta[:,:,ti], options.min_limit)

                plt.savefig('%s/seq%d_t%d.pdf' % (options.out_dir,si,ti), dpi=1200)
                plt.close()


def enrich_activity(seqs, seqs_1hot, targets, activity_enrich, target_indexes):
    ''' Filter data for the most active sequences in the set. '''

    # compute the max across sequence lengths and mean across targets
    seq_scores = targets[:,:,target_indexes].max(axis=1).mean(axis=1, dtype='float64')

    # sort the sequences
    scores_indexes = [(seq_scores[si], si) for si in range(seq_scores.shape[0])]
    scores_indexes.sort(reverse=True)

    # filter for the top
    enrich_indexes = sorted([scores_indexes[si][1] for si in range(seq_scores.shape[0])])
    enrich_indexes = enrich_indexes[:int(activity_enrich*len(enrich_indexes))]
    seqs = [seqs[ei] for ei in enrich_indexes]
    seqs_1hot = seqs_1hot[enrich_indexes]
    targets = targets[enrich_indexes]

    return seqs, seqs_1hot, targets


def delta_matrix(seqs_1hot, sat_preds, satmut_len):
    ''' Compute the matrix of prediction deltas

    Args:
        seqs_1hot (Lx4 array): One-hot coding of all sequences.
        sat_preds: (SMxLxT array): Satmut sequence predictions.
        satmut_len: Saturated mutagenesis region length.

    Returns:
        sat_delta (4 x L_sm x T array): Delta matrix for saturated mutagenesis region.

    Todo:
        -Rather than computing the delta as the change at that nucleotide's prediction,
            compute it as the mean change across the sequence. That way, we better
            pick up on motif-flanking interactions.
    '''
    seqs_n = int(sat_preds.shape[0] / (1 + 3*satmut_len))
    num_targets = sat_preds.shape[2]

    # left-over from previous version
    # we're just expecting one sequence now
    si = 0

    # initialize
    sat_delta = np.zeros((4,satmut_len,num_targets), dtype='float64')

    # jump to si's mutated sequences
    smi = seqs_n + si*3*satmut_len

    # jump to satmut region in preds (incorrect with target pooling)
    # spi = int((sat_preds.shape[1] - satmut_len) // 2)

    # jump to satmut region in sequence
    ssi = int((seqs_1hot.shape[0] - satmut_len) // 2)

    # to study across sequence length
    # sat_delta_length = np.zeros((4, satmut_len, sat_preds.shape[1], num_targets))

    # compute delta matrix
    for li in range(satmut_len):
        for ni in range(4):
            if seqs_1hot[ssi+li,ni] == 1:
                sat_delta[ni,li,:] = 0
            else:
                # to study across sequence length
                # sat_delta_length[ni,li,:,:] = sat_preds[smi] - sat_preds[si]

                # sat_delta[ni,li,:] = sat_preds[smi,spi+li,:] - sat_preds[si,spi+li,:]
                sat_delta[ni,li,:] = sat_preds[smi].sum(axis=0, dtype='float64') - sat_preds[si].sum(axis=0, dtype='float64')
                smi += 1

    # to study across sequence length
    '''
    sat_delta_length = sat_delta_length.mean(axis=0)

    if not os.path.isdir('length'):
        os.mkdir('length')
    for ti in range(sat_delta_length.shape[2]):
        plt.figure()
        sns.heatmap(sat_delta_length[:,:,ti], linewidths=0, cmap='RdBu_r')
        plt.savefig('length/delta_length_s%d_t%d.pdf' % (si,ti))
        plt.close()
    '''

    return sat_delta


def loss_gain(sat_delta, sat_preds_si, satmut_len):
    # compute min and max
    sat_min = sat_delta.min(axis=0)
    sat_max = sat_delta.max(axis=0)

    # determine sat mut region
    sm_start = (sat_preds_si.shape[0] - satmut_len) // 2
    sm_end = sm_start + satmut_len

    # compute loss and gain matrixes
    sat_loss = sat_min - sat_preds_si[sm_start:sm_end,:]
    sat_gain = sat_max - sat_preds_si[sm_start:sm_end,:]

    return sat_loss, sat_gain


def parse_input(input_file, sample):
    ''' Parse an input file that might be FASTA or HDF5. '''

    try:
        # input_file is FASTA

        # read sequences and headers
        seqs = []
        seq_headers = []
        for line in open(input_file):
            if line[0] == '>':
                seq_headers.append(line[1:].rstrip())
                seqs.append('')
            else:
                seqs[-1] += line.rstrip()

        # convert to arrays
        seqs = np.array(seqs)
        seq_headers = np.array(seq_headers)

        # one hot code sequences
        seqs_1hot = []
        for seq in seqs:
            seqs_1hot.append(basenji.dna_io.dna_1hot(seq))
        seqs_1hot = np.array(seqs_1hot)

        # sample
        if sample:
            sample_i = np.array(random.sample(xrange(seqs_1hot.shape[0]), sample))
            seqs_1hot = seqs_1hot[sample_i]
            seq_headers = seq_headers[sample_i]
            seqs = seqs[sample_i]

        # initialize targets variable
        targets = None

    except (UnicodeDecodeError):
        # input_file is HDF5

        try:
            # load (sampled) test data from HDF5
            hdf5_in = h5py.File(input_file, 'r')
            seqs_1hot = np.array(hdf5_in['test_in'])
            targets = np.array(hdf5_in['test_out'])
            # seq_headers = np.array(hdf5_in['test_headers'])
            # target_labels = np.array(hdf5_in['target_labels'])   # TEMP
            hdf5_in.close()

            # sample
            if sample:
                sample_i = np.array(random.sample(range(seqs_1hot.shape[0]), sample))
                seqs_1hot = seqs_1hot[sample_i]
                targets = targets[sample_i]
                # seq_headers = seq_headers[sample_i]

            # convert to ACGT sequences
            seqs = basenji.dna_io.hot1_dna(seqs_1hot)

        except IOError:
            parser.error('Could not parse input file as FASTA or HDF5.')

    return seqs, seqs_1hot, targets


def plot_heat(ax, sat_delta_ti, min_limit):
    ''' Plot satmut deltas.

    Args:
        ax (Axis): matplotlib axis to plot to.
        sat_delta_ti (4 x L_sm array): Single target delta matrix for saturated mutagenesis region,
        min_limit (float): Minimum heatmap limit.
    '''
    vlim = max(min_limit, abs(sat_delta_ti).max())
    sns.heatmap(sat_delta_ti, linewidths=0, cmap='RdBu_r', vmin=-vlim, vmax=vlim, xticklabels=False, ax=ax)
    ax.yaxis.set_ticklabels('TGCA', rotation='horizontal') # , size=10)


def plot_predictions(ax, preds, satmut_len, seq_len, buffer):
    ''' Plot the raw predictions for a sequence and target
        across the specificed saturated mutagenesis region.

    Args:
        ax (Axis): matplotlib axis to plot to.
        preds (L array): Target predictions for one sequence.
        satmut_len (int): Satmut length from which to determine
                           the values to plot.
        seq_len (int): Full sequence length.
        buffer (int): Ignored buffer sequence on each side
    '''

    # repeat preds across pool width
    target_pool = (seq_len - 2*buffer) // preds.shape[0]
    epreds = preds.repeat(target_pool)

    satmut_start = (epreds.shape[0] - satmut_len) // 2
    satmut_end = satmut_start + satmut_len

    ax.plot(epreds[satmut_start:satmut_end], linewidth=1)
    ax.set_xlim(0, satmut_len)
    ax.axhline(0, c='black', linewidth=1, linestyle='--')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.5)


def plot_sad(ax, sat_loss_ti, sat_gain_ti):
    ''' Plot loss and gain SAD scores.

    Args:
        ax (Axis): matplotlib axis to plot to.
        sat_loss_ti (L_sm array): Minimum mutation delta across satmut length.
        sat_gain_ti (L_sm array): Maximum mutation delta across satmut length.
    '''

    rdbu = sns.color_palette("RdBu_r", 10)

    ax.plot(-sat_loss_ti, c=rdbu[0], label='loss', linewidth=1)
    ax.plot(sat_gain_ti, c=rdbu[-1], label='gain', linewidth=1)
    ax.set_xlim(0, len(sat_loss_ti))
    ax.legend()
    # ax_sad.grid(True, linestyle=':')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.5)


def plot_weblogo(ax, seq, sat_loss_ti, min_limit):
    ''' Plot height-weighted weblogo sequence.

    Args:
        ax (Axis): matplotlib axis to plot to.
        seq ([ACGT]): DNA sequence
        sat_loss_ti (L_sm array): Minimum mutation delta across satmut length.
        min_limit (float): Minimum heatmap limit.
    '''
    # trim sequence to the satmut region
    satmut_len = len(sat_loss_ti)
    satmut_start = int((len(seq) - satmut_len) // 2)
    satmut_seq = seq[satmut_start:satmut_start+satmut_len]

    # determine nt heights
    vlim = max(min_limit, np.max(-sat_loss_ti))
    seq_heights = 0.1 + 1.9/vlim*(-sat_loss_ti)

    # make logo as eps
    eps_fd, eps_file = tempfile.mkstemp()
    seq_logo(satmut_seq, seq_heights, eps_file, color_mode='meme')

    # convert to png
    png_fd, png_file = tempfile.mkstemp()
    subprocess.call('convert -density 1200 %s %s' % (eps_file, png_file), shell=True)

    # plot
    logo = Image.open(png_file)
    ax.imshow(logo)
    ax.set_axis_off()

    # clean up
    os.close(eps_fd)
    os.remove(eps_file)
    os.close(png_fd)
    os.remove(png_file)


def satmut_seqs(seqs_1hot, satmut_len):
    ''' Construct a new array with the given sequences and saturated
        mutagenesis versions of them. '''

    seqs_n = seqs_1hot.shape[0]
    seq_len = seqs_1hot.shape[1]
    satmut_n = seqs_n + seqs_n*satmut_len*3

    # initialize satmut seqs 1hot
    sat_seqs_1hot = np.zeros((satmut_n, seq_len, 4), dtype='bool')

    # copy over seqs_1hot
    sat_seqs_1hot[:seqs_n,:,:] = seqs_1hot

    satmut_start = (seq_len - satmut_len) // 2
    satmut_end = satmut_start + satmut_len

    # add saturated mutagenesis
    smi = seqs_n
    for si in range(seqs_n):
        for li in range(satmut_start, satmut_end):
            for ni in range(4):
                if seqs_1hot[si,li,ni] != 1:
                    # copy sequence
                    sat_seqs_1hot[smi,:,:] = seqs_1hot[si,:,:]

                    # mutate to ni
                    sat_seqs_1hot[smi,li,:] = np.zeros(4)
                    sat_seqs_1hot[smi,li,ni] = 1

                    # update index
                    smi += 1

    return sat_seqs_1hot


def subplot_params(seq_len):
    ''' Specify subplot layout parameters for various sequence lengths. '''
    if seq_len < 500:
        spp = {'heat_cols': 400,
                'pred_start': 0,
                'pred_span': 322,
                'sad_start': 1,
                'sad_span': 321,
                'logo_start': 0,
                'logo_span': 323}
    else:
        spp = {'heat_cols': 400,
                'pred_start': 0,
                'pred_span': 321,
                'sad_start': 1,
                'sad_span': 320,
                'logo_start': 0,
                'logo_span': 322}

    return spp


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
