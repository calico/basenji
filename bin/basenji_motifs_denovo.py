#!/usr/bin/env python
from optparse import OptionParser
import collections
import json
import os
import pdb
import random
import subprocess

import h5py
import igraph
import leidenalg
from natsort import natsorted
import nimfa
import nmslib
import numpy as np
import pysam

import matplotlib.pyplot as plt
import seaborn as sns

from basenji import seqnn

'''
basenji_motifs_denovo.py

Study motifs in a Basenji model, via activations on given sequences.
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <params_file> <model_file> <bed_file>'
    parser = OptionParser(usage)
    parser.add_option('-a', dest='align_seqlets',
        default=False, action='store_true',
        help='Align seqlets [Default: %default]')
    parser.add_option('-b', dest='background_fasta',
        default=None, help='Homer background FASTA.')
    parser.add_option('-f', dest='genome_fasta',
        default=None,
        help='Genome FASTA for sequences [Default: %default]')
    parser.add_option('-l', dest='embed_layer',
        default=None, type='int', help='Embed sequences using the specified layer index.')
    parser.add_option('-o', dest='out_dir',
        default='motifs_out',
        help='Output directory [Default: %default]')
    parser.add_option('-p', dest='predict_h5_file',
        default=None, help='basenji_predict output HDF5.')
    parser.add_option('-r', dest='range_step',
        default=1, type='int',
        help='Range step for using activation values [Default: %default]')
    parser.add_option('-s', dest='seqlet_length',
        default=20, type='int',
        help='Seqlet length to extract for motif analysis [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) == 3:
        params_file = args[0]
        model_file = args[1]
        bed_file = args[2]
    else:
        parser.error('Must provide parameter and model files and BED file')

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    ################################################################
    # predict

    if options.predict_h5_file is None:
        cmd = 'basenji_predict_bed.py'
        cmd += ' -f %s' % options.genome_fasta
        cmd += ' -l %d' % options.embed_layer
        cmd += ' -o %s' % options.out_dir
        subprocess.call(cmd, shell=True)
        options.predict_h5_file = '%s/predict.h5' % options.out_dir

    ################################################################
    # read model, sequences, and predictions

    # read params
    with open(params_file) as params_open:
        params = json.load(params_open)

    # build model
    seqnn_model = seqnn.SeqNN(params['model'])
    seqnn_model.restore(model_file)

    # read predictions
    seqlet_acts, seqlet_intervals = read_preds(options.predict_h5_file, range_step=options.range_step)
    seqlet_acts = np.clip(seqlet_acts, 0, np.inf)

    # transform seqlets w/ options.seqlet_length
    seqlet_intervals = extend_intervals(seqlet_intervals, options.seqlet_length)

    # remove seqlets beyond
    seqlet_acts, seqlet_intervals = filter_seqlets(seqlet_acts, seqlet_intervals, options.genome_fasta)

    # extract seqlet DNA
    fasta_open = pysam.Fastafile(options.genome_fasta)
    seqlet_dna = [fasta_open.fetch(sint.chr, sint.start, sint.end) for sint in seqlet_intervals]
    fasta_open.close()

    # remove uninformative filters
    seqlet_acts, feature_mask = filter_features(seqlet_acts, return_mask=True)


    ################################################################
    # features

    features_out_dir = '%s/features' % options.out_dir
    if not os.path.isdir(features_out_dir):
        os.mkdir(features_out_dir)

    # read weights
    kernel_weights = seqnn_model.get_conv_weights(options.embed_layer)

    sfi = 0
    for fi in range(len(feature_mask)):
        print('feature %d' % fi)
        if feature_mask[fi]:
            # plot kernel
            plot_kernel(kernel_weights[fi], '%s/f%d_weights.pdf' % (features_out_dir, fi))

            # plot logo
            plot_logo(seqlet_acts[:,sfi], seqlet_dna,
                      '%s/f%d' % (features_out_dir,fi), options.align_seqlets)

            # homer
            # run_homer(seqlet_acts[:,sfi], seqlet_dna,
            #           '%s/f%d' % (features_out_dir,fi), options.background_fasta)

            sfi += 1

    ################################################################
    # factorized

    factors_out_dir = '%s/factors' % options.out_dir
    if not os.path.isdir(factors_out_dir):
        os.mkdir(factors_out_dir)

    num_factors = seqlet_acts.shape[1]
    seqlet_nmf = nimfa.Nmf(seqlet_acts, rank=num_factors)()

    for fi in range(num_factors):
        print('factor %d' % fi)

        seqlet_basis_fi = seqlet_nmf.basis()[:,fi]

        # write coef vector
        write_factor(seqlet_nmf.coef()[fi,:], feature_mask,
                     '%s/f%d_coef.txt' % (factors_out_dir,fi))

        # plot logo
        plot_logo(seqlet_basis_fi, seqlet_dna,
                  '%s/f%d' % (factors_out_dir,fi), options.align_seqlets)

        # homer
        # run_homer(seqlet_basis_fi, seqlet_dna,
        #           '%s/f%d' % (features_out_dir,fi), options.background_fasta)



    ################################################################
    # clustered

    # compute nearest neighbor graph
    # seqlet_nn = nearest_neighbors(seqlet_acts)

    # compute leiden clustering
    # seqlet_clusters = cluster_leiden(seqlet_nn)


def cluster_leiden(seq_nn, resolution=2):
    # compute leiden clustering
    partition = leidenalg.find_partition(seq_nn,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution)

    # extract cluster memberships into categorial
    membership = np.array(partition.membership)
    clusters = pd.Categorical(
        values=membership.astype('U'),
        categories=natsorted(np.unique(membership).astype('U')))

    return clusters


def extend_intervals(seqlet_intervals, seqlet_length):
    seqlet_intervals_extend = []
    for seq_int in seqlet_intervals:
        mid_pos = (seq_int.end + seq_int.start) // 2
        sie_start = mid_pos - seqlet_length//2
        sie_end = sie_start + seqlet_length
        seqlet_intervals_extend.append(Interval(
            chr=seq_int.chr,
            start=sie_start,
            end=sie_end
            ))
    return seqlet_intervals_extend


def filter_features(act, min_var=1e-6, return_mask=False, verbose=False):
    """ Filter uninformative features with no values or zero variance. """
    filter_mask = np.zeros(act.shape[1], dtype='bool')

    # filter for some activation
    act_sums = act.sum(axis=0)
    nz_mask = (act_sums > 0)
    if verbose:
        sum_filtered = act.shape[-1] - nz_mask.sum()
        print('Filtering %d unactivated features.' % sum_filtered)
    filter_mask = np.logical_or(filter_mask, nz_mask)

    # filter for some variance
    act_vars = act.var(axis=0)
    var_mask = (act_vars > min_var)
    if verbose:
        var_filtered = act.shape[-1] - var_mask.sum()
        print('Filtering %d low variance features.' % var_filtered)
    filter_mask = np.logical_or(filter_mask, var_mask)

    act = act[:,filter_mask]

    if return_mask:
        return act, filter_mask
    else:
        return act


def filter_seqlets(seqlet_acts, seqlet_intervals, genome_fasta_file, end_distance=100, verbose=True):
    """ Filter seqlets by valid chromosome coordinates. """

    # read chromosome lengths
    chr_lengths = {}
    genome_fasta_open = pysam.Fastafile(genome_fasta_file)
    for chrom in genome_fasta_open.references:
        chr_lengths[chrom] = genome_fasta_open.get_reference_length(chrom)
    genome_fasta_open.close()

    # check coordinates
    filter_mask = np.zeros(len(seqlet_intervals), dtype='bool')
    for si, seq_int in enumerate(seqlet_intervals):
        left_valid = (seq_int.start > end_distance)
        right_valid = (seq_int.end + end_distance < chr_lengths[seq_int.chr])
        filter_mask[si] = left_valid and right_valid

    if verbose:
        print('Removing %d seqlets near chromosome ends.' % (len(seqlet_intervals) - filter_mask.sum()))

    # filter
    seqlet_acts = seqlet_acts[filter_mask]
    seqlet_intervals = [seq_int for si, seq_int in enumerate(seqlet_intervals) if filter_mask[si]]

    return seqlet_acts, seqlet_intervals


def make_feature_fasta(seqlet_feature, seqlet_dna, feature_fasta_file, max_pct=0.666):
    """ Write FASTA file of high-scoring seqlets for some feature. """
    feature_t = max_pct*np.max(seqlet_feature)
    feature_fasta_out = open(feature_fasta_file, 'w')
    for si in range(len(seqlet_feature)):
        if seqlet_feature[si] >= feature_t:
            print('>%d' % si, file=feature_fasta_out)
            print(seqlet_dna[si], file=feature_fasta_out)
    feature_fasta_out.close()


def nearest_neighbors(X, neighbors=16, threads=1):
    # initialize HNSW index on Cosine Similarity
    nn_index = nmslib.init(method='hnsw', space='cosinesimil')
    nn_index.addDataPointBatch(X)
    nn_index.createIndex({'post': 2}, print_progress=True)

    # get nearest neighbours
    Xn = nn_index.knnQueryBatch(X, k=(neighbors+1), num_threads=threads)

    # extract graph edges
    sources = []
    targets = []
    for i, neigh in enumerate(Xn):
        sources += [i]*(neighbors-1)
        targets += list(neigh[0][1:])

    # construct igraph
    nn_graph = igraph.Graph(directed=True)
    nn_graph.add_vertices(X.shape[0])
    nn_graph.add_edges(list(zip(sources, targets)))

    return nn_graph


def plot_kernel(kernel_weights, out_pdf):
    depth, width = kernel_weights.shape
    fig_width = 2 + np.log2(width)

    # normalize
    kernel_weights -= kernel_weights.mean(axis=0)

    # plot
    sns.set(font_scale=2)
    plt.figure(figsize=(fig_width, depth))
    sns.heatmap(kernel_weights, cmap='PRGn', linewidths=0.2, center=0)
    ax = plt.gca()
    ax.set_xticklabels(range(1,depth+1))

    if depth == 4:
        ax.set_yticklabels('ACGT', rotation='horizontal')
    else:
        ax.set_yticklabels(range(width), rotation='horizontal')

    plt.savefig(out_pdf)
    plt.close()


def plot_logo(seqlet_acts, seqlet_dna, out_prefix, align_seqlets=False):
    # make feature fasta
    feature_fasta_file = '%s_dna.fa' % out_prefix
    make_feature_fasta(seqlet_acts, seqlet_dna, feature_fasta_file)

    if align_seqlets:
        # write and run multiple sequence alignment
        feature_afasta_file = '%s_msa.fa' % out_prefix
        muscle_cmd = 'muscle -diags -maxiters 2 -in %s -out %s' % (feature_fasta_file, feature_afasta_file)
        subprocess.call(muscle_cmd, shell=True)
    else:
        feature_afasta_file = feature_fasta_file

    # make weblogo
    weblogo_opts = '-X NO -Y NO --errorbars NO --fineprint ""'
    weblogo_opts += ' -C "#CB2026" A A'
    weblogo_opts += ' -C "#34459C" C C'
    weblogo_opts += ' -C "#FBB116" G G'
    weblogo_opts += ' -C "#0C8040" T T'
    weblogo_eps = '%s_logo.eps' % out_prefix
    weblogo_cmd = 'weblogo %s < %s > %s' % (weblogo_opts, feature_afasta_file, weblogo_eps)
    subprocess.call(weblogo_cmd, shell=True)


def read_preds(predict_h5_file, range_step=1, verbose=True):
    """ Read predictions output by basenji_predict_bed.py """
    predict_h5 = h5py.File(predict_h5_file)
    num_seqs, seq_length, num_dim = predict_h5['preds'].shape

    # determine desired subset of positions
    positions = np.arange(0, seq_length, range_step)
    right_dist = seq_length - 1 - positions[-1]
    positions += right_dist // 2

    # read representations
    seqlet_reprs = predict_h5['preds'][:,positions,:]

    # read sequences
    chrom = [chrm.decode('UTF-8') for chrm in predict_h5['chrom']]
    start = predict_h5['start'][:]
    end = predict_h5['end'][:]

    # filter invalid
    seqs_valid = (start >= 0)
    if verbose and seqs_valid.sum() < num_seqs:
        print('Filtering %d sequences running off the end.' % (num_seqs-seqs_valid.sum()))
    num_seqs = seqs_valid.sum()
    seqlet_reprs = seqlet_reprs[seqs_valid]
    chrom = [c for ci, c in enumerate(chrom) if seqs_valid[ci]]
    start = start[seqs_valid]
    end = end[seqs_valid]

    seqlet_intervals = []
    for si in range(num_seqs):
        seq_chrom = chrom[si]
        seq_start = start[si]
        seq_end = end[si]
        seq_window = (seq_end - seq_start) // seq_length

        for pi in positions:
            seqlet_start = seq_start + pi*seq_window
            seqlet_end = seqlet_start + seq_window
            seqlet_interval = Interval(
                chr=seq_chrom,
                start=seqlet_start,
                end=seqlet_end)
            seqlet_intervals.append(seqlet_interval)

    predict_h5.close()

    # reshape and uptype
    seqlet_reprs = seqlet_reprs.reshape((-1,num_dim)).astype('float32')
    if verbose:
        print('Representations: ', seqlet_reprs.shape)

    return seqlet_reprs, seqlet_intervals

def run_homer(seqlet_acts, seqlet_dna, out_prefix, background_fasta):
    # make feature fasta
    feature_fasta_file = '%s_dna.fa' % out_prefix
    make_feature_fasta(seqlet_acts, seqlet_dna, feature_fasta_file)

    homer_opts = '-norevopp -noknown -chopify -noweight -bits'

    feature_homer_dir = '%s_homer' % out_prefix
    cmd = 'findMotifs.pl %s fasta %s -fasta %s %s' % \
            (feature_fasta_file, feature_homer_dir, background_fasta, homer_opts)
    subprocess.call(homer_cmd, shell=True)


def write_factor(factor_coef, feature_mask, out_file):
    """ Write NMF factor in light of feature_mask. """
    out_open = open(out_file, 'w')
    sfi = 0
    for fi in range(len(feature_mask)):
        if feature_mask[fi]:
            print(sfi, factor_coef[sfi], file=out_open)
            sfi += 1
    out_open.close()


Interval = collections.namedtuple('Interval', ['chr', 'start', 'end'])

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
