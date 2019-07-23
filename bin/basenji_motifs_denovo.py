#!/usr/bin/env python
from optparse import OptionParser
import collections
import json
import multiprocessing
import os
import pdb
import random
import subprocess
import time

import h5py
import igraph
import leidenalg
from natsort import natsorted
# import nimfa
import nmslib
import numpy as np
import pysam
from ristretto import nmf

import matplotlib.pyplot as plt
import seaborn as sns

from basenji import seqnn
from basenji import dna_io

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
    parser.add_option('-d', dest='meme_db',
        default='%s/data/motifs/Homo_sapiens.meme' % os.environ['BASSETDIR'],
        help='MEME database used to annotate motifs')
    parser.add_option('-e', dest='embed_layer',
        default=None, type='int',
        help='Embed sequences using the specified layer index.')
    parser.add_option('-f', dest='genome_fasta',
        default=None,
        help='Genome FASTA for sequences [Default: %default]')
    parser.add_option('-l', dest='site_length',
      default=None, type='int',
      help='Prediction site length. [Default: params.seq_length]')
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
    parser.add_option('-t', dest='threads',
        default=1, type='int',
        help='Number of threads [Default: %default]')
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
        cmd += ' -e %d' % options.embed_layer
        cmd += ' -f %s' % options.genome_fasta
        cmd += ' -l %d' % options.site_length
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

    # construct negative seqlets for motif analysis
    negatives_fasta_file = '%s/seqlets_neg.fa' % options.out_dir
    make_negative_fasta(seqlet_intervals, seqlet_dna, negatives_fasta_file)

    # remove uninformative filters
    seqlet_acts, feature_mask = filter_features(seqlet_acts, return_mask=True)


    ################################################################
    # features

    features_out_dir = '%s/features' % options.out_dir
    if not os.path.isdir(features_out_dir):
        os.mkdir(features_out_dir)

    # read weights
    kernel_weights = seqnn_model.get_conv_weights(options.embed_layer)

    feature_args = []

    sfi = 0
    for fi in range(len(feature_mask)):
        # print('feature %d' % fi)
        if feature_mask[fi]:
            fa = (seqlet_acts[:,sfi], seqlet_dna, kernel_weights[fi],
                 '%s/f%d' % (features_out_dir,fi), negatives_fasta_file,
                  options.align_seqlets, options.meme_db)
            feature_args.append(fa)
            sfi += 1

    if options.threads == 1:
        for fa in feature_args:
            process_feature(*fa)
    else:
        mp = multiprocessing.Pool(options.threads)
        mp.starmap(process_feature, feature_args)


    ################################################################
    # factorized

    factors_out_dir = '%s/factors' % options.out_dir
    if not os.path.isdir(factors_out_dir):
        os.mkdir(factors_out_dir)

    num_factors = seqlet_acts.shape[1] // 2
    t0 = time.time()
    print('Computing NMF...', end='')
    # seqlet_nmf = nimfa.Nmf(seqlet_acts, rank=num_factors)()
    seqlet_W, seqlet_H = nmf.compute_rnmf(seqlet_acts, rank=num_factors)
    print('done in %ds' % (time.time()-t0))


    factor_args = []
    for fi in range(num_factors):
        fa = (seqlet_H[fi,:], seqlet_W[:,fi], seqlet_dna, feature_mask,
              '%s/f%d' % (factors_out_dir,fi), negatives_fasta_file,
              options.align_seqlets, options.meme_db)
        factor_args.append(fa)

    if options.threads == 1:
        for fa in factor_args:
            process_factor(*fa)
    else:
        mp.starmap(process_factor, factor_args)


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


def end_align_fasta(seqlets_fasta, msa_fasta, gaps=2, pwm_iter=1, epochs=2):
    """Align seqlets in a FASTA file, allowing only end gaps."""

    # read seqlet DNA
    seqs_1hot = []
    for line in open(seqlets_fasta):
        if line[0] != '>':
            seqlet_dna = line.rstrip()
            seqlet_1hot = dna_io.dna_1hot(seqlet_dna)
            seqs_1hot.append(seqlet_1hot)
    seqs_1hot = np.array(seqs_1hot)
    num_seqs, width, depth = seqs_1hot.shape

    # extend with blanks for shifts
    num_nan = num_seqs*depth
    gap_col = np.array([np.nan]*num_nan).reshape((num_seqs,1,depth))
    msa_1hot = np.concatenate([gap_col, seqs_1hot, gap_col], axis=1)

    # gaps != 2 not implemented
    assert(gaps == 2)

    for ei in range(epochs):
        for si in range(num_seqs):
            if si % pwm_iter == 0:
                pwm = (msa_1hot[:,gaps:-gaps,:] + .1).mean(axis=0)

            # extract sequence
            seq_1hot = seqs_1hot[si]

            # score gap positions
            gap_scores = []
            for gi in range(gaps+1):
                g1 = gaps - gi
                g2 = g1 + pwm.shape[0]
                gap_1hot = seq_1hot[g1:g2]
                gscore = np.log(pwm[gap_1hot]).sum()
                gap_scores.append(gscore)

            # find best
            gi = np.argmax(gap_scores)
            gj = width + gaps - (gaps - gi)

            # set msa
            msa_1hot[si] = np.nan
            msa_1hot[si,gi:gj,:] = seq_1hot

    # write to FASTA
    write_msa(msa_1hot, msa_fasta)

    return msa_1hot


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


def make_negative_fasta(seqlet_intervals, seqlet_dna, negatives_fasta_file, max_seqs=16384):
    """Write sampled seqlets as motif negatives."""

    num_seqs = len(seqlet_intervals)
    if num_seqs <= max_seqs:
        sample_i = np.arange(num_seqs)
    else:
        sample_i = np.random.choice(num_seqs, max_seqs)

    negatives_fasta_out = open(negatives_fasta_file, 'w')
    for si in sample_i:
        sint = seqlet_intervals[si]
        print('>%s:%d-%d\n%s' % (sint.chr, sint.start, sint.end, seqlet_dna[si]), file=negatives_fasta_out)
    negatives_fasta_out.close()


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
    fig_width = 2 + 1.5*np.log2(width)

    # normalize
    kernel_weights -= kernel_weights.mean(axis=0)

    # plot
    sns.set(font_scale=1.5)
    plt.figure(figsize=(fig_width, depth))
    sns.heatmap(kernel_weights, cmap='PRGn', linewidths=0.2, center=0)
    ax = plt.gca()
    ax.set_xticklabels(range(1,width+1))

    if depth == 4:
        ax.set_yticklabels('ACGT', rotation='horizontal')
    else:
        ax.set_yticklabels(range(1,depth+1), rotation='horizontal')

    plt.savefig(out_pdf)
    plt.close()


def plot_logo(seqlet_acts, seqlet_dna, out_prefix, align_seqlets=False):
    # make feature fasta
    feature_fasta_file = '%s_dna.fa' % out_prefix
    make_feature_fasta(seqlet_acts, seqlet_dna, feature_fasta_file)

    if align_seqlets:
        # write and run multiple sequence alignment
        feature_afasta_file = '%s_msa.fa' % out_prefix

        # align
        end_align_fasta(feature_fasta_file, feature_afasta_file)
        # muscle_cmd = 'muscle -diags -maxiters 2 -in %s -out %s' % (feature_fasta_file, feature_afasta_file)
        # subprocess.call(muscle_cmd, shell=True)

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


def process_feature(seqlet_acts, seqlet_dna, kernel_weights, out_prefix, background_fasta, align_seqlets=False, meme_db=None):
    """Perform all analyses on one feature."""
    print(out_prefix)

    # plot kernel weights
    plot_kernel(kernel_weights, '%s_weights.pdf' % out_prefix)

    # plot logo
    plot_logo(seqlet_acts, seqlet_dna, out_prefix, align_seqlets)

    # homer
    run_homer(seqlet_acts, seqlet_dna, out_prefix, background_fasta)

    # meme
    run_dreme(seqlet_acts, seqlet_dna, out_prefix, background_fasta, meme_db)


def process_factor(seqlet_H, seqlet_W, seqlet_dna, feature_mask, out_prefix, background_fasta, align_seqlets=False, meme_db=None):
    """Perform all analyses on one factor."""
    print(out_prefix)

    # write coef vector
    write_factor(seqlet_H, feature_mask, '%s_coef.txt' % out_prefix)

    # plot logo
    plot_logo(seqlet_W, seqlet_dna, out_prefix, align_seqlets)

    # homer
    run_homer(seqlet_W, seqlet_dna, out_prefix, background_fasta)

    # meme
    run_dreme(seqlet_W, seqlet_dna, out_prefix, background_fasta, meme_db)


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


def run_dreme(seqlet_acts, seqlet_dna, out_prefix, background_fasta, meme_db):
    # make feature fasta
    feature_fasta = '%s_dna.fa' % out_prefix
    make_feature_fasta(seqlet_acts, seqlet_dna, feature_fasta)

    # run dreme
    dreme_opts = '-norc -e 0.0001 -m 4'
    dreme_dir = '%s_dreme' % out_prefix
    dreme_cmd = 'dreme %s -p %s -n %s -oc %s' % \
            (dreme_opts, feature_fasta, background_fasta, dreme_dir)

    dreme_std = open('%s_dreme.txt' % out_prefix, 'w')
    subprocess.call(dreme_cmd, stdout=dreme_std, stderr=dreme_std, shell=True)
    dreme_std.close()

    # run tomtom
    if meme_db is not None:
        meme_file = '%s/dreme.txt' % dreme_dir
        tomtom_opts = '-dist pearson -thresh 0.1'
        tomtom_dir = '%s/tomtom' % dreme_dir
        tomtom_cmd = 'tomtom %s -oc %s %s %s' % \
            (tomtom_opts, tomtom_dir, meme_file, meme_db)

        tomtom_std = open('%s/tomtom.txt' % dreme_dir, 'w')
        subprocess.call(tomtom_cmd, stdout=tomtom_std, stderr=tomtom_std, shell=True)
        tomtom_std.close()


def run_homer(seqlet_acts, seqlet_dna, out_prefix, background_fasta):
    # make feature fasta
    feature_fasta_file = '%s_dna.fa' % out_prefix
    make_feature_fasta(seqlet_acts, seqlet_dna, feature_fasta_file)

    homer_opts = '-S 16 -e .02 -minlp -50'
    homer_opts += ' -norevopp -noknown -chopify -noweight -bits -basic'

    homer_dir = '%s_homer' % out_prefix
    homer_cmd = 'findMotifs.pl %s fasta %s -fasta %s %s' % \
            (feature_fasta_file, homer_dir, background_fasta, homer_opts)

    homer_std = open('%s_homer.txt' % out_prefix, 'w')
    subprocess.call(homer_cmd, stdout=homer_std, stderr=homer_std, shell=True)
    homer_std.close()


def write_factor(factor_coef, feature_mask, out_file):
    """ Write NMF factor in light of feature_mask. """
    out_open = open(out_file, 'w')
    sfi = 0
    for fi in range(len(feature_mask)):
        if feature_mask[fi]:
            print(sfi, factor_coef[sfi], file=out_open)
            sfi += 1
    out_open.close()


def write_msa(msa_1hot, fasta_file):
    """Write a multiple sequence alignment, stored as a numpy
    array with NaN for gaps, to FASTA."""
    fasta_open = open(fasta_file, 'w')

    num_seqs, width, depth = msa_1hot.shape
    for si in range(num_seqs):
        mseq = []
        for wi in range(width):
            if np.isnan(msa_1hot[si,wi,0]):
                mseq.append('-')
            elif msa_1hot[si,wi,0]:
                mseq.append('A')
            elif msa_1hot[si,wi,1]:
                mseq.append('C')
            elif msa_1hot[si,wi,2]:
                mseq.append('G')
            else:
                mseq.append('T')
        print('>%d\n%s' % (si, ''.join(mseq)), file=fasta_open)

    fasta_open.close()

Interval = collections.namedtuple('Interval', ['chr', 'start', 'end'])

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
