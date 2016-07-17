#!/usr/bin/env python
from optparse import OptionParser
from collections import OrderedDict
import multiprocessing
import sys

import h5py
import numpy as np
import pyBigWig
import pysam
import tensorflow as tf

import basenji

'''
basenji_hdf5.py

Tile the genome and project the full functional profile to latent space
using a given model. Save the result in HDF5 for Basenji learning.

To Do:
 -If unmappable regions are cutting my data, I could squeeze a little more out
   by allowing them to sit at the edges of sequences where I'm not making
   predictions anyway.
'''

################################################################################
def main():
    usage = 'usage: %prog [options] <fasta_file> <sample_wigs_file> <params_file> <model_file> <hdf5_file>'
    parser = OptionParser(usage)
    parser.add_option('-g', dest='gaps_file', help='Genome assembly gaps BED [Default: %default]')
    parser.add_option('-l', dest='seq_length', default=1000, type='int', help='Sequence length [Default: %default]')
    parser.add_option('-p', dest='processes', default=1, type='int', help='Number parallel processes to load data [Default: %default]')
    parser.add_option('-t', dest='test_pct', type='float', default=0.05, help='Proportion of the data for testing [Default: %default]')
    parser.add_option('-v', dest='valid_pct', type='float', default=0.05, help='Proportion of the data for validation [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 3:
        parser.error('Must provide genome file, sample Wig/BigWig labels and paths, and model output file')
    else:
        fasta_file = args[0]
        sample_wigs_file = args[1]
        params_file = args[2]
        model_file = args[3]
        hdf5_file = args[4]


    ################################################################
    # prepare genomic segments
    ################################################################
    chrom_segments = basenji.genome.load_chromosomes(fasta_file)

    # remove gaps
    if options.gaps_file:
        chrom_segments = basenji.genome.split_contigs(chrom_segments, options.gaps_file)

    # ditch the chromosomes
    segments = []
    for chrom in chrom_segments:
        segments += [(chrom, seg_start, seg_end) for seg_start, seg_end in chrom_segments[chrom]]


    ################################################################
    # one hot code sequences
    ################################################################
    seqs_1hot = segments_1hot(segments, options.seq_length, fasta_file)

    print('%d sequences' % seqs_1hot.shape[0])


    ################################################################
    # load model
    ################################################################
    job = basenji.dna_io.read_job_params(params_file)
    job['num_targets'] = targets.shape[1]
    job['batch_size'] = 1024

    model = basenji.autoencoder.AE(job)


    ################################################################
    # bigwig read and process
    ################################################################
    # get wig files and labels
    target_wigs = OrderedDict()
    for line in open(sample_wigs_file):
        a = line.split()
        target_wigs[a[0]] = a[1]
    num_targets = len(target_wigs)

    # initialize multiprocessing pool
    pool = multiprocessing.Pool(options.processes)

    # batch segment processing
    bstart = 0
    while bstart < len(segments):
        # determine batch end
        bend = batch_end(segments, bstart, 200000)

        # bigwig_read parameters
        bwr_params = [(wig_file, segments[bstart:bend], options.seq_length) for wig_file in target_wigs.values()]

        # pull the target values in parallel
        wig_targets = pool.starmap(bigwig_batch, bwr_params)

        # convert and transpose to S x L x T
        wig_targets = np.array(wig_targets)
        targets_wig = np.transpose(wig_targets, axes=(1,2,0))

        # map to latent space
        batcher = basenji.batcher.BatcherT(targets_wig, model.batch_size)
        targets_latent.append(model.latent(sess, batcher))

        # update batch
        bstart = bend

    # convert list to array
    targets_latent = np.array(targets_latent)


    ################################################################
    # write to train, valid, test HDF5
    ################################################################
    # determine # of each
    total_n = seqs_1hot.shape[0]
    valid_n = options.valid_pct*total_n
    test_n = options.test_pct*total_n
    train_n = total_n - valid_n - test_n

    # shuffle (little nervous about memory here)
    order = np.random.permutation(total_n)
    seqs_1hot = seqs_1hot[order]
    targets_latent = targets_latent[order]

    # write to HDF5
    hdf5_out = h5py.File(hdf5_file, 'w')

    # train
    hdf5_out.create_dataset('train_in', seqs_1hot[:train_n])
    hdf5_out.create_dataset('train_out', targets_latent[:train_n])

    # valid
    vi = train_n
    hdf5_out.create_dataset('valid_in', seqs_1hot[vi:vi+valid_n])
    hdf5_out.create_dataset('valid_out', targets_latent[vi:vi+valid_n])

    # test
    ti = train_n + valid_n
    hdf5_out.create_dataset('test_in', seqs_1hot[ti:])
    hdf5_out.create_dataset('test_out', targets_latent[ti:])

    hdf5_out.close()


################################################################################
def batch_end(segments, bstart, batch_max):
    ''' Determine the batch end that will keep the
          batch length under the given max. '''

    bi = bstart
    blength = 0

    while bi < len(segments) and blength < batch_max:
        seg_start, seg_end = segments[bi]
        blength += seg_end - seg_start
        bi += 1

    bend = bi
    if bstart >= bend or bend > len(segments):
        print("I've made a terrible mistake. On batching segments", file=sys.stderr)
        exit(1)

    return bend


################################################################################
def bigwig_batch(wig_file, segments, seq_length):
    ''' Read a batch of segment values from a bigwig file

    Args:
      wig_file: Bigwig filename
      segments: list of (chrom,start,end) genomic segments to read
      seq_length: sequence length to break them into

    Returns:
      targets: target Bigwig value matrix
    '''

    # initialize target values
    targets = []

    # open wig
    wig_in = pyBigWig.open(wig_file)

    for chrom, seg_start, seg_end in segments:
        # read values
        seg_values = wig_in.values(chrom, seg_start, seg_end)

        # break up into batchable sequences (as below in segments_1hot)
        seq_start = 0
        seq_end = seq_start + seq_length
        while seq_end < len(seg_values):
            # append
            targets.append(seg_values[seq_start:seq_end])

            # update
            seq_start += batch_length
            seq_end += batch_length

    return targets


################################################################################
def segments_1hot(fasta_file, segments, seq_length):
    ''' Read and 1-hot code sequences in their segment batches.

    Args
     fasta_file: FASTA genome
     segments: list of (chrom,start,end) genomic segments to read
     seq_length: sequence length to break them into

    Returns:
     seqs_1hot: You know.
    '''

    # open fasta
    fasta = pysam.Fastafile(fasta_file)

    # initialize 1-hot coding list
    seqs_1hot = []

    for chrom, seg_start, seg_end in segments:
        # read sequence
        seg_seq = fasta.fetch(chrom, seg_start, seg_end)

        # break up into batchable sequences (as above in bigwig_batch)
        seq_start = 0
        seq_end = seq_start + seq_length
        while seq_end < len(seg_seq):
            # append
            seqs_1hot.append(basenji.dna_io.dna_1hot(seg_seq[seq_start:seq_end]))

            # update
            seg_start += batch_length
            seg_end += batch_length

    return np.array(seqs_1hot)


################################################################################
if __name__ == '__main__':
    main()
