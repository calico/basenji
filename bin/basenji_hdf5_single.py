#!/usr/bin/env python
# Copyright 2017 Calico LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from optparse import OptionParser
from collections import OrderedDict
import math
import multiprocessing
import os
import random
import subprocess
import sys
import tempfile
import time

import h5py
import joblib
import numpy as np
import pyBigWig
import pysam
import tensorflow as tf

import basenji

'''
basenji_hdf5_single.py

Tile the genome and project the full functional profile to latent space
using a given model. Save the result in HDF5 for Basenji learning.
'''

################################################################################
def main():
    usage = 'usage: %prog [options] <fasta_file> <sample_wigs_file> <hdf5_file>'
    parser = OptionParser(usage)
    parser.add_option('-b', dest='limit_bed', help='Limit to segments that overlap regions in a BED file')
    parser.add_option('-c', dest='clip', default=None, type='float', help='Clip target values to have minimum [Default: %default]')
    parser.add_option('-d', dest='sample_pct', default=1.0, type='float', help='Down-sample the segments')
    parser.add_option('-f', dest='fourier_dim', default=None, type='int', help='Fourier transform dimension [Default: %default]')
    parser.add_option('-g', dest='gaps_file', help='Genome assembly gaps BED [Default: %default]')
    parser.add_option('-l', dest='seq_length', default=131072, type='int', help='Sequence length [Default: %default]')
    parser.add_option('--log2', dest='log10to2', default=False, action='store_true', help='Transform values from log10 to log2 [Default: %default]')
    parser.add_option('-m', dest='params_file', help='Dimension reduction hyper-parameters file')
    parser.add_option('-n', dest='na_t', default=0.25, type='float', help='Remove sequences with an NA% greater than this threshold [Default: %default]')
    parser.add_option('--no_full', dest='no_full', default=False, action='store_true', help='Do not save full test sequence targets [Default: %default]')
    parser.add_option('-o', dest='out_bed_file', help='Output the train/valid/test sequences as a BED file')
    parser.add_option('-p', dest='processes', default=1, type='int', help='Number parallel processes to load data [Default: %default]')
    parser.add_option('-s', dest='stride', default=None, type='int', help='Stride to advance segments [Default: seq_length]')
    parser.add_option('--scent', dest='scent_file', help='Dimension reduction model file')
    parser.add_option('-t', dest='test_pct_or_chr', type='str', default=0.05, help='Proportion of the data for testing [Default: %default]')
    parser.add_option('-u', dest='unmap_bed', help='Unmappable segments to set to NA')
    parser.add_option('-w', dest='pool_width', type='int', default=128, help='Average pooling width [Default: %default]')
    parser.add_option('--w5', dest='w5', default=False, action='store_true', help='Coverage files are w5 rather than BigWig [Default: %default]')
    parser.add_option('-v', dest='valid_pct_or_chr', type='str', default=0.05, help='Proportion of the data for validation [Default: %default]')
    parser.add_option('-z', dest='compression', help='h5py compression [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 3:
        parser.error('Must provide genome file, sample Wig/BigWig labels and paths, and model output file')
    else:
        fasta_file = args[0]
        sample_wigs_file = args[1]
        hdf5_file = args[2]

    random.seed(1)

    ################################################################
    # assess bigwigs
    ################################################################
    # get wig files and labels
    target_wigs = OrderedDict()
    for line in open(sample_wigs_file):
        a = line.split()
        target_wigs[a[0]] = a[1]

    if options.fourier_dim is not None and 2*options.fourier_dim >= options.seq_length/options.pool_width:
        print("Fourier transform to %d dims won't compress %d length sequences with %d pooling" % (options.fourier_dim, options.seq_length, options.pool_width), file=sys.stderr)
        exit(1)


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

    # standardize order
    segments.sort()

    # filter for large enough
    segments = [cse for cse in segments if cse[2]-cse[1] >= options.seq_length]

    # down-sample
    if options.sample_pct < 1.0:
        segments = random.sample(segments, int(options.sample_pct*len(segments)))

    # limit to a BED file
    if options.limit_bed is not None:
        segments = limit_segments(segments, options.limit_bed)


    ################################################################
    # one hot code sequences
    ################################################################
    seqs_1hot, seqs_segments = segments_1hot(fasta_file, segments, options.seq_length, options.stride)
    print('%d sequences one hot coded' % seqs_1hot.shape[0])


    ################################################################
    # load model
    ################################################################
    if options.params_file:
        job = basenji.dna_io.read_job_params(options.params_file)
        job['num_targets'] = len(target_wigs)
        job['batch_size'] = 1024
        job['model'] = job.get('model','autoencoder')

        if job['model'] == 'autoencoder':
            model = basenji.autoencoder.AE(job)
            saver = tf.train.Saver()
        else:
            model = joblib.load(options.scent_file)


    ################################################################
    # bigwig read and process
    ################################################################
    print('Reading and pre-processing bigwigs for %d segments' % len(segments), flush=True)

    targets_real = []
    targets_imag = []

    include_indexes = []
    include_marker = 0

    targets_test = []
    test_indexes = []
    test_marker = 0

    update_i = 0
    ssi = 0

    # initialize multiprocessing pool
    pool = multiprocessing.Pool(options.processes)

    with tf.Session() as sess:
        if options.scent_file and job['model'] == 'autoencoder':
            saver.restore(sess, options.scent_file)

        # batch segment processing
        bstart = 0
        while bstart < len(segments):
            if update_i % 1 == 0:
                print('Tiling from %s:%d-%d' % segments[bstart], flush=True)

            # determine batch end
            bend = batch_end(segments, bstart, 400000)

            # bigwig_read parameters
            bwr_params = [(wig_file, segments[bstart:bend], options.seq_length, options.pool_width, options.stride, options.log10to2, options.cov_multiplier) for wig_file in target_wigs.values()]

            # pull the target values in parallel
            if options.w5:
                wig_targets = pool.starmap(w5_batch, bwr_params)
            else:
                wig_targets = pool.starmap(bigwig_batch, bwr_params)

            # transpose to S x L x T (making a copy?)
            targets_wig = np.transpose(np.array(wig_targets), axes=(1,2,0))

            # clip
            if options.clip is not None:
                targets_wig = targets_wig.clip(options.clip)

            # sample indexes from this batch
            if options.test_pct_or_chr.startswith('chr'):
                test_bindexes = [twi for twi in range(targets_wig.shape[0]) if seqs_segments[ssi+twi] == options.test_pct_or_chr]
            else:
                test_pct = float(options.test_pct_or_chr)
                test_bindexes = [twi for twi in range(targets_wig.shape[0]) if random.random() < test_pct]


            # capture test indexes
            test_indexes += [test_marker+tbi for tbi in test_bindexes]

            # update test marker
            test_marker += targets_wig.shape[0]

            # save the full test targets
            if not options.no_full:
                targets_test.append(targets_wig[test_bindexes])

            # map to latent space
            if options.scent_file is None:
                targets_latent = targets_wig
            else:
                targets_latent = latent_transform(sess, model, job, targets_wig)

            # compress across length
            if options.fourier_dim is None:
                targets_rfour = targets_latent
                targets_ifour = None
            else:
                targets_rfour, targets_ifour = fourier_transform(targets_latent, options.fourier_dim)

            # save
            targets_real.append(targets_rfour)
            targets_imag.append(targets_ifour)

            # update seqs_segments index
            ssi += targets_wig.shape[0]

            # update batch
            bstart = bend
            update_i += 1

    pool.close()

    # stack arrays
    targets_real = np.vstack(targets_real)
    if options.fourier_dim is not None:
        targets_imag = np.vstack(targets_imag)
    if not options.no_full:
        targets_test = np.vstack(targets_test)

    print('%d target sequences' % targets_real.shape[0])


    ################################################################
    # correct for unmappable regions
    ################################################################
    if options.unmap_bed is not None:
        seqs_na = annotate_na(seqs_segments, options.unmap_bed, options.seq_length, options.pool_width)

        # determine mappable sequences and update test indexes
        map_indexes = []
        test_indexes_na = []
        test_i = 0

        for i in range(seqs_na.shape[0]):
            # mappable
            if seqs_na[i,:].mean() < options.na_t:
                map_indexes.append(i)

                if i in test_indexes:
                    test_indexes_na.append(test_i)

                test_i += 1

            # unmappable
            else:
                # forget it
                pass

        # update data structures
        targets_real = targets_real[map_indexes]
        if options.fourier_dim is not None:
            targets_imag = targets_imag[map_indexes]

        seqs_1hot = seqs_1hot[map_indexes]
        seqs_segments = [seqs_segments[mi] for mi in map_indexes]
        seqs_na = seqs_na[map_indexes]

        test_indexes = test_indexes_na


    ################################################################
    # write to train, valid, test HDF5
    ################################################################

    if options.valid_pct_or_chr.startswith('chr'):
        # sample valid chromosome
        valid_indexes = [si for si in range(len(seqs_segments)) if seqs_segments[si][0] == options.valid_pct_or_chr]

    else:
        # sample valid indexes (we already have test)
        valid_pct = float(options.valid_pct_or_chr)
        valid_n = int(options.valid_pct*targets_real.shape[0])
        nontest_indexes = set(range(targets_real.shape[0])) - set(test_indexes)
        valid_indexes = random.sample(nontest_indexes, valid_n)

    # remainder is training
    train_indexes = list(set(range(len(seqs_segments))) - set(valid_indexes) - set(test_indexes))

    # training may requires shuffle
    random.shuffle(train_indexes)
    random.shuffle(valid_indexes)
    random.shuffle(test_indexes)

    # write to HDF5
    hdf5_out = h5py.File(hdf5_file, 'w')

    # store pooling
    hdf5_out.create_dataset('pool_width', data=options.pool_width, dtype='int')

    # store targets
    target_labels = np.array(list(target_wigs.keys()), dtype='S')
    hdf5_out.create_dataset('target_labels', data=target_labels)

    # HDF5 train
    hdf5_out.create_dataset('train_in', data=seqs_1hot[train_indexes], dtype='bool', compression=options.compression)
    hdf5_out.create_dataset('train_out', data=targets_real[train_indexes], dtype='float16', compression=options.compression)
    if options.fourier_dim is not None:
        hdf5_out.create_dataset('train_out_imag', data=targets_imag[train_indexes], dtype='float16', compression=options.compression)
    hdf5_out.create_dataset('train_na', data=seqs_na[train_indexes], dtype='bool', compression=options.compression)

    # HDF5 valid
    hdf5_out.create_dataset('valid_in', data=seqs_1hot[valid_indexes], dtype='bool', compression=options.compression)
    hdf5_out.create_dataset('valid_out', data=targets_real[valid_indexes], dtype='float16', compression=options.compression)
    if options.fourier_dim is not None:
        hdf5_out.create_dataset('valid_out_imag', data=targets_imag[valid_indexes], dtype='float16', compression=options.compression)
    hdf5_out.create_dataset('valid_na', data=seqs_na[valid_indexes], dtype='bool', compression=options.compression)

    # HDF5 test
    hdf5_out.create_dataset('test_in', data=seqs_1hot[test_indexes], dtype='bool', compression=options.compression)
    hdf5_out.create_dataset('test_out', data=targets_real[test_indexes], dtype='float16', compression=options.compression)
    if options.fourier_dim is not None:
        hdf5_out.create_dataset('test_out_imag', data=targets_imag[test_indexes], dtype='float16', compression=options.compression)
    if not options.no_full:
        hdf5_out.create_dataset('test_out_full', data=targets_test, dtype='float16', compression=options.compression)
    hdf5_out.create_dataset('test_na', data=seqs_na[test_indexes], dtype='bool', compression=options.compression)

    hdf5_out.close()

    # output BED file
    if options.out_bed_file:
        out_bed_out = open(options.out_bed_file, 'w')
        for si in train_indexes:
            print('%s\t%d\t%d\ttrain' % seqs_segments[si], file=out_bed_out)
        for si in valid_indexes:
            print('%s\t%d\t%d\tvalid' % seqs_segments[si], file=out_bed_out)
        for si in test_indexes:
            print('%s\t%d\t%d\ttest' % seqs_segments[si], file=out_bed_out)
        out_bed_out.close()


################################################################################
def annotate_na(seqs_segments, unmap_bed, seq_length, pool_width):
    ''' Intersect the sequence segments with unmappable regions
         and annoate the segments as NaN to possible be ignored.

    Args:
      seqs_segments: list of (chrom,start,end) sequence segments
      unmap_bed: unmappable regions BED file
      seq_length: sequence length
      pool_width: pooled bin width

    Returns:
      seqs_na: NxL binary NA indicators
    '''

    # print sequence segments to file
    segs_temp = tempfile.NamedTemporaryFile()
    segs_bed = segs_temp.name
    segs_out = open(segs_bed, 'w')
    for (chrom, start, end) in seqs_segments:
        print('%s\t%d\t%d' % (chrom,start,end), file=segs_out)
    segs_out.close()

    # hash segments to indexes
    segment_indexes = {}
    for i in range(len(seqs_segments)):
        segment_indexes[seqs_segments[i]] = i

    # initialize NA array
    seqs_na = np.zeros((len(seqs_segments),seq_length//pool_width), dtype='bool')

    # intersect with unmappable regions
    p = subprocess.Popen('bedtools intersect -wo -a %s -b %s' % (segs_bed, unmap_bed), shell=True, stdout=subprocess.PIPE)
    for line in p.stdout:
        line = line.decode("utf-8")
        a = line.split()

        seg_chrom = a[0]
        seg_start = int(a[1])
        seg_end = int(a[2])
        seg_tup = (seg_chrom,seg_start,seg_end)

        unmap_start = int(a[4])
        unmap_end = int(a[5])

        seg_unmap_start_i = math.floor((unmap_start - seg_start) / pool_width)
        seg_unmap_end_i = math.ceil((unmap_end - seg_start) / pool_width)

        seqs_na[segment_indexes[seg_tup],seg_unmap_start_i:seg_unmap_end_i] = True

    return seqs_na


################################################################################
def batch_end(segments, bstart, batch_max):
    ''' Determine the batch end that will keep the
          batch length under the given max. '''

    bi = bstart
    blength = 0

    while bi < len(segments) and blength < batch_max:
        chrom, seg_start, seg_end = segments[bi]
        blength += seg_end - seg_start
        bi += 1

    bend = bi
    if bstart >= bend or bend > len(segments):
        print("I've made a terrible mistake. On batching segments", file=sys.stderr)
        exit(1)

    return bend


################################################################################
def bigwig_batch(wig_file, segments, seq_length, pool_width=1, stride=None, log10to2=False, cov_multiplier=1):
    ''' Read a batch of segment values from a bigwig file

    Args:
      wig_file: Bigwig filename
      segments: list of (chrom,start,end) genomic segments to read,
                  assuming those segments are appropriate length
      seq_length: sequence length to break them into
      pool_width: average pool adjacent nucleotides of this width
      stride: advance the sequences by this amount.

    Returns:
      targets: target Bigwig value matrix
    '''

    # set stride
    if stride is None:
        stride = seq_length

    # initialize target values
    targets = []

    # open wig
    wig_in = pyBigWig.open(wig_file)

    for chrom, seg_start, seg_end in segments:
        # read values
        try:
            seg_values = np.array(wig_in.values(chrom, seg_start, seg_end), dtype='float16')
        except:
            print("WARNING: %s doesn't see %s:%d-%d. Setting to all zeros." % (wig_file,chrom,seg_start,seg_end))
            # seg_values = np.array([np.nan]*(seg_end-seg_start), dtype='float16')
            seg_values = np.zeros(seg_end-seg_start, dtype='float16')

        # set NaN's to zero
        seg_values = np.nan_to_num(seg_values)

        # transform
        if cov_multiplier != 1:
            seg_values *= cov_multiplier
        if log10to2:
            seg_values = np.log2(np.power(10,seg_values))

        # break up into batchable sequences (as below in segments_1hot)
        bstart = 0
        bend = bstart + seq_length
        while bend <= len(seg_values):
            # extract (and pool)
            if pool_width == 1:
                sv = seg_values[bstart:bend]
            else:
                sv = seg_values[bstart:bend].reshape((seq_length//pool_width, pool_width)).sum(axis=1)

            # append
            targets.append(sv)

            # update
            bstart += stride
            bend += stride

    return targets


################################################################################
def limit_segments(segments, filter_bed):
    ''' Limit to segments overlapping the given BED.

    Args
     segments: list of (chrom,start,end) genomic segments
     filter_bed: BED file to filter by

    Returns:
     fsegments: list of (chrom,start,end) genomic segments
    '''

    # print segments to BED
    seg_fd, seg_bed_file = tempfile.mkstemp()
    seg_bed_out = open(seg_bed_file, 'w')
    for chrom, seg_start, seg_end in segments:
        print('%s\t%d\t%d' % (chrom, seg_start, seg_end), file=seg_bed_out)
    seg_bed_out.close()

    # intersect w/ filter_bed
    fsegments = []
    p = subprocess.Popen('bedtools intersect -u -a %s -b %s' % (seg_bed_file, filter_bed), shell=True, stdout=subprocess.PIPE)
    for line in p.stdout:
        a = line.decode('utf-8').split()
        chrom = a[0]
        seg_start = int(a[1])
        seg_end = int(a[2])
        fsegments.append((chrom,seg_start,seg_end))

    p.communicate()

    os.close(seg_fd)
    os.remove(seg_bed_file)

    return fsegments


################################################################################
def filter_boring(targets, var_t=.01):
    ''' Filter boring segments without signal variance.

    Args
     targets: SxLxT array of target values
     var_t: Average variance threshold

    Returns:
     targets_exciting: SxLxT array of target values
    '''
    target_lvar_max = targets.var(axis=1).max(axis=1)
    exciting_mask = (target_lvar_max > var_t)
    return targets[exciting_mask]


################################################################################
def fourier_transform(targets, dim):
    ''' Fourier transform.

    Args
     targets: SxLxT array of target values
     dim: # of fourier dimensions

    Returns:
     fourier_real: transformed targets, real component
     fourier_imag: transformed targets, imaginary component
    '''
    tn = targets.shape[2]
    fourier_real = np.zeros((targets.shape[0],dim,tn), dtype='float16')
    fourier_imag = np.zeros((targets.shape[0],dim,tn), dtype='float16')

    for ti in range(tn):
        fourier_ti = np.fft.rfft(targets[:,:,ti])[:,:dim]
        fourier_real[:,:,ti] = fourier_ti.real.astype('float16')
        fourier_imag[:,:,ti] = fourier_ti.imag.astype('float16')

    return fourier_real, fourier_imag


################################################################################
def latent_transform(sess, model, job, targets_wig):
    ''' Transform raw data to latent representation.

    Args
     sess: TensorFlow session
     model: TF or sklearn model
     job: dictionary of model hyper-parameters
     targets_wig: SxLxT array of target values

    Returns:
     targets_latent: SxDxT array of target values
    '''

    S, L, T = targets_wig.shape
    targets_length = targets_wig.reshape((S*L, T))

    if job['model'] == 'pca':
        targets_length_latent = model.transform(targets_length)
    else:
        batcher = basenji.batcher.BatcherT(targets_length, model.batch_size)
        targets_length_latent = model.latent(sess, batcher)

    targets_latent = targets_length_latent.reshape((S,L,job['latent_dim']))

    return targets_latent


################################################################################
def segments_1hot(fasta_file, segments, seq_length, stride):
    ''' Read and 1-hot code sequences in their segment batches.

    Args
     fasta_file: FASTA genome
     segments: list of (chrom,start,end) genomic segments to read
     seq_length: sequence length to break them into

    Returns:
     seqs_1hot: You know.
     seqs_segments: list of (chrom,start,end) sequence segments
    '''

    # open fasta
    fasta = pysam.Fastafile(fasta_file)

    # initialize 1-hot coding list
    seqs_1hot = []

    # segment corresponding to each sequence
    seqs_segments = []

    for chrom, seg_start, seg_end in segments:
        # read sequence
        seg_seq = fasta.fetch(chrom, seg_start, seg_end)

        # break up into batchable sequences (as above in bigwig_batch)
        bstart = 0
        bend = bstart + seq_length
        while bend < len(seg_seq):
            # append
            seqs_1hot.append(basenji.dna_io.dna_1hot(seg_seq[bstart:bend]))

            seqs_segments.append((chrom,seg_start+bstart,seg_start+bend))

            # update
            bstart += stride
            bend += stride

    return np.array(seqs_1hot), seqs_segments


################################################################################
def w5_batch(w5_file, segments, seq_length, pool_width=1, stride=None, log10to2=False, cov_multiplier=1):
    ''' Read a batch of segment values from a bigwig file

    Args:
      w5_file: wiggle HDF5 filename
      segments: list of (chrom,start,end) genomic segments to read,
                  assuming those segments are appropriate length
      seq_length: sequence length to break them into
      pool_width: average pool adjacent nucleotides of this width
      stride: advance the sequences by this amount.

    Returns:
      targets: target Bigwig value matrix
    '''

    # set stride
    if stride is None:
        stride = seq_length

    # initialize target values
    targets = []

    # open wig h5
    w5_in = h5py.File(w5_file)

    for chrom, seg_start, seg_end in segments:
        if chrom in w5_in:
            seg_values = w5_in[chrom][seg_start:seg_end]
        else:
            print("WARNING: %s doesn't see %s:%d-%d. Setting to all zeros." % (w5_file,seg_chrom,seg_start,seg_end))
            seg_values = np.zeros(seg_end-seg_start, dtype='float16')

        # set NaN's to zero
        seg_values = np.nan_to_num(seg_values)

        # transform
        if cov_multiplier != 1:
            seg_values *= cov_multiplier
        if log10to2:
            seg_values = np.log2(np.power(10,seg_values))

        # break up into batchable sequences (as below in segments_1hot)
        bstart = 0
        bend = bstart + seq_length
        while bend <= len(seg_values):
            # extract (and pool)
            if pool_width == 1:
                sv = seg_values[bstart:bend]
            else:
                sv = seg_values[bstart:bend].reshape((seq_length//pool_width, pool_width)).sum(axis=1)

            # append
            targets.append(sv)

            # update
            bstart += stride
            bend += stride

    w5_in.close()

    return targets


################################################################################
if __name__ == '__main__':
    main()
