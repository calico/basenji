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

import slurm

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
    usage = 'usage: %prog [options] <fasta_file> <sample_wigs_file> <hdf5_file>'
    parser = OptionParser(usage)
    parser.add_option('-b', dest='limit_bed', help='Limit to segments that overlap regions in a BED file')
    parser.add_option('-c', dest='clip', default=None, type='float', help='Clip target values to have minimum [Default: %default]')
    parser.add_option('--cluster_dir', dest='cluster_dir', default='basenji_hdf5')
    parser.add_option('-d', dest='sample_pct', default=1.0, type='float', help='Down-sample the segments')
    parser.add_option('-f', dest='fourier_dim', default=None, type='int', help='Fourier transform dimension [Default: %default]')
    parser.add_option('-g', dest='gaps_file', help='Genome assembly gaps BED [Default: %default]')
    parser.add_option('-l', dest='seq_length', default=1024, type='int', help='Sequence length [Default: %default]')
    parser.add_option('--log2', dest='log10to2', default=False, action='store_true', help='Transform values from log10 to log2 [Default: %default]')
    parser.add_option('--mult_cov', dest='cov_multiplier', default=1, type='float', help='Coverage multiplier, useful when the read extension and pool width do not match [Default: %default]')
    parser.add_option('-n', dest='na_t', default=0.25, type='float', help='Remove sequences with an NA% greater than this threshold [Default: %default]')
    parser.add_option('-o', dest='out_bed_file', help='Output the train/valid/test sequences as a BED file')
    parser.add_option('-p', dest='processes', default=1, type='int', help='Number parallel processes to load data [Default: %default]')
    parser.add_option('-s', dest='stride', type='int', help='Stride to advance segments [Default: seq_length]')
    parser.add_option('-t', dest='test_pct_or_chr', type='str', default=0.05, help='Proportion of the data for testing [Default: %default]')
    parser.add_option('-u', dest='unmap_bed', help='Unmappable segments to set to NA')
    parser.add_option('-w', dest='pool_width', type='int', default=1, help='Average pooling width [Default: %default]')
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

    if options.stride is None:
        options.stride = options.seq_length

    ################################################################
    # assess bigwigs
    ################################################################
    # get wig files and labels
    target_wigs = OrderedDict()
    for line in open(sample_wigs_file, encoding='UTF-8'):
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

    if not os.path.isdir(options.cluster_dir):
        os.mkdir(options.cluster_dir)

    # print segments to BED file
    seg_bed_file = '%s/segments.bed' % options.cluster_dir
    seg_bed_out = open(seg_bed_file, 'w')
    for chrom, seg_start, seg_end in segments:
        print('%s\t%d\t%d' % (chrom,seg_start,seg_end), file=seg_bed_out)
    seg_bed_out.close()


    ################################################################
    # bigwig read and process
    ################################################################
    print('Reading and pre-processing bigwigs for %d segments' % len(segments), flush=True)

    targets_real = []
    targets_imag = []

    # generate numpy arrays on cluster
    jobs = []
    for target_label in target_wigs.keys():
        wig_file = target_wigs[target_label]
        npy_file = '%s/%s' % (options.cluster_dir, target_label)
        if not os.path.isfile(npy_file) and not os.path.isfile('%s.npy'%npy_file):
            print(npy_file)

            if os.path.splitext(wig_file)[1] == 'h5'::
                script = 'seqs_hdf5.py'
            else:
                script = 'bigwig_hdf5.py'

            cmd = 'echo $HOSTNAME; %s -l %d -s %d -w %d %s %s %s' % (script, options.seq_length, options.stride, options.pool_width, wig_file, seg_bed_file, npy_file)
            name = 'hdf5_%s'%target_label
            outf = '%s/%s.out' % (options.cluster_dir, target_label)
            errf = '%s/%s.err' % (options.cluster_dir, target_label)
            j = slurm.Job(cmd, name, outf, errf, queue='flash', mem=16000, time='4:0:0')
            jobs.append(j)

    slurm.multi_run(jobs)

    # load into targets_real, targets_imag
    for target_label in target_wigs.keys():
        npy_file = '%s/%s.npy' % (options.cluster_dir, target_label)
        wig_targets = np.load(npy_file)
        targets_real.append(wig_targets)

    # transpose from TxSxL to SxLxT
    targets_real = np.transpose(np.array(targets_real), axes=(1,2,0))

    print('%d target sequences' % targets_real.shape[0])


    ################################################################
    # one hot code sequences
    ################################################################
    seqs_1hot, seqs_segments = segments_1hot(fasta_file, segments, options.seq_length, options.stride)
    print('%d sequences one hot coded' % seqs_1hot.shape[0])


    ################################################################
    # correct for unmappable regions
    ################################################################
    if options.unmap_bed is not None:
        seqs_na = annotate_na(seqs_segments, options.unmap_bed, options.seq_length, options.pool_width)

        # determine mappable sequences and update test indexes
        map_indexes = []
        test_i = 0

        for i in range(seqs_na.shape[0]):
            # mappable
            if seqs_na[i,:].mean() < options.na_t:
                map_indexes.append(i)

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


    ################################################################
    # write to train, valid, test HDF5
    ################################################################

    # choose test indexes
    if options.test_pct_or_chr.startswith('chr'):
        test_indexes = [si for si in range(len(seqs_segments)) if seqs_segments[si][0] == options.test_pct_or_chr]
    else:
        test_pct = float(options.test_pct_or_chr)
        test_indexes = [twi for twi in range(len(seqs_segments)) if random.random() < test_pct]

    # choose valid indexes
    if options.valid_pct_or_chr.startswith('chr'):
        # valid_indexes = np.array([seq_seg[0] == options.valid_pct_or_chr for seq_seg in seqs_segments])
        valid_indexes = [si for si in range(len(seqs_segments)) if seqs_segments[si][0] == options.valid_pct_or_chr]
    else:
        valid_pct = float(options.valid_pct_or_chr)
        valid_n = int(valid_pct*len(seqs_segments))
        nontest_indexes = set(range(len(seqs_segments))) - set(test_indexes)
        valid_indexes = random.sample(nontest_indexes, valid_n)

    # remainder is training
    train_indexes = list(set(range(len(seqs_segments))) - set(valid_indexes) - set(test_indexes))

    # training may require shuffling
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

        # skip minor overlaps to the first
        first_start = seg_start + seg_unmap_start_i*pool_width
        first_end = first_start + pool_width
        first_overlap = first_end - unmap_start
        if first_overlap < 0.25*pool_width:
            seg_unmap_start_i += 1

        # skip minor overlaps to the last
        last_start = seg_start + (seg_unmap_end_i-1)*pool_width
        last_overlap = unmap_end - last_start
        if last_overlap < 0.25*pool_width:
            seg_unmap_end_i -= 1

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
def segments_1hot(fasta_file, segments, seq_length, stride):
    ''' Read and 1-hot code sequences in their segment batches.

    Args
     fasta_file: FASTA genome
     segments: list of (chrom,start,end) genomic segments to read
     seq_length: sequence length to break them into
     stride: distance to advance each sequence

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
if __name__ == '__main__':
    main()
