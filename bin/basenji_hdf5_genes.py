#!/usr/bin/env python
# Copyright 2017 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from optparse import OptionParser
from collections import OrderedDict
import multiprocessing
import os
import sys
import time

import h5py
import numpy as np
import pyBigWig
import pysam

import basenji

'''
basenji_hdf5_genes.py

Tile a set of genes and save the result in HDF5 for Basenji processing.

Notes:
 -At the moment, I'm excluding target measurements, but that could be included
  if I want to measure accuracy on specific genes.
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <fasta_file> <gtf_file> <hdf5_file>'
    parser = OptionParser(usage)
    parser.add_option('-g', dest='genome_file', default='%s/assembly/human.hg19.genome'%os.environ['HG19'], help='Chromosome lengths file [Default: %default]')
    parser.add_option('-l', dest='seq_length', default=1024, type='int', help='Sequence length [Default: %default]')
    parser.add_option('-c', dest='center_t', default=1/3, type='float', help='Center proportion in which TSSs are required to be [Default: %default]')
    parser.add_option('-p', dest='processes', default=1, type='int', help='Number parallel processes to load data [Default: %default]')
    parser.add_option('-t', dest='target_wigs_file', default=None, help='Store target values, extracted from this list of WIG files')
    parser.add_option('-w', dest='pool_width', type='int', default=1, help='Average pooling width [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 3:
        parser.error('Must provide genes as GTF, genome FASTA, and output HDF5')
    else:
        fasta_file = args[0]
        gtf_file = args[1]
        hdf5_file = args[2]


    ################################################################
    # organize transcript TSS's by chromosome

    # read transcripts
    transcripts = basenji.gff.read_genes(gtf_file, key_id='transcript_id')

    # read transcript --> gene mapping
    transcript_genes = basenji.gff.t2g(gtf_file)

    # open fasta
    fasta = pysam.Fastafile(fasta_file)

    # hash transcript TSS's by chromosome
    chrom_tss = {}
    for tx_id in transcripts:
        if transcripts[tx_id].chrom in fasta.references:
            chrom_tss.setdefault(transcripts[tx_id].chrom,[]).append((transcripts[tx_id].tss(),tx_id))

    # sort TSS's by chromosome
    for chrom in chrom_tss:
        chrom_tss[chrom].sort()


    ################################################################
    # determine segments / map transcripts

    chrom_sizes = OrderedDict()
    for line in open(options.genome_file):
        a = line.split()
        chrom_sizes[a[0]] = int(a[1])

    merge_distance = options.center_t * options.seq_length

    seq_coords = []
    transcript_map = OrderedDict()

    # ordering by options.genome_file allows for easier
    #  bigwig output in downstream scripts.

    for chrom in chrom_sizes:
        if chrom in chrom_tss:
            ctss = chrom_tss[chrom]

            left_i = 0
            while left_i < len(ctss):
                # left TSS
                left_tss = ctss[left_i][0]

                # right TSS
                right_i = left_i
                while right_i+1 < len(ctss) and ctss[right_i+1][0] - left_tss < merge_distance:
                    right_i += 1
                right_tss = ctss[right_i][0]

                # determine segment midpoint
                seg_mid = (left_tss + right_tss) // 2

                # extend
                seg_start = seg_mid - options.seq_length//2
                seg_end = seg_start + options.seq_length

                # rescue
                if seg_start < 0 or seg_end >= chrom_sizes[chrom]:
                    if chrom_sizes[chrom] == options.seq_length:
                        seg_start = 0
                        seg_end = options.seq_length
                    elif chrom_sizes[chrom] > options.seq_length:
                        # also rescuable but not important right now
                        pass

                # save segment
                if seg_start >= 0 and seg_end <= chrom_sizes[chrom]:
                    seq_coords.append((chrom,seg_start,seg_end))

                    # annotate TSS to indexes
                    seg_index = len(seq_coords)-1
                    for i in range(left_i,right_i+1):
                        tx_tss, tx_id = ctss[i]
                        tx_pos = (tx_tss - seg_start) // options.pool_width
                        transcript_map[tx_id] = (seg_index,tx_pos)

                # update
                left_i = right_i + 1


    ################################################################
    # extract target values

    if options.target_wigs_file:
        t0 = time.time()

        # get wig files and labels
        target_wigs = OrderedDict()
        for line in open(options.target_wigs_file):
            a = line.split()
            target_wigs[a[0]] = a[1]

        # initialize multiprocessing pool
        pool = multiprocessing.Pool(options.processes)

        # bigwig_read parameters
        bwt_params = [(wig_file, transcript_map, seq_coords, options.pool_width) for wig_file in target_wigs.values()]

        # pull the target values in parallel
        transcript_targets = pool.starmap(bigwig_transcripts, bwt_params)

        # convert to array
        transcript_targets = np.transpose(np.array(transcript_targets))


    ################################################################
    # extract sequences

    seqs_1hot = []

    for chrom, start, end in seq_coords:
        seq = fasta.fetch(chrom, start, end)
        seqs_1hot.append(basenji.dna_io.dna_1hot(seq))

    seqs_1hot = np.array(seqs_1hot)

    fasta.close()


    ################################################################
    # save to HDF5

    # write to HDF5
    hdf5_out = h5py.File(hdf5_file, 'w')

    # store pooling
    hdf5_out.create_dataset('pool_width', data=options.pool_width, dtype='int')

    # store transcript map
    transcripts = list(transcript_map.keys())
    transcript_index = np.array([transcript_map[tx_id][0] for tx_id in transcripts])
    transcript_pos = np.array([transcript_map[tx_id][1] for tx_id in transcripts])

    hdf5_out.create_dataset('transcripts', data=np.array(transcripts, dtype='S'))
    hdf5_out.create_dataset('transcript_index', data=transcript_index)
    hdf5_out.create_dataset('transcript_pos', data=transcript_pos)

    # store genes
    genes = [transcript_genes[tx_id] for tx_id in transcripts]
    hdf5_out.create_dataset('genes', data=np.array(genes, dtype='S'))

    # store sequences
    hdf5_out.create_dataset('seqs_1hot', data=seqs_1hot, dtype='bool')

    # store targets
    if options.target_wigs_file:
        # labels
        target_labels = np.array([tl for tl in target_wigs.keys()], dtype='S')
        hdf5_out.create_dataset('target_labels', data=target_labels)

        # values
        hdf5_out.create_dataset('transcript_targets', data=transcript_targets, dtype='float16')

    # store sequence coordinates
    seq_chrom = np.array([sc[0] for sc in seq_coords], dtype='S')
    seq_start = np.array([sc[1] for sc in seq_coords])
    seq_end = np.array([sc[2] for sc in seq_coords])

    hdf5_out.create_dataset('seq_chrom', data=seq_chrom)
    hdf5_out.create_dataset('seq_start', data=seq_start)
    hdf5_out.create_dataset('seq_end', data=seq_end)

    hdf5_out.close()


################################################################################
def bigwig_transcripts(wig_file, transcript_map, seq_coords, pool_width=1):
    ''' Read gene target values from a bigwig

    Args:
      wig_file: Bigwig filename
      transcript_map: OrderedDict mapping transcript_id to (seq index,seq pos) tuples
      seq_coords: list of (chrom,start,end) sequence coordinates
      pool_width: average pool adjacent nucleotides of this width

    Returns:
      transcript_targets:
    '''

    # initialize target values
    transcript_targets = np.zeros(len(transcript_map), dtype='float16')

    # open wig
    wig_in = pyBigWig.open(wig_file)

    # so we can warn about missing chromosomes just once
    warned_chroms = set()

    # for each transcript
    tx_i = 0
    for transcript in transcript_map:
        # determine sequence and position
        seq_i, seq_pos = transcript_map[transcript]

        # extract sequence coordinates
        seq_chrom, seq_start, seq_end = seq_coords[seq_i]

        # determine gene genomic coordinates
        tx_start = seq_start + seq_pos*pool_width
        tx_end = tx_start + pool_width

        # pull sum (formerly mean value)
        try:
            # transcript_targets[tx_i] = wig_in.stats(seq_chrom, tx_start, tx_end)[0]
            transcript_targets[tx_i] = np.array(wig_in.values(seq_chrom, tx_start, tx_end), dtype='float32').sum()

        except RuntimeError:
            if seq_chrom not in warned_chroms:
                print("WARNING: %s doesn't see %s (%s:%d-%d). Setting to all zeros. No additional warnings will be offered for %s" % (wig_file,transcript,seq_chrom,seq_start,seq_end,seq_chrom), file=sys.stderr)
                warned_chroms.add(seq_chrom)

        # check NaN
        if np.isnan(transcript_targets[tx_i]):
            print('WARNING: %s (%s:%d-%d) pulled NaN from %s. Setting to zero.' % (transcript, seq_chrom, seq_start, seq_end, wig_file), file=sys.stderr)
            transcript_targets[tx_i] = 0

        tx_i += 1

    # close wig file
    wig_in.close()

    return transcript_targets


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
