#!/usr/bin/env python
from optparse import OptionParser

import h5py
import numpy as np
import pysam

import basenji

'''
basenji_genes.py

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
    parser.add_option('-l', dest='seq_length', default=1024, type='int', help='Sequence length [Default: %default]')
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

    merge_distance = .2*options.seq_length

    seq_segments = []
    transcript_map = {}

    for chrom in chrom_tss:
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

            # save segment
            seq_segments.append((chrom,seg_start,seg_end))

            # annotate TSS to indexes
            seg_index = len(seq_segments)-1
            for i in range(left_i,right_i+1):
                tx_tss, tx_id = ctss[i]
                tx_pos = (tx_tss - seg_start) // options.pool_width
                transcript_map[tx_id] = (chrom,seg_index,tx_pos)

            # update
            left_i = right_i + 1


    ################################################################
    # extract sequences

    seqs_1hot = []

    for chrom, seg_start, seg_end in seq_segments:
        seg_seq = fasta.fetch(chrom, seg_start, seg_end)
        seqs_1hot.append(basenji.dna_io.dna_1hot(seg_seq))

    seqs_1hot = np.array(seqs_1hot)

    fasta.close()


    ################################################################
    # save to HDF5

    # write to HDF5
    hdf5_out = h5py.File(hdf5_file, 'w')

    # store pooling
    hdf5_out.create_dataset('pool_width', data=options.pool_width, dtype='int')

    # store transcript map
    transcripts = sorted(list(transcript_map.keys()))
    transcript_chrom = np.array([transcript_map[tx_id][0] for tx_id in transcripts], dtype='S')
    transcript_index = np.array([transcript_map[tx_id][1] for tx_id in transcripts])
    transcript_pos = np.array([transcript_map[tx_id][2] for tx_id in transcripts])

    hdf5_out.create_dataset('transcript_chrom', data=transcript_chrom)
    hdf5_out.create_dataset('transcript_index', data=transcript_index)
    hdf5_out.create_dataset('transcript_pos', data=transcript_chrom)

    # store sequences
    hdf5_out.create_dataset('test_in', data=seqs_1hot, dtype='bool')

    # store segments
    seg_chrom = np.array([ss[0] for ss in seq_segments], dtype='S')
    seg_start = np.array([ss[1] for ss in seq_segments])
    seg_end = np.array([ss[2] for ss in seq_segments])

    hdf5_out.create_dataset('seg_chrom', data=seg_chrom)
    hdf5_out.create_dataset('seg_start', data=seg_start)
    hdf5_out.create_dataset('seg_end', data=seg_end)

    hdf5_out.close()


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
