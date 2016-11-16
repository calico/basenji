#!/usr/bin/env python
from optparse import OptionParser
import gc
import math
import sys
import time

import numpy as np
import pyBigWig
import pysam
from scipy.sparse import csr_matrix

import size

'''
bam_cov.py

Compute a coverage track from a BAM file.
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <bam_file> <bigwig_file>'
    parser = OptionParser(usage)
    parser.add_option('-c', dest='cut_bias_kmer', default=None, action='store_true', help='Normalize coverage for a cutting bias model for k-mers [Default: %default]')
    parser.add_option('-m', dest='multi_window', default=None, type='int', help='Window size with which to model coverage in order to distribute multi-mapping read weight [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 2:
        parser.error('Must provide input BAM file and output BigWig filename')
    else:
        bam_file = args[0]
        bigwig_file = args[1]

    ################################################################
    # compute genome length

    bam_in = pysam.Samfile(bam_file, 'rb')

    chromosomes = bam_in.references
    chrom_lengths = bam_in.lengths
    genome_length = np.sum(chrom_lengths)


    ################################################################
    # construct multi-mapper alignment matrix

    # intialize genome uniquc coverage
    genome_unique_coverage = np.zeros(genome_length, dtype='uint16')

    # intialize sparse matrix in COO format
    # multi_read_pos = []
    multi_reads = []
    multi_positions = []
    multi_weight = []

    # initialize dict mapping read_id's to indexes
    multi_read_index = {}

    ri = 0
    for align in pysam.Samfile(bam_file, 'rb'):
        if not align.is_unmapped:
            read_id = (align.query_name, align.is_read1)

            # determine alignment position
            chrom_pos = align.reference_start
            if align.is_reverse:
                chrom_pos = align.reference_end

            # determine genome index
            gi = genome_index(chrom_lengths, align.reference_id, chrom_pos)

            # determine multi-mapping state
            nh_tag = 1
            if align.has_tag('NH'):
                nh_tag = align.get_tag('NH')

            # if unique
            if nh_tag == 1:
                genome_unique_coverage[gi] += 1

            # if multi
            else:
                # if read name is new
                if align.query_name not in multi_read_index:
                    # map it to an index
                    multi_read_index[read_id] = ri
                    ri += 1

                # store alignment matrix
                ari = multi_read_index[read_id]
                # multi_read_pos.append((ari,gi))
                multi_reads.append(ari)
                multi_positions.append(gi)
                multi_weight.append(np.float16(1/nh_tag))

    # store number of reads
    num_reads = len(multi_read_index)

    # convert sparse matrix
    # multi_weight_matrix = csr_matrix((multi_read_index,multi_weight), shape=(num_reads,genome_length))
    multi_weight_matrix = csr_matrix((multi_weight,(multi_reads,multi_positions)), shape=(num_reads,genome_length))

    # validate that weights sum to 1
    multi_sum = multi_weight_matrix.sum(axis=1)
    for read_id, ri in multi_read_index.items():
        if np.isclose(multi_sum[ri], 1):
            print('Multi-weighted coverage for %s != 1' % read_id, file=sys.stderr)
            exit(1)

    # clean up
    del multi_reads
    del multi_positions
    del multi_weight
    del multi_read_index
    del multi_sum
    gc.collect()


    ################################################################
    # run EM to distribute multi-mapping read weights

    if options.multi_window is not None:
        pass


    ################################################################
    # compute k-mer cut bias normalization

    if options.cut_bias_kmer is not None:
        kmer_norms = compute_cut_norm(options.cut_bias_kmer, multi_weight_matrix, chromosomes, chrom_lengths, options.fasta_file)


    ################################################################
    # compute genomic coverage / normalize for cut bias / output

    t0 = time.time()

    bigwig_out = pyBigWig.open(bigwig_file, 'w')

    # add header
    bigwig_out.addHeader(list(zip(chromosomes, chrom_lengths)))

    gi = 0
    for ci in range(len(chromosomes)):
        print(' Outputting %s' % chromosomes[ci])
        cl = chrom_lengths[ci]

        # compute multi-mapper chromosome coverage
        chrom_weight_matrix = multi_weight_matrix[:,gi:gi+cl]
        chrom_multi_coverage = np.array(chrom_weight_matrix.sum(axis=0, dtype='float32')).squeeze()

        # sum with unique coverage
        chrom_coverage_array = genome_unique_coverage[gi:gi+cl] + chrom_multi_coverage
        chrom_coverage = chrom_coverage_array.tolist()

        # normalize for cut bias
        if options.cut_bias_kmer is not None:
            # chrom_coverage =
            pass

        # add to bigwig
        t0 = time.time()
        bigwig_out.addEntries(chromosomes[ci], 0, values=chrom_coverage, span=1, step=1)
        print('pybigwig add: %ds' % (time.time()-t0))
        sys.stdout.flush()

        # update genomic index
        gi += cl

        gc.collect()

    bigwig_out.close()

    print('Output BigWig: %ds' % (time.time()-t0))


def compute_cut_norms(cut_bias_kmer, read_weights, chromosomes, chrom_lengths, fasta_file):
    ''' Compute cut bias normalizations.

    Args:
     cut_bias_kmer
     read_weights
     chromosomes
     chrom_lengths
     fasta_file

     Returns:
      kmer_norms
    '''

    kmer_left = cut_bias_kmer//2
    kmer_right = cut_bias_kmer - kmer_left - 1

    # initialize kmer cut counts
    kmer_cuts = initialize_kmers(cut_bias_kmer, 1)

    # open fasta
    fasta_open = pysam.Fastafile(fasta_file)

    # traverse genome and count
    gi = 0
    for ci in range(len(chromosomes)):
        cl = chrom_lengths[ci]

        # compute chromosome coverage
        chrom_coverage = np.array(read_weights[:,gi:gi+cl].sum(axis=0, dtype='float32')).squeeze()

        # read in the chromosome sequence
        seq = fasta_open.fetch(chromosomes[ci], 0, cl)

        for li in range(kmer_left, chrom_lengths[ci] - kmer_right):
            if chrom_coverage[li] > 0:
                # extract k-mer
                kmer_start = li - kmer_left
                kmer_end = kmer_start + options.cut_bias_kmer
                kmer = seq[kmer_start:kmer_end]

                # consider flipping to reverse complement
                rc_kmer = rc(kmer)
                if kmer > rc(kmer):
                    kmer = rc_kmer

                # count
                kmer_cuts[kmer] += genome_coverage[gi]

            gi += 1

    # compute k-mer normalizations
    kmer_sum = sum(kmer_cuts.values())
    kmer_norms = {}
    for kmer in kmer_cuts:
        # kmer_norms[kmer] =
        pass

    return kmer_norms


def genome_index(chrom_lengths, align_ci, align_pos):
    ''' Read and 1-hot code sequences in their segment batches.

    Args
     chrom_lengths ([int]): List of chromosome lengths.
     align_ci (int): Alignment chromosome index.
     align_pos (int): Alignment start position.

    Returns:
     gi (int): Genomic index
    '''

    gi = 0
    for ci in range(len(chrom_lengths)):
        if align_ci == ci:
            gi += align_pos
            break
        else:
            gi += chrom_lengths[ci]
    return gi


def initialize_kmers(k, pseudocount=1.):
    ''' Initialize a dict mapping kmers to cut counts.

    Args
     k (int): k-mer size
     pseudocount (float): Initial pseudocount

    '''
    kmer_cuts = {}
    for i in range(int(math.pow(4,k))):
        kmer = int2kmer(k,i)
        kmer_cuts[kmer] = pseudocount
    return kmer_cuts

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
