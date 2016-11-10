#!/usr/bin/env python
from optparse import OptionParser
import gc
import math
import sys
import time

import numpy as np
import pyBigWig
import pysam
from scipy.sparse import dok_matrix

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
    parser.add_option('-u', dest='assume_unique', default=False, action='store_true', help='Assume alignments are unique reads and ignore query_names [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 2:
        parser.error('Must provide input BAM file and output BigWig filename')
    else:
        bam_file = args[0]
        bigwig_file = args[1]

    ################################################################
    # map read query_name's to indexes

    t0 = time.time()

    # map reads to indexes
    read_index, num_reads = index_bam_reads(bam_file, options.assume_unique)

    print('Map read query_names to indexes: %ds' % (time.time()-t0))


    ################################################################
    # compute genome length

    bam_in = pysam.Samfile(bam_file, 'rb')

    chromosomes = bam_in.references
    chrom_lengths = bam_in.lengths
    genome_length = np.sum(chrom_lengths)


    ################################################################
    # initialize read coverage weights

    t0 = time.time()

    read_weights = dok_matrix((num_reads, genome_length), dtype='float16')

    # initialize for assume_unique
    ri = 0

    for align in bam_in:
        read_id = (align.query_name, align.is_read1)
        if not options.assume_unique:
            ri = read_index[read_id]

        # determine alignment position
        align_pos = align.reference_start
        if align.is_reverse:
            align_pos = align.reference_end

        # determine genome index
        gi = genome_index(chrom_lengths, align.reference_id, align_pos)

        # initialize weight
        read_weights[ri,gi] = 1/read_counts[ri]

        # update for assume_unique
        ri += 1

    # clean up big read_index
    del read_index
    gc.collect()

    print('Initialize read coverage weights: %ds' % (time.time()-t0))

    # convert to CSR format
    t0 = time.time()
    read_weights = read_weights.tocsr()
    print('Convert to CSR: %ds' % (time.time()-t0))


    ################################################################
    # run EM to distribute multi-mapping read weights

    if options.multi_window is not None:
        pass


    ################################################################
    # compute k-mer cut bias normalization

    if options.cut_bias_kmer is not None:
        kmer_norms = compute_cut_norm(options.cut_bias_kmer, read_weights, chromosomes, chrom_lengths, options.fasta_file)


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

        # compute chromosome coverage
        t0 = time.time()
        chrom_weights = read_weights[:,gi:gi+cl]
        print('chrom_weights memory: %d, time %ds' % (sys.getsizeof(chrom_weights), time.time()-t0))

        t0 = time.time()
        chrom_coverage_matrix = chrom_weights.sum(axis=0, dtype='float32')
        print('chrom_coverage_matrix memory: %d, time %ds' % (sys.getsizeof(chrom_coverage_matrix), time.time()-t0))

        t0 = time.time()
        chrom_coverage_array = np.array(chrom_coverage_matrix).squeeze()
        print('chrom_coverage_array memory: %d, time %ds' % (sys.getsizeof(chrom_coverage_array), time.time()-t0))

        t0 = time.time()
        chrom_coverage = chrom_coverage_array.tolist()
        print('chrom_coverage memory: %d, time %ds' % (sys.getsizeof(chrom_coverage), time.time()-t0))

        # normalize for cut bias
        if options.cut_bias_kmer is not None:
            # chrom_coverage =
            pass

        # add to bigwig
        t0 = time.time()
        bigwig_out.addEntries(chromosomes[ci], 0, values=chrom_coverage, span=1, step=1)
        print('pybigwig add: %ds' % (time.time()-t0))

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


def index_bam_reads(bam_file, assume_unique):
    ''' Index the reads aligned in a BAM file, and determine whether
         the query_name's and NH tags are trustworthy.

    Args
     bam_file (str):

    Output
     read_index ({str: int}): Dict mapping query_name to an int index.
     num_reads (int): Number of aligned reads
    '''

    read_index = {}
    read_counts = []

    ri = 0
    for align in pysam.Samfile(bam_file, 'rb'):
        if not align.is_unmapped:
            read_id = (align.query_name, align.is_read1)

            if assume_unique:
                num_reads += 1

            else:
                # if read name is new
                if align.query_name not in read_index:
                    # map it to an index
                    read_index[read_id] = ri
                    ri += 1

                # determine the number of multimaps
                nh_tag = 1
                if align.has_tag('NH'):
                    nh_tag = 1 / align.get_tag('NH')

                # count the number of multimap-adjusted read alignments
                ari = read_index[read_id]
                if len(read_counts) <= ari:
                    read_counts.append(0)
                read_counts[ari] += nh_tag

    # check read counts
    for read_id in read_index:
        if read_counts[read_index[read_id]] > 1:
            print('Multi-weighted coverage for %s > 1' % read_id, file=sys.stderr)
            exit(1)

    # save number of reads
    if not assume_unique:
        num_reads = len(read_index)

    return read_index, num_reads


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
