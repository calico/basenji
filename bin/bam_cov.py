#!/usr/bin/env python
from optparse import OptionParser
import gc
import math
import sys
import time

import numpy as np
import pyBigWig
import pysam
from scipy.ndimage.filters import gaussian_filter1d
from scipy.sparse import csc_matrix, csr_matrix

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
    parser.add_option('-d', dest='duplicate_max', default=2, type='int', help='Maximum coverage at a single position, which must be addressed due to PCR duplicates [Default: %default]')
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
                if read_id not in multi_read_index:
                    # map it to an index
                    multi_read_index[read_id] = ri
                    ri += 1

                # store alignment matrix
                ari = multi_read_index[read_id]
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
        if not np.isclose(multi_sum[ri], 1, rtol=1e-3):
            print('Multi-weighted coverage for (%s,%s) %f != 1' % (read_id[0],read_id[1],multi_sum[ri]), file=sys.stderr)
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
        distribute_multi(multi_weight_matrix, genome_unique_coverage, chrom_lengths, options.multi_window, 10)


    ################################################################
    # compute k-mer cut bias normalization

    if options.cut_bias_kmer is not None:
        kmer_norms = compute_cut_norm(options.cut_bias_kmer, multi_weight_matrix, chromosomes, chrom_lengths, options.fasta_file)


    ################################################################
    # compute genomic coverage / normalize for cut bias / output

    print('Outputting coverage')

    bigwig_out = pyBigWig.open(bigwig_file, 'w')

    # add header
    bigwig_out.addHeader(list(zip(chromosomes, chrom_lengths)))

    gi = 0
    for ci in range(len(chromosomes)):
        t0 = time.time()
        print('  %s' % chromosomes[ci], end='', flush=True)
        cl = chrom_lengths[ci]

        # compute multi-mapper chromosome coverage
        chrom_weight_matrix = multi_weight_matrix[:,gi:gi+cl]
        chrom_multi_coverage = np.array(chrom_weight_matrix.sum(axis=0, dtype='float32')).squeeze()

        # sum with unique coverage
        chrom_coverage_array = genome_unique_coverage[gi:gi+cl] + chrom_multi_coverage
        chrom_coverage = chrom_coverage_array.tolist()

        # add to bigwig
        t0 = time.time()
        bigwig_out.addEntries(chromosomes[ci], 0, values=chrom_coverage, span=1, step=1)

        # update genomic index
        gi += cl

        # clean up temp storage
        gc.collect()

        # update user
        print(', %ds' % (time.time()-t0), flush=True)

    t0 = time.time()
    bigwig_out.close()
    print(' Close BigWig: %ds' % (time.time()-t0))


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

def row_nzcols(m, ri):
    ''' Return row ri's nonzero columns. '''
    return m.indices[m.indptr[ri]:m.indptr[ri+1]]


def distribute_multi(multi_weight_matrix, genome_unique_coverage, chrom_lengths, multi_window, max_iterations=10, converge_t=.01):
    ''' Distribute multi-mapping read weight proportional to coverage in a local window.

    In
     multi_weight_matrix: R (reads) x G (genomic position) array of multi-mapping read weight.
     genome_unique_coverage: G (genomic position) array of unique coverage counts.
     multi_window: Window size used to estimate local coverage.
     chrom_lengths ([int]): List of chromosome lengths.
     max_iterations: Maximum iterations through the reads.
     converge_t: Per read weight difference below which we consider convergence.

    '''
    print('Distributing %d multi-mapping reads.' % multi_weight_matrix.shape[0])

    # initialize genome coverage
    genome_coverage = np.zeros(len(genome_unique_coverage), dtype='float16')

    for it in range(max_iterations):
        print(' Iteration %d' % (it+1), end='', flush=True)
        t_it = time.time()

        # track convergence
        iteration_change = 0

        # update genome coverage estimates
        estimate_coverage(genome_coverage, genome_unique_coverage, multi_weight_matrix, chrom_lengths)

        # re-allocate multi-reads proportionally to coverage estimates
        t_r = time.time()
        for ri in range(multi_weight_matrix.shape[0]):
            # update user
            if ri % 100000 == 100000-1:
                print('\n  processed %d reads in %ds' % (ri+1, time.time()-t_r), end='', flush=True)
                t_r = time.time()

            # get read's aligning positions
            multi_read_positions = row_nzcols(multi_weight_matrix, ri)
            multi_positions_len = len(multi_read_positions)

            # get coverage estimates
            multi_positions_coverage = np.array([genome_coverage[pos] for pos in multi_read_positions])

            # normalize coverage as weights
            multi_positions_weight = multi_positions_coverage / multi_positions_coverage.sum()

            # re-proportion read weight
            for pi in range(multi_positions_len):
                # get genomic position
                pos = multi_read_positions[pi]

                # track convergence
                iteration_change += abs(multi_positions_weight[pi] - multi_weight_matrix[ri,pos])

                # set new weight
                multi_weight_matrix[ri,pos] = multi_positions_weight[pi]

        # assess coverage
        iteration_change /= multi_weight_matrix.shape[0]
        # print(', %.3f change per multi-read' % iteration_change, flush=True)
        print('\n Finished in %ds with %.3f change per multi-read' % (time.time()-t_it, iteration_change), flush=True)
        if iteration_change < converge_t:
            break


def distribute_multi_succint(multi_weight_matrix, genome_unique_coverage, multi_window, chrom_lengths, max_iterations=10, converge_t=.01):
    ''' Wondering if I can speed things up by vectorizing these operations, but still much to test.

    1. start by comparing the times to make my genome_coverage estimate here with multi_weight_matrix.sum(axis=0) versus looping through the arrays myself.
    2. then benchmark that dot product.
    3. then benchmark the column normalization.
    4. then determine a way to assess convergence. maybe sample 1k reads to act as my proxy?

    '''

    print('Distributing %d multi-mapping reads.' % multi_weight_matrix.shape[0])

    # initialize genome coverage
    genome_coverage = np.zeros(len(genome_unique_coverage), dtype='float16')

    for it in range(max_iterations):
        print(' Iteration %d' % (it+1), end='', flush=True)
        t_it = time.time()

        # track convergence
        iteration_change = 0

        # estimate genome coverage
        genome_coverage = genome_unique_coverage + multi_weight_matrix.sum(axis=0)

        # smooth estimate by chromosome
        gi = 0
        for clen in chrom_lengths:
            genome_coverage[gi:gi+clen] = gaussian_filter1d(genome_coverage[gi:gi+clen].astype('float32'), sigma=25, truncate=3)
            gi = gi+clen

        # re-distribute multi-mapping reads
        multi_weight_matrix = multi_weight_matrix.dot(genome_coverage)

        # normalize
        multi_weight_matrix /= multi_weight_matrix.mean(axis=1)

        # assess coverage (not sure how to do that. maybe an approximation of the big sparse matrix?)
        iteration_change /= multi_weight_matrix.shape[0]
        # print(', %.3f change per multi-read' % iteration_change, flush=True)
        print(' Finished in %ds with %.3f change per multi-read' % (time.time()-t_it, iteration_change), flush=True)
        if iteration_change / multi_weight_matrix.shape[0] < converge_t:
            break


def estimate_coverage(genome_coverage, genome_unique_coverage, multi_weight_matrix, chrom_lengths):
    ''' Estimate smoothed genomic coverage.

    In
     chrom_lengths ([int]): List of chromosome lengths.

    '''

    # start with unique coverage
    np.copyto(genome_coverage, genome_unique_coverage)

    # add in multi-map coverage
    for ri in range(multi_weight_matrix.shape[0]):
        ri_ptr, ri_ptr1 = multi_weight_matrix.indptr[ri:ri+2]
        ci = 0
        for pos in multi_weight_matrix.indices[ri_ptr:ri_ptr1]:
            genome_coverage[pos] += multi_weight_matrix.data[ri_ptr+ci]
            ci += 1

    # smooth by chromosome
    gi = 0
    for clen in chrom_lengths:
        genome_coverage[gi:gi+clen] = gaussian_filter1d(genome_coverage[gi:gi+clen].astype('float32'), sigma=25, truncate=3)
        gi = gi+clen


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
