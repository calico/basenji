#!/usr/bin/env python
from optparse import OptionParser
from collections import OrderedDict
import gc
import h5py
import itertools
import math
import os
import random
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyBigWig
import pysam
from scipy.ndimage.filters import gaussian_filter1d
from scipy.stats import norm
from scipy.sparse import csc_matrix, csr_matrix
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import basenji

'''
bam_cov.py

Transform BAM alignments to a normalized BigWig coverage track.

Notes:
 -I'm making a tradeoff here on proper PCR duplicate maxing whereby I'm combining the
  forward and reverse strands. To keep them separate, I'd need to double the memory.
  Alternatively, I could remove duplicates at an earlier stage, but I'd have to
  better understand how multi-mapping reads are handled; I'm worried it will throw
  away multi-mappers so the NH tag no longer reflects the number of alignments.

 -It's probably worth trying to perform local filtering after every significant
  read weight change, instead of a single big filter after every full parse through
  the reads.
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <bam_file> <bigwig_file>'
    parser = OptionParser(usage)
    parser.add_option('-c', dest='cut_bias_kmer', default=None, action='store_true', help='Normalize coverage for a cutting bias model for k-mers [Default: %default]')
    parser.add_option('-d', dest='duplicate_max', default=2, type='int', help='Maximum coverage at a single position, which must be addressed due to PCR duplicates [Default: %default]')
    parser.add_option('-f', dest='fasta_file', default='%s/assembly/hg19.fa'%os.environ['HG19'], help='FASTA to obtain sequence to control for GC% [Default: %default]')
    parser.add_option('-g', dest='gc', default=False, action='store_true', help='Control for local GC% [Default: %default]')
    parser.add_option('-m', dest='multi_em', default=0, type='int', help='Iterations of EM to distribute multi-mapping reads [Default: %default]')
    parser.add_option('-o', dest='out_dir', default='bam_cov', help='Output directory [Default: %default]')
    parser.add_option('-r', dest='strand_corr_shift', default=False, action='store_true', help='Learn the fragment shift by fitting a strand cross correlation model [Default: %default]')
    parser.add_option('-s', dest='smooth_sd', default=32, type='float', help='Gaussian standard deviation to smooth coverage estimates with [Default: %default]')
    parser.add_option('-t', dest='shift', default=0, type='int', help='Fragment shift [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 2:
        parser.error('Must provide input BAM file and output BigWig filename')
    else:
        bam_file = args[0]
        bigwig_file = args[1]

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    ################################################################
    # initialize genome coverage

    bam_in = pysam.Samfile(bam_file, 'rb')
    chrom_lengths = OrderedDict(zip(bam_in.references, bam_in.lengths))
    bam_in.close()

    # determine single or paired
    sp = single_or_pair(bam_file)

    # tell duplicate_max to expect double coverage for pair
    if options.duplicate_max is not None and sp == 'pair':
        options.duplicate_max *= 2

    # initialize
    genome_coverage = GenomeCoverage(chrom_lengths, smooth_sd=options.smooth_sd, duplicate_max=options.duplicate_max, shift=options.shift, fasta_file=options.fasta_file)

    # estimate fragment shift
    if options.strand_corr_shift:
        if sp == 'single':
            genome_coverage.learn_shift_single(bam_file, out_dir=options.out_dir)
        else:
            genome_coverage.learn_shift_pair(bam_file)

    # read alignments
    genome_coverage.read_bam(bam_file)


    ################################################################
    # run EM to distribute multi-mapping read weights

    if options.multi_em > 0:
        genome_coverage.distribute_multi(options.multi_em)


    ################################################################
    # compute k-mer cut bias normalization

    if options.cut_bias_kmer is not None:
        kmer_norms = compute_cut_norm(options.cut_bias_kmer, multi_weight_matrix, chromosomes, chrom_lengths, options.fasta_file)

    ################################################################
    # normalize for GC content

    if options.gc:
        genome_coverage.learn_gc(fragment_sd=options.smooth_sd, out_dir=options.out_dir)


    ################################################################
    # compute genomic coverage / normalize for cut bias / output

    genome_coverage.write_bigwig(bigwig_file, sp)


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

    lengths_list = list(chrom_lengths.values())

    kmer_left = cut_bias_kmer//2
    kmer_right = cut_bias_kmer - kmer_left - 1

    # initialize kmer cut counts
    kmer_cuts = initialize_kmers(cut_bias_kmer, 1)

    # open fasta
    fasta_open = pysam.Fastafile(fasta_file)

    # traverse genome and count
    gi = 0
    for ci in range(len(chromosomes)):
        cl = lengths_list[ci]

        # compute chromosome coverage
        chrom_coverage = np.array(read_weights[:,gi:gi+cl].sum(axis=0, dtype='float32')).squeeze()

        # read in the chromosome sequence
        seq = fasta_open.fetch(chromosomes[ci], 0, cl)

        for li in range(kmer_left, cl - kmer_right):
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

def row_nzcols_geti(m, ri):
    ''' Return row ri's nonzero columns. '''
    return m.indices[m.indptr[ri]:m.indptr[ri+1]]

def row_nzcols_get(m, ri):
    ''' Return row ri's nonzero columns. '''
    return m.data[m.indptr[ri]:m.indptr[ri+1]]

def row_nzcols_set(m, ri, v):
    ''' Set row ri's nonzero columsn to v. '''
    m.data[m.indptr[ri]:m.indptr[ri+1]] = v


def single_or_pair(bam_file):
    ''' Check the first read to guess if the BAM has single or paired end reads. '''
    bam_in = pysam.Samfile(bam_file)
    align = bam_in.__next__()
    if align.is_paired:
        sp = 'pair'
    else:
        sp = 'single'
    bam_in.close()

    return single


def distribute_multi_succint(multi_weight_matrix, genome_unique_coverage, multi_window, chrom_lengths, max_iterations=10, converge_t=.01):
    ''' Wondering if I can speed things up by vectorizing these operations, but still much to test.

    1. start by comparing the times to make my genome_coverage estimate here with multi_weight_matrix.sum(axis=0) versus looping through the arrays myself.
    2. then benchmark that dot product.
    3. then benchmark the column normalization.
    4. then determine a way to assess convergence. maybe sample 1k reads to act as my proxy?

    '''

    print('Distributing %d multi-mapping reads.' % multi_weight_matrix.shape[0])

    lengths_list = list(chrom_lengths.values())

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
        for clen in lengths_list:
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


def regplot_gc(vals1, vals2, model, out_pdf):
    gold = sns.color_palette('husl',8)[1]

    plt.figure(figsize=(6,6))

    # plot data and seaborn model
    ax = sns.regplot(vals1, vals2, color='black', order=3, scatter_kws={'color':'black', 's':4, 'alpha':0.5}, line_kws={'color':gold})

    # plot my model predictions
    svals1 = np.sort(vals1)
    preds2 = model.predict(svals1[:,np.newaxis])
    ax.plot(svals1, preds2)

    # adjust axis
    ymin, ymax = basenji.plots.scatter_lims(vals2)
    ax.set_xlim(0.2, 0.8)
    ax.set_xlabel('GC%')
    ax.set_ylim(ymin,ymax)
    ax.set_ylabel('Coverage')

    ax.grid(True, linestyle=':')

    plt.savefig(out_pdf)
    plt.close()


def regplot_shift(vals1, vals2, preds2, out_pdf):
    gold = sns.color_palette('husl',8)[1]

    plt.figure(figsize=(6,6))

    # plot data and seaborn model
    ax = sns.regplot(vals1, vals2, color='black', order=3, scatter_kws={'color':'black', 's':4, 'alpha':0.5}, line_kws={'color':gold})

    # plot my model predictions
    ax.plot(vals1, preds2)

    # adjust axis
    ymin, ymax = basenji.plots.scatter_lims(vals2)
    ax.set_xlabel('Shift')
    ax.set_ylim(ymin,ymax)
    ax.set_ylabel('Covariance')

    ax.grid(True, linestyle=':')

    plt.savefig(out_pdf)
    plt.close()


class GenomeCoverage:
    '''
     chrom_lengths (OrderedDict): Mapping chromosome names to lengths.
     fasta (pysam Fastafile):

     unique_counts: G (genomic position) array of unique coverage counts.
     multi_weight_matrix: R (reads) x G (genomic position) array of multi-mapping read weight.

     smooth_sd (int): Gaussian filter standard deviation.
     duplicate_max (int): Maximum coverage per position due to PCR duplicate fear.
     shift (int): Alignment shift to maximize cross-strand coverage correlation
    '''

    def __init__(self, chrom_lengths, smooth_sd=32, duplicate_max=2, shift=0, fasta_file=None):
        self.chrom_lengths = chrom_lengths
        self.genome_length = sum(chrom_lengths.values())
        self.unique_counts = np.zeros(self.genome_length, dtype='uint16')
        self.active_blocks = None

        self.smooth_sd = smooth_sd
        self.duplicate_max = duplicate_max
        self.shift = shift

        self.fasta = None
        if fasta_file is not None:
            self.fasta = pysam.Fastafile(fasta_file)

        self.gc_model = None


    def distribute_multi(self, max_iterations=4, converge_t=.05):
        ''' Distribute multi-mapping read weight proportional to coverage in a local window.

        In
         max_iterations: Maximum iterations through the reads.
         converge_t: Per read weight difference below which we consider convergence.

        '''
        t0 = time.time()
        num_multi_reads = self.multi_weight_matrix.shape[0]
        print('Distributing %d multi-mapping reads.' % num_multi_reads, flush=True)

        # choose read indexes at which we'll re-estimate genome coverage
        #  (currently, estimate_coverage takes up more time unless the # reads
        #   is huge, but that may change if I can better optimize it.)
        iteration_estimates = max(1, int(np.round(num_multi_reads // 20000000)))
        restimate_indexes = np.linspace(0, num_multi_reads, iteration_estimates+1)[:-1]

        # initialize genome coverage
        genome_coverage = np.zeros(len(self.unique_counts), dtype='float16')

        for it in range(max_iterations):
            print(' Iteration %d' % (it+1), end='', flush=True)
            t_it = time.time()

            # track convergence
            iteration_change = 0

            # re-allocate multi-reads proportionally to coverage estimates
            print('  Re-allocating multi-reads.', end='', flush=True)
            t_r = time.time()
            for ri in range(num_multi_reads):
                # update user
                if ri % 1000000 == 1000000-1:
                    print('\n  processed %d reads in %ds' % (ri+1, time.time()-t_r), end='', flush=True)
                    t_r = time.time()

                # update genome coverage estimates
                if ri in restimate_indexes:
                    print('  Estimating genomic coverage.', end='', flush=True)
                    self.estimate_coverage(genome_coverage)
                    print(' Done.', flush=True)

                # get read's aligning positions
                multi_read_positions = row_nzcols_geti(self.multi_weight_matrix, ri)
                multi_positions_len = len(multi_read_positions)

                # get coverage estimates
                multi_positions_coverage = np.array([genome_coverage[pos] for pos in multi_read_positions])

                # normalize coverage as weights
                mpc_sum = multi_positions_coverage.sum(dtype='float64')
                if mpc_sum > 0:
                    multi_positions_weight = multi_positions_coverage / mpc_sum
                else:
                    print('Error: read %d coverage sum == %.4f' % (ri, mpc_sum))
                    print(multi_positions_coverage)
                    exit(1)

                # get previous weights
                multi_read_prev = row_nzcols_get(self.multi_weight_matrix, ri)

                # compute change
                read_pos_change = np.abs(multi_read_prev - multi_positions_weight).sum()

                if read_pos_change > .001:
                    # track convergence
                    iteration_change += read_pos_change

                    # set new weights
                    row_nzcols_set(self.multi_weight_matrix, ri, multi_positions_weight)

            print(' Done.', flush=True)

            # clean up temp storage
            gc.collect()

            # assess coverage
            iteration_change /= self.multi_weight_matrix.shape[0]
            print(' Iteration completed in %ds with %.3f change per multi-read' % (time.time()-t_it, iteration_change), flush=True)
            if iteration_change < converge_t:
                break


    def estimate_coverage(self, genome_coverage, pseudocount=.01):
        ''' Estimate smoothed genomic coverage.

        In
         genome_coverage: G (genomic position) array of estimated coverage counts.
         pseudocount (int): Coverage pseudocount.
        '''

        lengths_list = list(self.chrom_lengths.values())

        # start with unique coverage, and add pseudocount
        np.copyto(genome_coverage, self.unique_counts)
        genome_coverage += pseudocount

        # add in multi-map coverage
        for ri in range(self.multi_weight_matrix.shape[0]):
            ri_ptr, ri_ptr1 = self.multi_weight_matrix.indptr[ri:ri+2]
            ci = 0
            for pos in self.multi_weight_matrix.indices[ri_ptr:ri_ptr1]:
                genome_coverage[pos] += self.multi_weight_matrix.data[ri_ptr+ci]
                ci += 1

        # limit duplicates
        if self.duplicate_max is not None:
            np.copyto(genome_coverage, np.clip(genome_coverage, 0, self.duplicate_max))

        ''' if I switch to active blocks method
        if self.active_blocks is None:
            # determine centromere-like empty blocks
            print('Inferring active blocks.', flush=True)
            self.infer_active_blocks_groupby(genome_coverage)
        '''

        # smooth by chromosome
        if self.smooth_sd > 0:
            gi = 0
            for clen in lengths_list:
                gc.collect()
                genome_coverage[gi:gi+clen] = gaussian_filter1d(genome_coverage[gi:gi+clen].astype('float32'), sigma=self.smooth_sd, truncate=3)
                gi += clen

            ''' if I switch to active blocks method
            for gis, gie in self.active_blocks:
                genome_coverage[gis:gie] = gaussian_filter1d(genome_coverage[gis:gie].astype('float32'), sigma=self.smooth_sd, truncate=3)
            '''


    def gc_normalize(self, chrom, coverage):
        ''' Apply a model to normalize for GC content.

        In
         chrom (str): Chromosome
         coverage ([float]): Coverage array
         pseudocount (int): Coverage pseudocount.

        Out
         model (sklearn object): To control for GC%.
        '''

        # get sequence
        seq = self.fasta.fetch(chrom)
        assert(len(seq) == len(coverage))

        # compute GC boolean vector
        seq_gc = np.array([nt in 'CG' for nt in seq], dtype='float32')

        # gaussian filter1d
        seq_gc_gauss = gaussian_filter1d(seq_gc, sigma=self.fragment_sd, truncate=3)

        # compute norm quantity
        seq_gc_norm = self.gc_model.predict(seq_gc_gauss[:,np.newaxis])

        # apply it
        return coverage * np.exp(-seq_gc_norm+self.gc_base)


    def learn_gc(self, fragment_sd=64, pseudocount=.01, out_dir=None):
        ''' Learn a model to normalize for GC content.

        In
         fragment_sd (int): Gaussian filter standard deviation.
         pseudocount (int): Coverage pseudocount.

        Out
         model (sklearn object): To control for GC%.
        '''

        t0 = time.time()
        print('Fitting GC model.', flush=True, end='')

        # save
        self.fragment_sd = fragment_sd

        # helpers
        chroms_list = list(self.chrom_lengths.keys())
        fragment_sd3 = int(np.round(3*self.fragment_sd))

        # gaussian mask
        gauss_kernel = norm.pdf(np.arange(-fragment_sd3, fragment_sd3), loc=0, scale=self.fragment_sd)
        gauss_invsum = 1.0 / gauss_kernel.sum()

        #######################################################
        # construct training data

        # determine multi-mapping positions
        multi_positions = sorted(set(self.multi_weight_matrix.indices))

        # traverse the list and grab positions within
        train_suggested = []
        for mi in range(1,len(multi_positions)-1):
            mp1 = multi_positions[mi]
            mp2 = multi_positions[mi+1]
            train_suggested += list(np.arange(mp1, mp2, 1000))[1:]

        # sub-sample if necessary
        sample_num = 20000
        if len(train_suggested) > sample_num:
            train_suggested = random.sample(train_suggested, sample_num)

        nfilter = 0

        # compute position GC content
        train_pos = []
        train_gc = []
        train_cov = []
        for gi in train_suggested:
            # determine chromosome and position
            ci, pos = self.index_genome(gi)

            # get sequence
            seq_start = max(0, pos - fragment_sd3)
            seq_end = pos + fragment_sd3
            seq = self.fasta.fetch(chroms_list[ci], seq_start, seq_end)

            # filter for clean sequences
            if len(seq) == 2*fragment_sd3 and seq.find('N') == -1:

                # compute GC%
                seq_gc = np.array([nt in 'CG' for nt in seq], dtype='float32')
                gauss_gc = (seq_gc * gauss_kernel).sum() * gauss_invsum
                train_gc.append(gauss_gc)
                train_pos.append(gi)

                # compute coverage
                seq_cov = self.unique_counts[gi-fragment_sd3:gi+fragment_sd3] + pseudocount
                if self.duplicate_max is not None:
                    seq_cov = np.clip(seq_cov, 0, self.duplicate_max)
                gauss_cov = (seq_cov * gauss_kernel).sum() * gauss_invsum
                train_cov.append(np.log2(gauss_cov))

            elif len(seq) != 2*fragment_sd3:
                print('WARNING: GC sequence %s:%d-%d has length %d != %d' % (chroms_list[ci],seq_start,seq_end,len(seq), 2*fragment_sd3), file=sys.stderr)

            else:
                nfilter += 1

        print('WARNING: %d/%d GC sequences removed for having Ns' % (nfilter, len(train_suggested)), file=sys.stderr)

        # convert to arrays
        train_gc = np.array(train_gc)
        train_cov = np.array(train_cov)

        #######################################################
        # fit model

        # polynomial regression
        self.gc_model = make_pipeline(PolynomialFeatures(3), Ridge())
        self.gc_model.fit(train_gc[:,np.newaxis], train_cov)

        # print score
        score = self.gc_model.score(train_gc[:,np.newaxis], train_cov)

        # determine genomic baseline
        self.learn_gc_base()

        print(' Done in %ds.' % (time.time()-t0))
        print('GC model explains %.4f of variance.' % score, flush=True)

        if out_dir is not None:
            # plot training fit
            regplot_gc(train_gc, train_cov, self.gc_model, '%s/gc_model.pdf' % out_dir)

            # print table
            gc_out = open('%s/gc_table.txt' % out_dir, 'w')
            gc_range = np.arange(0,1,.01)
            gc_norm = self.gc_model.predict(gc_range[:,np.newaxis])
            for i in range(len(gc_range)):
                print(gc_range[i], gc_norm[i], file=gc_out)
            gc_out.close()


    def learn_gc_base(self):
        ''' Determine the genome-wide GC model baseline

        In
         self.gc_model

        Out
         self.gc_base
        '''

        # helper
        chroms_list = list(self.chrom_lengths.keys())
        fragment_sd3 = int(np.round(3*self.fragment_sd))

        # gaussian mask
        gauss_kernel = norm.pdf(np.arange(-fragment_sd3, fragment_sd3), loc=0, scale=self.fragment_sd)
        gauss_invsum = 1.0 / gauss_kernel.sum()

        # sample genomic positions
        sample_every = self.genome_length / 10000
        train_positions = list(np.arange(0, self.genome_length, sample_every))[1:]
        train_gc = []

        # fetch GC%
        for gi in train_positions:
            # determine chromosome and position
            ci, pos = self.index_genome(gi)

            # get sequence
            seq_start = max(0, pos - fragment_sd3)
            seq_end = pos + fragment_sd3
            seq = self.fasta.fetch(chroms_list[ci], seq_start, seq_end)

            # filter for clean sequences
            if len(seq) == 2*fragment_sd3 and seq.find('N') == -1:

                # compute GC%
                seq_gc = np.array([nt in 'CG' for nt in seq], dtype='float32')
                gauss_gc = (seq_gc * gauss_kernel).sum() * gauss_invsum
                train_gc.append(gauss_gc)

        # convert to arrays
        train_gc = np.array(train_gc)

        # compute mean prediction
        self.gc_base = np.mean(self.gc_model.predict(train_gc[:,np.newaxis]))


    def index_genome(self, gi):
        ''' Compute chromosome and position for a genome index.

        Args
         gi (int): Genomic index

        Returns:
         ci (int): Chromosome index.
         pos (int): Position.
        '''

        lengths_list = list(self.chrom_lengths.values())

        # chromosome index
        ci = 0

        # helper counters
        gii = 0
        cii = 0

        # while gi is beyond this chromosome
        while ci < len(lengths_list) and gi - gii > lengths_list[ci]:
            # advance genome index
            gii += lengths_list[ci]

            # advance chromosome
            ci += 1

        # we shouldn't be beyond the chromosomes
        assert(ci < len(lengths_list))

        # set position
        pos = gi - gii

        return ci, pos


    def genome_index(self, ci, pos):
        ''' Compute genome index for an chromosome and position.

        Args
         ci (int): Chromosome index.
         pos (int): Position.

        Returns:
         gi (int): Genomic index
        '''

        lengths_list = list(self.chrom_lengths.values())

        gi = 0
        for gci in range(len(lengths_list)):
            if ci == gci:
                gi += pos
                break
            else:
                gi += lengths_list[gci]
        return gi


    def learn_shift_pair(self, bam_file):
        ''' Learn the optimal fragment shift from paired end fragments. '''

        t0 = time.time()
        print('Learning shift from single-end sequences.', end='', flush=True)

        # read proper pair template lengths
        template_lengths = []
        for align in pysam.Samfile(bam_file):
            if align.is_proper_pair and align.is_read1:
                template_lengths.append(align.template_length)

        # compute mean
        self.shift = int(np.round(np.mean(template_lengths) / 2))

        print(' Done in %ds.' % (time.time()-t0))
        print('Shift: %d' % self.shift, flush=True)


    def learn_shift_single(self, bam_file, shift_min=50, shift_max=350, out_dir=None):
        ''' Learn the optimal fragment shift that maximizes across strand correlation '''

        t0 = time.time()
        print('Learning shift from single-end sequences.', end='', flush=True)

        # find the largest chromosome
        chrom_max = None
        for chrom, clen in self.chrom_lengths.items():
            if clen > self.chrom_lengths.get(chrom_max,0):
                chrom_max = chrom
        chrom_length = self.chrom_lengths[chrom_max]

        # initialize counts
        counts_fwd = np.zeros(chrom_length, dtype='float32')
        counts_rev = np.zeros(chrom_length, dtype='float32')

        # initialize masks
        multi_mask = np.zeros(chrom_length, dtype='bool')

        # compute unique counts for the fwd and rev strands
        for align in pysam.Samfile(bam_file):
            # if aligned to max_chrom
            if not align.is_unmapped and align.reference_name == chrom_max:
                # determine mappability
                multi = align.has_tag('NH') and align.get_tag('NH') > 1

                if align.is_reverse:
                    # determine alignment position
                    ci = align.reference_end

                    if multi:
                        # record as multi-mapping
                        multi_mask[ci] = True
                    else:
                        # count
                        counts_rev[ci] += 1

                else:
                    # determine alignment position
                    ci = align.reference_start

                    if multi:
                        # record as multi-mapping
                        multi_mask[ci] = True
                    else:
                        # count
                        counts_fwd[ci] += 1

        # limit duplicates
        if self.duplicate_max is not None:
            np.copyto(counts_fwd, np.clip(counts_fwd, 0, self.duplicate_max))
            np.copyto(counts_rev, np.clip(counts_rev, 0, self.duplicate_max))

        # mean normalize
        mean_count = 0.5*counts_fwd[~multi_mask].mean(dtype='float64') + 0.5*counts_rev[~multi_mask].mean(dtype='float64')
        counts_fwd -= mean_count
        counts_rev -= mean_count

        # compute pearsonr correlations
        telo_buf = chrom_length // 20
        shifts = np.arange(shift_min, shift_max)
        strand_corrs = np.zeros(len(shifts), dtype='float32')
        for si in range(len(shifts)):
            d = shifts[si]
            counts_dot = np.multiply(counts_fwd[telo_buf:-telo_buf], counts_rev[telo_buf+d:-telo_buf+d])
            counts_mask = multi_mask[telo_buf:-telo_buf] | multi_mask[telo_buf+d:-telo_buf+d]
            strand_corrs[si] = counts_dot[~counts_mask].mean()

        # polynomial regression
        strand_corrs_smooth = gaussian_filter1d(strand_corrs, sigma=12, truncate=3)

        # find max
        self.shift = (shift_min + np.argmax(strand_corrs_smooth)) // 2

        print(' Done in %ds.' % (time.time()-t0))
        print('Shift: %d' % self.shift, flush=True)

        if out_dir is not None:
            # plot training fit
            regplot_shift(shifts, strand_corrs, strand_corrs_smooth, '%s/shift_model.pdf' % out_dir)

            # print table
            shift_out = open('%s/shift_table.txt' % out_dir, 'w')
            for si in range(len(shifts)):
                print(shifts[si], strand_corrs[si], strand_corrs_smooth[si], '*'*int(2*self.shift==shifts[si]), file=shift_out)
            shift_out.close()


    def read_bam(self, bam_file):
        ''' Read alignments from a BAM file into unique and multi data structures. '''

        t0 = time.time()
        print('Reading alignments from BAM.', flush=True, end='')

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

                # determine shift
                if self.shift == 0:
                    # don't shift anyone
                    align_shift = 0
                else:
                    if align.is_proper_pair:
                        # shift proper pairs according to mate
                        align_shift = align.template_length // 2
                    else:
                        # shift others by estimated amount
                        align_shift = self.shift

                # determine alignment position
                chrom_pos = align.reference_start + align_shift
                if align.is_reverse:
                    chrom_pos = align.reference_end - align_shift

                # determine genome index
                gi = self.genome_index(align.reference_id, chrom_pos)

                # determine multi-mapping state
                nh_tag = 1
                if align.has_tag('NH'):
                    nh_tag = align.get_tag('NH')

                # if unique
                if nh_tag == 1:
                    self.unique_counts[gi] += 1

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

        print(' Done in %ds.' % (time.time()-t0), flush=True)

        # convert sparse matrix
        t0 = time.time()
        print('Constructing multi-read CSR matrix.', flush=True, end='')
        self.multi_weight_matrix = csr_matrix((multi_weight,(multi_reads,multi_positions)), shape=(len(multi_read_index),self.genome_length))
        print(' Done in %ds.' % (time.time()-t0), flush=True)

        # validate that weights sum to 1
        multi_sum = self.multi_weight_matrix.sum(axis=1)
        for read_id, ri in multi_read_index.items():
            if not np.isclose(multi_sum[ri], 1, rtol=1e-3):
                print('Multi-weighted coverage for (%s,%s) %f != 1' % (read_id[0],read_id[1],multi_sum[ri]), file=sys.stderr)
                exit(1)


    def infer_active_blocks(self, genome_coverage, min_inactive=50000):
        # compute inactive blocks
        self.active_blocks = []
        active_start = 0
        zero_start = None
        zero_run = 0

        # compute zero booleans
        zero_counts = (genome_coverage == 0)

        i = 0
        for zc in zero_counts:
            # if zero count
            if zc:
                # if it's the first
                if zero_start is None:
                    # set the start
                    zero_start = i

                # increment the run counter
                zero_run += 1

            # if it's >0
            else:
                # if a sufficiently long zero run is ending
                if zero_run >= min_inactive:
                    # unless it's the first nonzero entry
                    if active_start < zero_start:
                        # save the previous active block
                        self.active_blocks.append((active_start,zero_start))

                    # begin a new active block
                    active_start = i

                # set active_start if it the chr starts without a zero block
                if active_start is None:
                    active_start = i

                # refresh zero counters
                zero_start = None
                zero_run = 0

            i += 1

        # consider a final active block
        if zero_start is None or zero_run < min_inactive:
            self.active_blocks.append((active_start, len(genome_coverage)))
        else:
            self.active_blocks.append((active_start, zero_start))

        blocks_out = open('active_blocks.txt', 'w')
        for i in range(len(self.active_blocks)):
            print(i, self.active_blocks[i][0], self.active_blocks[i][1], self.active_blocks[i][1]-self.active_blocks[i][0], file=blocks_out)
        blocks_out.close()


    def infer_active_blocks_groupby(self, genome_coverage, min_inactive=50000):
        ''' Infer active genomic blocks that we'll need to consider.

        The non-groupby version is inefficient. This one should improve it,
         but it's unfinished.
        '''
        # compute inactive blocks
        self.active_blocks = []
        active_start = 0
        zero_run = 0

        # compute zero runs
        constant_runs = [(k, sum([True for b in g])) for k,g in itertools.groupby(genome_coverage == 0)]

        # # parse runs
        gi = 0
        for pos_count, run_len in constant_runs:
            if pos_count == 0:
                zero_run = run_len
            else:
                if zero_run > min_inactive:
                    # unless it's the first nonzero entry
                    if active_start < gi - zero_run:
                        # save the previous active block
                        self.active_blocks.append((active_start,zero_start))

                    # begin a new active block
                    active_start = i


    def write_bigwig(self, bigwig_file, single_or_pair, zero_eps=.003):
        ''' Compute and write out coverage to bigwig file.

        Go chromosome by chromosome here to facilitate printing,
        and save memory.

        In:
         bigwig_file: BigWig filename.
        '''

        print('Outputting coverage')

        bigwig_out = pyBigWig.open(bigwig_file, 'w')

        # add header
        bigwig_out.addHeader(list(self.chrom_lengths.items()))

        chroms_list = list(self.chrom_lengths.keys())
        lengths_list = list(self.chrom_lengths.values())

        gi = 0
        for ci in range(len(chroms_list)):
            t0 = time.time()
            print('  %s' % chroms_list[ci], end='', flush=True)
            cl = lengths_list[ci]

            # compute multi-mapper chromosome coverage
            chrom_weight_matrix = self.multi_weight_matrix[:,gi:gi+cl]
            chrom_multi_coverage = np.array(chrom_weight_matrix.sum(axis=0, dtype='float32')).squeeze()

            # sum with unique coverage
            chrom_coverage_array = self.unique_counts[gi:gi+cl] + chrom_multi_coverage

            # limit duplicates
            if self.duplicate_max is not None:
                chrom_coverage_array = np.clip(chrom_coverage_array, 0, self.duplicate_max)

            # Gaussian smooth
            if self.smooth_sd > 0:
                chrom_coverage_array = gaussian_filter1d(chrom_coverage_array, sigma=self.smooth_sd, truncate=3)

            # normalize for GC content
            if self.gc_model is not None:
                pre_sum = chrom_coverage_array.sum()
                chrom_coverage_array = self.gc_normalize(chroms_list[ci], chrom_coverage_array)
                post_sum = chrom_coverage_array.sum()
                # print('  GC normalization altered mean coverage by %.3f (%d/%d)' % (post_sum/pre_sum, post_sum, pre_sum))

            # correct for double coverage in paired end data
            if single_or_pair == 'pair':
                chrom_coverage_array /= 2.0

            # set small values to zero
            chrom_coverage_array[chrom_coverage_array < zero_eps] = 0

            # add to bigwig
            bigwig_out.addEntries(chroms_list[ci], 0, values=chrom_coverage_array.astype('float16'), span=1, step=1)

            # update genomic index
            gi += cl

            # clean up temp storage
            gc.collect()

            # update user
            print(', %ds' % (time.time()-t0), flush=True)

        t0 = time.time()
        bigwig_out.close()
        print(' Close BigWig: %ds' % (time.time()-t0))


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
