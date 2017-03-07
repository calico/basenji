#!/usr/bin/env python
from optparse import OptionParser
import os
import subprocess
import sys

import h5py
import numpy as np
import pyBigWig
import tensorflow as tf

import basenji

from basenji_test import bigwig_open

'''
basenji_sed.py

Compute SNP expression difference scores for variants in a VCF file.

Note:
 -Generating snp_seqs_1hot altogether is going to run out of memory for
   larger VCF files. One solution is to switch to batches, like basenji_sad.py.
   Another is a streaming interface like PredStream.
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <params_file> <model_file> <genes_hdf5_file> <vcf_file>'
    parser = OptionParser(usage)
    parser.add_option('-c', dest='csv', default=False, action='store_true', help='Print table as CSV [Default: %default]')
    parser.add_option('-g', dest='genome_file', default='%s/assembly/human.hg19.genome'%os.environ['HG19'], help='Chromosome lengths file [Default: %default]')
    parser.add_option('-i', dest='index_snp', default=False, action='store_true', help='SNPs are labeled with their index SNP as column 6 [Default: %default]')
    parser.add_option('-o', dest='out_dir', default='sed', help='Output directory for tables and plots [Default: %default]')
    parser.add_option('-s', dest='score', default=False, action='store_true', help='SNPs are labeled with scores as column 7 [Default: %default]')
    parser.add_option('-t', dest='target_wigs_file', default=None, help='Store target values, extracted from this list of WIG files')
    (options,args) = parser.parse_args()

    if len(args) != 4:
        parser.error('Must provide parameters and model files, genes HDF5 file, and QTL VCF file')
    else:
        params_file = args[0]
        model_file = args[1]
        genes_hdf5_file = args[2]
        vcf_file = args[3]

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    #################################################################
    # read in genes HDF5

    genes_hdf5_in = h5py.File(genes_hdf5_file)

    seq_chrom = [chrom.decode('UTF-8') for chrom in genes_hdf5_in['seq_chrom']]
    seq_start = list(genes_hdf5_in['seq_start'])
    seq_end = list(genes_hdf5_in['seq_end'])
    seq_coords = list(zip(seq_chrom,seq_start,seq_end))

    seqs_1hot = genes_hdf5_in['seqs_1hot']
    print(seqs_1hot.shape)

    transcripts = [tx.decode('UTF-8') for tx in genes_hdf5_in['transcripts']]
    transcript_index = list(genes_hdf5_in['transcript_index'])
    transcript_pos = list(genes_hdf5_in['transcript_pos'])

    transcript_map = {}
    for ti in range(len(transcripts)):
        transcript_map[transcripts[ti]] = (transcript_index[ti], transcript_pos[ti])

    if 'transcript_targets' in genes_hdf5_in:
        transcript_targets = genes_hdf5_in['transcript_targets']
        target_labels = [tl.decode('UTF-8') for tl in genes_hdf5_in['target_labels']]
    else:
        transcript_targets = None
        target_labels = None

    # map sequences to transcripts
    seq_transcripts = []
    for si in range(len(seq_coords)):
        seq_transcripts.append([])

    for transcript in transcript_map:
        tx_index, tx_pos = transcript_map[transcript]
        seq_transcripts[tx_index].append((transcript,tx_pos))


    #################################################################
    # prep SNPs

    # load SNPs
    snps = basenji.vcf.vcf_snps(vcf_file, options.index_snp, options.score, False)

    # intersect w/ segments
    seq_snps = basenji.vcf.intersect_seq_snps(vcf_file, seq_coords)

    # filter sequences for overlaps
    seq_mask = np.array([len(seq_snps[si]) > 0 for si in seq_snps])
    seqs_1hot = seqs_1hot[seq_mask]
    seq_coords = [seq_coords[si] for si in range(len(seq_coords)) if seq_mask[si]]
    seq_transcripts = [seq_transcripts[si] for si in range(len(seq_transcripts)) if seq_mask[si]]

    # transcript data structures break
    del transcript_map


    #################################################################
    # setup model

    job = basenji.dna_io.read_job_params(params_file)

    job['batch_length'] = seqs_1hot.shape[1]
    job['seq_depth'] = seqs_1hot.shape[2]
    job['target_pool'] = int(np.array(genes_hdf5_in['pool_width']))
    job['save_reprs'] = True

    if transcript_targets is not None:
        job['num_targets'] = transcript_targets.shape[1]

    if 'num_targets' not in job:
        print("Must specify number of targets (num_targets) in the parameters file. I know, it's annoying. Sorry.", file=sys.stderr)
        exit(1)

    # build model
    model = basenji.rnn.RNN()
    model.build(job)

    # label targets
    if target_labels is None:
        if options.targets_file is None:
            target_labels = ['t%d' % ti for ti in range(job['num_targets'])]
        else:
            target_labels = [line.split()[0] for line in open(options.targets_file)]


    #################################################################
    # predict

    # initialize saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # load variables into session
        saver.restore(sess, model_file)

        # determine prediction buffer
        pred_buffer = model.batch_buffer // model.target_pool

        # initialize prediction stream
        seq_pg = basenji.stream.PredGradStream(sess, model, seqs_1hot, 16)

        #################################################################
        # collect and print SEDs

        header_cols = ('rsid', 'index', 'score', 'ref', 'alt', 'gene', 'target', 'ref_pred', 'alt pred', 'ser', 'sed')
        if options.csv:
            sed_out = open('%s/sed_table.csv' % options.out_dir, 'w')
            print(','.join(header_cols), file=sed_out)
        else:
            sed_out = open('%s/sed_table.txt' % options.out_dir, 'w')
            print(' '.join(header_cols), file=sed_out)

        for soi in range(len(seq_snps)):
            # get predictions and gradients
            preds, grads = seq_pg[soi]

            # for each overlapping SNP
            for snp_i in seq_snps[soi]:
                snp = snps[snp_i]
                print(snp)

                # set index SNP
                snp_is = '%-13s' % '.'
                if options.index_snp:
                    snp_is = '%-13s' % snp.index_snp

                # set score
                snp_score = '%5s' % '.'
                if options.score:
                    snp_score = '%5.3f' % snp.score

                # set nt indexes
                ref_nt = basenji.vcf.cap_allele(snp.ref_allele)
                ref_nt_i = nt_index(ref_nt)
                alt_nt = basenji.vcf.cap_allele(snp.alt_alleles[0])
                alt_nt_i = nt_index(alt_nt)

                # find genes
                for transcript, tx_pos in sequence_transcripts[soi]:
                    # compute distance between SNP and gene
                    tx_gpos = seq_coords[soi][1] + (tx_pos + 0.5)*model.target_pool
                    snp_dist = abs(tx_gpos - snp.pos)

                    # compute transcript pos in predictions
                    tx_pos_buf = tx_pos - pred_buffer

                    for ti in range(job['num_targets']):
                        # compute SED score
                        # snp_gene_sed = (alt_preds[tx_pos_buf,ti] - ref_preds[tx_pos_buf,ti])
                        # snp_gene_ser = np.log2(alt_preds[tx_pos_buf,ti]+1) - np.log2(ref_preds[tx_pos_buf,ti]+1)
                        snp_gene_sed = grads[tx_pos_buf,alt_nt_i,ti] - grads[tx_pos_buf,ref_nt_i,ti]

                        # print to table
                        cols = (snp.rsid, snp_is, snp_score, ref_nt, alt_nt, transcript, snp_dist, target_labels[ti], preds[tx_pos_buf,ti], snp_gene_sed)
                        if options.csv:
                            print(','.join([str(c) for c in cols]), file=sed_out)
                        else:
                            print('%-13s %s %5s %6s %6s %12s %5d %12s %6.4f %7.4f' % cols, file=sed_out)

    sed_out.close()

    #################################################################
    # clean up

    genes_hdf5_in.close()


def nt_index(nt):
    if nt == 'A':
        nti = 0
    elif nt == 'C':
        nti = 1
    elif nt == 'G':
        nti = 2
    elif nt == 'T':
        nti = 3
    else:
        print('Cannot recognize SNP nt %s' % nt, file=sys.stderr)
        nti = None

    return nti

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
