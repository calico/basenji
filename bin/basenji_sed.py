#!/usr/bin/env python
from optparse import OptionParser
import os
import pickle
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
    parser.add_option('-b', dest='batch_size', default=None, type='int', help='Batch size [Default: %default]')
    parser.add_option('-c', dest='csv', default=False, action='store_true', help='Print table as CSV [Default: %default]')
    parser.add_option('-g', dest='genome_file', default='%s/assembly/human.hg19.genome'%os.environ['HG19'], help='Chromosome lengths file [Default: %default]')
    parser.add_option('-i', dest='index_snp', default=False, action='store_true', help='SNPs are labeled with their index SNP as column 6 [Default: %default]')
    parser.add_option('-o', dest='out_dir', default='sed', help='Output directory for tables and plots [Default: %default]')
    parser.add_option('-p', dest='processes', default=None, type='int', help='Number of processes, passed by multi script')
    parser.add_option('-s', dest='score', default=False, action='store_true', help='SNPs are labeled with scores as column 7 [Default: %default]')
    parser.add_option('-t', dest='target_wigs_file', default=None, help='Store target values, extracted from this list of WIG files')
    parser.add_option('--ti', dest='track_indexes', help='Comma-separated list of target indexes to output BigWig tracks')
    (options,args) = parser.parse_args()

    if len(args) == 4:
        # single worker
        params_file = args[0]
        model_file = args[1]
        genes_hdf5_file = args[2]
        vcf_file = args[3]

    elif len(args) == 6:
        # multi worker
        options_pkl_file = args[0]
        params_file = args[1]
        model_file = args[2]
        genes_hdf5_file = args[3]
        vcf_file = args[4]
        worker_num = int(args[5])

         # load options
        options_pkl = open(options_pkl_file, 'rb')
        options = pickle.load(options_pkl)
        options_pkl.close()

        # update output directory
        options.out_dir = '%s/job%d' % (options.out_dir, worker_num)

    else:
        parser.error('Must provide parameters and model files, genes HDF5 file, and QTL VCF file')

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    if options.track_indexes is None:
        options.track_indexes = []
    else:
        options.track_indexes = [int(ti) for ti in options.track_indexes.split(',')]
        if not os.path.isdir('%s/tracks' % options.out_dir):
            os.mkdir('%s/tracks' % options.out_dir)

    #################################################################
    # reads in genes HDF5

    genes_hdf5_in = h5py.File(genes_hdf5_file)

    seq_coords, seqs_1hot, seq_transcripts, transcript_targets, target_labels = read_hdf5(genes_hdf5_in)

    #################################################################
    # filter for worker sequences

    if options.processes is not None:
        seq_mask = np.array([si % options.processes == worker_num for si in range(seqs_1hot.shape[0])])
        seqs_1hot = seqs_1hot[seq_mask,:,:]
        seq_coords = [seq_coords[si] for si in range(len(seq_coords)) if seq_mask[si]]
        seq_transcripts = [seq_transcripts[si] for si in range(len(seq_transcripts)) if seq_mask[si]]


    #################################################################
    # prep SNPs

    # load SNPs
    snps = basenji.vcf.vcf_snps(vcf_file, options.index_snp, options.score, False)

    # intersect w/ segments
    snps_segs = basenji.vcf.intersect_snps_seqs(vcf_file, seq_coords)


    #################################################################
    # construct SNP sequences

    snp_seqs_1hot = []

    # for each SNP
    for snp_i in range(len(snps)):
        # for each segment it overlaps
        for seg_i in snps_segs[snp_i]:
            # get the segment coordinates
            seq_chrom, seq_start, seq_end = seq_coords[seg_i]

            # determine the SNP's position in the segment
            snp_seq_pos = snps[snp_i].pos-1 - seq_start

            # write reference allele
            snp_seqs_1hot.append(np.copy(seqs_1hot[seg_i]))
            basenji.dna_io.hot1_set(snp_seqs_1hot[-1], snp_seq_pos, snps[snp_i].ref_allele)

            # write alternative allele
            snp_seqs_1hot.append(np.copy(seqs_1hot[seg_i]))
            basenji.dna_io.hot1_set(snp_seqs_1hot[-1], snp_seq_pos, snps[snp_i].alt_alleles[0])

    snp_seqs_1hot = np.array(snp_seqs_1hot)
    print('snp_seqs_1hot', snp_seqs_1hot.shape)


    #################################################################
    # setup model

    job = basenji.dna_io.read_job_params(params_file)

    job['batch_length'] = snp_seqs_1hot.shape[1]
    job['seq_depth'] = snp_seqs_1hot.shape[2]
    job['target_pool'] = int(np.array(genes_hdf5_in['pool_width']))

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

        # initialize prediction stream
        seq_preds = basenji.stream.PredStream(sess, model, snp_seqs_1hot, 128)

        # determine prediction buffer
        pred_buffer = model.batch_buffer // model.target_pool

        #################################################################
        # collect and print SEDs

        header_cols = ('rsid', 'index', 'score', 'ref', 'alt', 'gene', 'target', 'ref_pred', 'alt pred', 'ser', 'sed')
        if options.csv:
            sed_out = open('%s/sed_table.csv' % options.out_dir, 'w')
            print(','.join(header_cols), file=sed_out)
        else:
            sed_out = open('%s/sed_table.txt' % options.out_dir, 'w')
            print(' '.join(header_cols), file=sed_out)

        pi = 0
        for snp_i in range(len(snps)):
            snp = snps[snp_i]

            # set index SNP
            snp_is = '%-13s' % '.'
            if options.index_snp:
                snp_is = '%-13s' % snp.index_snp

            # set score
            snp_score = '%5s' % '.'
            if options.score:
                snp_score = '%5.3f' % snp.score

            for seg_i in snps_segs[snp_i]:
                # get reference prediction (LxT)
                ref_preds = seq_preds[pi]
                pi += 1

                # get alternate prediction (LxT)
                alt_preds = seq_preds[pi]
                pi += 1

                # find genes
                for transcript, tx_pos in seq_transcripts[seg_i]:
                    # compute distance between SNP and gene
                    tx_gpos = seq_coords[seg_i][1] + (tx_pos + 0.5)*model.target_pool
                    snp_dist = abs(tx_gpos - snp.pos)

                    # compute transcript pos in predictions
                    tx_pos_buf = tx_pos - pred_buffer

                    for ti in range(job['num_targets']):
                        # compute SED score
                        snp_gene_sed = (alt_preds[tx_pos_buf,ti] - ref_preds[tx_pos_buf,ti])
                        snp_gene_ser = np.log2(alt_preds[tx_pos_buf,ti]+1) - np.log2(ref_preds[tx_pos_buf,ti]+1)

                        # print to table
                        cols = (snp.rsid, snp_is, snp_score, basenji.vcf.cap_allele(snp.ref_allele), basenji.vcf.cap_allele(snp.alt_alleles[0]), transcript, snp_dist, target_labels[ti], ref_preds[tx_pos_buf,ti], alt_preds[tx_pos_buf,ti], snp_gene_sed, snp_gene_ser)
                        if options.csv:
                            print(','.join([str(c) for c in cols]), file=sed_out)
                        else:
                            print('%-13s %s %5s %6s %6s %12s %5d %12s %6.4f %6.4f %7.4f %7.4f' % cols, file=sed_out)

                # print tracks
                for ti in options.track_indexes:
                    ref_bw_file = '%s/tracks/%s_%s_t%d_ref.bw' % (options.out_dir, snp.rsid, seg_i, ti)
                    alt_bw_file = '%s/tracks/%s_%s_t%d_alt.bw' % (options.out_dir, snp.rsid, seg_i, ti)
                    ref_bw_open = bigwig_open(ref_bw_file, options.genome_file)
                    alt_bw_open = bigwig_open(alt_bw_file, options.genome_file)

                    seq_chrom, seq_start, seq_end = seq_coords[seg_i]
                    bw_chroms = [seq_chrom]*ref_preds.shape[0]
                    bw_starts = [int(seq_start + model.batch_buffer + bi*model.target_pool) for bi in range(ref_preds.shape[0])]
                    bw_ends = [int(bws + model.target_pool) for bws in bw_starts]

                    ref_values = [float(p) for p in ref_preds[:,ti]]
                    ref_bw_open.addEntries(bw_chroms, bw_starts, ends=bw_ends, values=ref_values)

                    alt_values = [float(p) for p in alt_preds[:,ti]]
                    alt_bw_open.addEntries(bw_chroms, bw_starts, ends=bw_ends, values=alt_values)

                    ref_bw_open.close()
                    alt_bw_open.close()


    sed_out.close()


    #################################################################
    # clean up

    genes_hdf5_in.close()


def read_hdf5(genes_hdf5_in):
    #######################################
    # seq_coords

    seq_chrom = [chrom.decode('UTF-8') for chrom in genes_hdf5_in['seq_chrom']]
    seq_start = list(genes_hdf5_in['seq_start'])
    seq_end = list(genes_hdf5_in['seq_end'])
    seq_coords = list(zip(seq_chrom,seq_start,seq_end))

    #######################################
    # seqs_1hot

    seqs_1hot = genes_hdf5_in['seqs_1hot']
    print(seqs_1hot.shape)

    #######################################
    # transcript_map

    transcripts = [tx.decode('UTF-8') for tx in genes_hdf5_in['transcripts']]
    transcript_index = list(genes_hdf5_in['transcript_index'])
    transcript_pos = list(genes_hdf5_in['transcript_pos'])

    transcript_map = {}
    for ti in range(len(transcripts)):
        transcript_map[transcripts[ti]] = (transcript_index[ti], transcript_pos[ti])


    #######################################
    # transcript_targets / target_labels

    if 'transcript_targets' in genes_hdf5_in:
        transcript_targets = genes_hdf5_in['transcript_targets']
        target_labels = [tl.decode('UTF-8') for tl in genes_hdf5_in['target_labels']]
    else:
        transcript_targets = None
        target_labels = None

    #######################################
    # seq_transcripts

    seq_transcripts = []
    for si in range(len(seq_coords)):
        seq_transcripts.append([])

    for transcript in transcript_map:
        tx_index, tx_pos = transcript_map[transcript]
        seq_transcripts[tx_index].append((transcript,tx_pos))

    return seq_coords, seqs_1hot, seq_transcripts, transcript_targets, target_labels


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
