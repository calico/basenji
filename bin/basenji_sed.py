#!/usr/bin/env python
from optparse import OptionParser
import os
import subprocess
import sys
import tempfile

import h5py
import numpy as np
import pysam
import tensorflow as tf

import basenji

'''
basenji_sed.py

Compute SNP expression difference scores for variants in a VCF file.

Note:
 -I'm having trouble verifying that I'm not double counting the scenario where
  two transcripts map to the same TSS. But I should verify that.
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <params_file> <model_file> <genes_hdf5_file> <vcf_file>'
    parser = OptionParser(usage)
    parser.add_option('-b', dest='batch_size', default=None, type='int', help='Batch size [Default: %default]')
    parser.add_option('-c', dest='csv', default=False, action='store_true', help='Print table as CSV [Default: %default]')
    parser.add_option('-i', dest='index_snp', default=False, action='store_true', help='SNPs are labeled with their index SNP as column 6 [Default: %default]')
    parser.add_option('-o', dest='out_dir', default='sed', help='Output directory for tables and plots [Default: %default]')
    parser.add_option('-s', dest='score', default=False, action='store_true', help='SNPs are labeled with scores as column 7 [Default: %default]')
    parser.add_option('-t', dest='targets_file', default=None, help='File specifying target indexes and labels in table format')
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
    # reads in genes HDF5

    genes_hdf5_in = h5py.File(genes_hdf5_file)

    seg_chrom = [chrom.decode('UTF-8') for chrom in genes_hdf5_in['seg_chrom']]
    seg_start = np.array(genes_hdf5_in['seg_start'])
    seg_end = np.array(genes_hdf5_in['seg_end'])
    seqs_segments = list(zip(seg_chrom,seg_start,seg_end))

    seqs_1hot = genes_hdf5_in['seqs_1hot']

    transcripts = [tx.decode('UTF-8') for tx in genes_hdf5_in['transcripts']]
    transcript_index = np.array(genes_hdf5_in['transcript_index'])
    transcript_pos = np.array(genes_hdf5_in['transcript_pos'])

    transcript_map = {}
    for ti in range(len(transcripts)):
        transcript_map[transcripts[ti]] = (transcript_index[ti], transcript_pos[ti])


    #################################################################
    # prep SNPs

    # load SNPs
    snps = basenji.vcf.vcf_snps(vcf_file, options.index_snp, options.score, False)

    # intersect w/ segments
    snps_segs = intersect_snps(vcf_file, seqs_segments)


    #################################################################
    # construct SNP sequences

    snp_seqs_1hot = []

    for snp_i in range(len(snps)):
        for seg_i in snps_segs[snp_i]:
            seg_chrom, seg_start, seg_end = seqs_segments[seg_i]

            # determine the SNP's position in the segment
            snp_seg_pos = snps[snp_i].pos - seg_start

            # write reference allele
            snp_seqs_1hot.append(seqs_1hot[snp_i])
            basenji.dna_io.hot1_set(snp_seqs_1hot[-1], snp_seg_pos, snps[snp_i].ref_allele)

            # write alternative allele
            snp_seqs_1hot.append(seqs_1hot[snp_i])
            basenji.dna_io.hot1_set(snp_seqs_1hot[-1], snp_seg_pos, snps[snp_i].alt_alleles[0])

    snp_seqs_1hot = np.array(snp_seqs_1hot)


    #################################################################
    # setup model

    job = basenji.dna_io.read_job_params(params_file)

    job['batch_length'] = snp_seqs_1hot.shape[1]
    job['seq_depth'] = snp_seqs_1hot.shape[2]

    if 'num_targets' not in job:
        print("Must specify number of targets (num_targets) in the parameters file. I know, it's annoying. Sorry.", file=sys.stderr)
        exit(1)

    # build model
    dr = basenji.rnn.RNN()
    dr.build(job)

    if options.batch_size is not None:
        dr.batch_size = options.batch_size


    #################################################################
    # predict

    # initialize batcher
    batcher = basenji.batcher.Batcher(snp_seqs_1hot, batch_size=dr.batch_size)

    # initialie saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # load variables into session
        saver.restore(sess, model_file)

        # predict
        seq_preds = dr.predict(sess, batcher)


    #################################################################
    # collect and pred SEDs

    if options.targets_file is None:
        target_labels = ['t%d' % ti for ti in range(seq_preds.shape[1])]
    else:
        target_labels = [line.split()[0] for line in open(options.targets_file)]

    header_cols = ('rsid', 'index', 'score', 'ref', 'alt', 'gene', 'target', 'ref_pred', 'alt pred', 'sed')
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
            for transcript in transcript_map:
                tx_index, tx_pos = transcript_map[transcript]
                if tx_index == seg_i:
                    for ti in range(seq_preds.shape[2]):
                        snp_gene_sed = (alt_preds[tx_pos,ti] - ref_preds[tx_pos,ti])

                        cols = (snp.rsid, snp_is, snp_score, basenji.vcf.cap_allele(snp.ref_allele), basenji.vcf.cap_allele(snp.alt_alleles[0]), transcript, target_labels[ti], ref_preds[tx_pos,ti], alt_preds[tx_pos,ti], snp_gene_sed)
                        if options.csv:
                            print(','.join([str(c) for c in cols]), file=sed_out)
                        else:
                            print('%-13s %s %5s %6s %6s %12s %12s %6.4f %6.4f %7.4f' % cols, file=sed_out)

    sed_out.close()


    #################################################################
    # clean up

    genes_hdf5_in.close()


def intersect_snps(vcf_file, seqs_segments):
    # print segments to BED
    # hash segments to indexes
    seg_temp = tempfile.NamedTemporaryFile()
    seg_bed_file = seg_temp.name
    seg_bed_out = open(seg_bed_file, 'w')
    segment_indexes = {}
    for si in range(len(seqs_segments)):
        segment_indexes[seqs_segments[si]] = si
        print('%s\t%d\t%d' % seqs_segments[si], file=seg_bed_out)
    seg_bed_out.close()

    # hash SNPs to indexes
    snp_indexes = {}
    si = 0

    vcf_in = open(vcf_file)
    line = vcf_in.readline()
    while line[0] == '#':
        line = vcf_in.readline()
    while line:
        a = line.split()
        snp_id = a[2]
        snp_indexes[snp_id] = si
        si += 1
        line = vcf_in.readline()
    vcf_in.close()

    # initialize list of lists
    snp_segs = []
    for i in range(len(snp_indexes)):
        snp_segs.append([])

    # intersect
    p = subprocess.Popen('bedtools intersect -wo -a %s -b %s' % (vcf_file, seg_bed_file), shell=True, stdout=subprocess.PIPE)
    for line in p.stdout:
        line = line.decode('UTF-8')
        a = line.split()
        snp_id = a[2]
        seg_chrom = a[-4]
        seg_start = int(a[-3])
        seg_end = int(a[-2])
        seg_key = (seg_chrom,seg_start,seg_end)

        snp_segs[snp_indexes[snp_id]].append(segment_indexes[seg_key])

    p.communicate()

    return snp_segs


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
