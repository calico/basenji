#!/usr/bin/env python
from optparse import OptionParser
import gc
import os
import pdb
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
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <params_file> <model_file> <genes_hdf5_file> <vcf_file>'
    parser = OptionParser(usage)
    parser.add_option('-a', dest='all_sed', default=False, action='store_true', help='Print all variant-gene pairs, as opposed to only nonzero [Default: %default]')
    parser.add_option('-b', dest='batch_size', default=None, type='int', help='Batch size [Default: %default]')
    parser.add_option('-c', dest='csv', default=False, action='store_true', help='Print table as CSV [Default: %default]')
    parser.add_option('-g', dest='genome_file', default='%s/assembly/human.hg19.genome'%os.environ['HG19'], help='Chromosome lengths file [Default: %default]')
    parser.add_option('-o', dest='out_dir', default='sed', help='Output directory for tables and plots [Default: %default]')
    parser.add_option('-p', dest='processes', default=None, type='int', help='Number of processes, passed by multi script')
    parser.add_option('--rc', dest='rc', default=False, action='store_true', help='Average the forward and reverse complement predictions when testing [Default: %default]')
    parser.add_option('--ti', dest='track_indexes', help='Comma-separated list of target indexes to output BigWig tracks')
    parser.add_option('-x', dest='transcript_table', default=False, action='store_true', help='Print transcript table in addition to gene [Default: %default]')
    parser.add_option('-w', dest='tss_width', default=1, type='int', help='Width of bins considered to quantify TSS transcription [Default: %default]')
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

    seq_coords, seqs_1hot, seq_transcripts, transcript_targets, transcript_genes, multi_seq_genes, target_labels = read_hdf5(genes_hdf5_in)

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
    snps = basenji.vcf.vcf_snps(vcf_file)

    # intersect w/ segments
    print('Intersecting gene sequences with SNPs...', flush=True, end='')
    seqs_snps = basenji.vcf.intersect_seqs_snps(vcf_file, seq_coords, vision_p=0.5)
    print('done', flush=True)


    #################################################################
    # determine SNP sequences to be needed

    seqs_snps_list = []
    for seq_i in range(seqs_1hot.shape[0]):
        seq_chrom, seq_start, seq_end = seq_coords[seq_i]

        if seqs_snps[seq_i]:
            # add major allele
            seqs_snps_list.append((seq_i,None,None))

            # add minor alleles
            for snp_i in seqs_snps[seq_i]:
                # determine SNP position wrt sequence
                snp_seq_pos = snps[snp_i].pos-1 - seq_start

                # update primary sequence to use major allele
                basenji.dna_io.hot1_set(seqs_1hot[seq_i], snp_seq_pos, snps[snp_i].ref_allele)
                assert(basenji.dna_io.hot1_get(seqs_1hot[seq_i], snp_seq_pos) ==  snps[snp_i].ref_allele)

                # append descriptive tuple to list
                seqs_snps_list.append((seq_i, snp_seq_pos, snps[snp_i].alt_alleles[0]))


    #################################################################
    # setup model

    job = basenji.dna_io.read_job_params(params_file)

    job['batch_length'] = seqs_1hot.shape[1]
    job['seq_depth'] = seqs_1hot.shape[2]
    job['target_pool'] = int(np.array(genes_hdf5_in['pool_width']))

    if 'num_targets' not in job and transcript_targets is not None:
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
    # compute, collect, and print SEDs

    header_cols = ('rsid', 'ref', 'alt', 'gene', 'tss_dist', 'target', 'ref_pred', 'alt_pred', 'sed', 'ser')
    if options.csv:
        sed_gene_out = open('%s/sed_gene.csv' % options.out_dir, 'w')
        print(','.join(header_cols), file=sed_gene_out)
        if options.transcript_table:
            sed_tx_out = open('%s/sed_tx.csv' % options.out_dir, 'w')
            print(','.join(header_cols), file=sed_tx_out)

    else:
        sed_gene_out = open('%s/sed_gene.txt' % options.out_dir, 'w')
        print(' '.join(header_cols), file=sed_gene_out)
        if options.transcript_table:
            sed_tx_out = open('%s/sed_tx.txt' % options.out_dir, 'w')
            print(' '.join(header_cols), file=sed_tx_out)

    # helper variables
    adj = options.tss_width // 2
    pred_buffer = model.batch_buffer // model.target_pool

    # initialize saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # load variables into session
        saver.restore(sess, model_file)

        # initialize prediction stream
        seq_preds = PredStream(sess, model, seqs_1hot, seqs_snps_list, 128, options.rc)

        # prediction index
        pi = 0

        for seq_i in range(seqs_1hot.shape[0]):
            if seqs_snps[seq_i]:
                # get reference prediction (LxT)
                ref_preds = seq_preds[pi]
                pi += 1

                for snp_i in seqs_snps[seq_i]:
                    snp = snps[snp_i]

                    # get alternate prediction (LxT)
                    alt_preds = seq_preds[pi]
                    pi += 1

                    # initialize gene data structures
                    gene_pos_preds = {} # gene -> pos -> (ref_preds,alt_preds)
                    snp_dist_gene = {}

                    # process transcripts
                    for transcript, tx_pos in seq_transcripts[seq_i]:
                        # get gene id
                        gene = transcript_genes[transcript]

                        # compute distance between SNP and TSS
                        tx_gpos = seq_coords[seq_i][1] + (tx_pos + 0.5)*model.target_pool
                        snp_dist = abs(tx_gpos - snp.pos)
                        if gene in snp_dist_gene:
                            snp_dist_gene[gene] = min(snp_dist_gene[gene], snp_dist)
                        else:
                            snp_dist_gene[gene] = snp_dist

                        # compute transcript pos in predictions
                        tx_pos_buf = tx_pos - pred_buffer

                        # hash transcription positions and predictions to gene id
                        for tx_pos_i in range(tx_pos_buf-adj,tx_pos_buf+adj+1):
                            gene_pos_preds.setdefault(gene,{})[tx_pos_i] = (ref_preds[tx_pos_i,:],alt_preds[tx_pos_i,:])

                        # accumulate transcript predictions by (possibly) summing adjacent positions
                        ap = alt_preds[tx_pos_buf-adj:tx_pos_buf+adj+1,:].sum(axis=0)
                        rp = ref_preds[tx_pos_buf-adj:tx_pos_buf+adj+1,:].sum(axis=0)

                        # compute SED scores
                        snp_tx_sed = ap - rp
                        snp_tx_ser = np.log2(ap+1) - np.log2(rp+1)

                        # print rows to transcript table
                        if options.transcript_table:
                            for ti in range(ref_preds.shape[1]):
                                if options.all_sed or not np.isclose(snp_tx_sed[ti], 0, atol=1e-4):
                                    cols = (snp.rsid, basenji.vcf.cap_allele(snp.ref_allele), basenji.vcf.cap_allele(snp.alt_alleles[0]), transcript, snp_dist, target_labels[ti], rp[ti], ap[ti], snp_tx_sed[ti], snp_tx_ser[ti])
                                    if options.csv:
                                        print(','.join([str(c) for c in cols]), file=sed_tx_out)
                                    else:
                                        print('%-13s %s %5s %12s %5d %12s %6.4f %6.4f %7.4f %7.4f' % cols, file=sed_tx_out)

                    # process genes
                    for gene in gene_pos_preds:
                        gene_str = gene
                        if gene in multi_seq_genes:
                            gene_str = '%s_multi' % gene

                        # sum gene preds across positions
                        gene_rp = np.zeros(ref_preds.shape[1])
                        gene_ap = np.zeros(alt_preds.shape[1])
                        for pos_i in gene_pos_preds[gene]:
                            pos_rp, pos_ap = gene_pos_preds[gene][pos_i]
                            gene_rp += pos_rp
                            gene_ap += pos_ap

                        # compute SED scores
                        snp_gene_sed = gene_ap - gene_rp
                        snp_gene_ser = np.log2(gene_ap+1) - np.log2(gene_rp+1)

                        # print rows to gene table
                        for ti in range(ref_preds.shape[1]):
                            if options.all_sed or not np.isclose(snp_gene_sed[ti], 0, atol=1e-4):
                                cols = [snp.rsid, basenji.vcf.cap_allele(snp.ref_allele), basenji.vcf.cap_allele(snp.alt_alleles[0]), gene_str, snp_dist_gene[gene], target_labels[ti], gene_rp[ti], gene_ap[ti], snp_gene_sed[ti], snp_gene_ser[ti]]
                                if options.csv:
                                    print(','.join([str(c) for c in cols]), file=sed_gene_out)
                                else:
                                    print('%-13s %s %5s %12s %5d %12s %6.4f %6.4f %7.4f %7.4f' % tuple(cols), file=sed_gene_out)

                    # print tracks
                    for ti in options.track_indexes:
                        ref_bw_file = '%s/tracks/%s_%s_t%d_ref.bw' % (options.out_dir, snp.rsid, seq_i, ti)
                        alt_bw_file = '%s/tracks/%s_%s_t%d_alt.bw' % (options.out_dir, snp.rsid, seq_i, ti)
                        diff_bw_file = '%s/tracks/%s_%s_t%d_diff.bw' % (options.out_dir, snp.rsid, seq_i, ti)
                        ref_bw_open = bigwig_open(ref_bw_file, options.genome_file)
                        alt_bw_open = bigwig_open(alt_bw_file, options.genome_file)
                        diff_bw_open = bigwig_open(diff_bw_file, options.genome_file)

                        seq_chrom, seq_start, seq_end = seq_coords[seq_i]
                        bw_chroms = [seq_chrom]*ref_preds.shape[0]
                        bw_starts = [int(seq_start + model.batch_buffer + bi*model.target_pool) for bi in range(ref_preds.shape[0])]
                        bw_ends = [int(bws + model.target_pool) for bws in bw_starts]

                        ref_values = [float(p) for p in ref_preds[:,ti]]
                        ref_bw_open.addEntries(bw_chroms, bw_starts, ends=bw_ends, values=ref_values)

                        alt_values = [float(p) for p in alt_preds[:,ti]]
                        alt_bw_open.addEntries(bw_chroms, bw_starts, ends=bw_ends, values=alt_values)

                        diff_values = [alt_values[vi] - ref_values[vi] for vi in range(len(ref_values))]
                        diff_bw_open.addEntries(bw_chroms, bw_starts, ends=bw_ends, values=diff_values)

                        ref_bw_open.close()
                        alt_bw_open.close()
                        diff_bw_open.close()

                # clean up
                gc.collect()


    sed_gene_out.close()
    if options.transcript_table:
        sed_tx_out.close()

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

    seqs_1hot = np.array(genes_hdf5_in['seqs_1hot'])
    print('genes seqs_1hot', seqs_1hot.shape)

    #######################################
    # transcript_map

    transcripts = [tx.decode('UTF-8') for tx in genes_hdf5_in['transcripts']]
    transcript_index = list(genes_hdf5_in['transcript_index'])
    transcript_pos = list(genes_hdf5_in['transcript_pos'])

    transcript_map = {}
    for ti in range(len(transcripts)):
        transcript_map[transcripts[ti]] = (transcript_index[ti], transcript_pos[ti])

    #######################################
    # transcript_genes

    genes = [gid.decode('UTF-8') for gid in genes_hdf5_in['genes']]

    transcript_genes = {}
    for ti in range(len(transcripts)):
        transcript_genes[transcripts[ti]] = genes[ti]

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

    #######################################
    # multi_seq_genes

    gene_seqs = {}
    for seq_i in range(len(seq_coords)):
        for transcript, tx_pos in seq_transcripts[seq_i]:
            gene = transcript_genes[transcript]
            gene_seqs.setdefault(gene,set()).add(seq_i)

    multi_seq_genes = set()
    for gene in gene_seqs:
        if len(gene_seqs[gene]) > 1:
            multi_seq_genes.add(gene)

    return seq_coords, seqs_1hot, seq_transcripts, transcript_targets, transcript_genes, multi_seq_genes, target_labels



class PredStream:
    ''' Interface to acquire predictions via a buffered stream mechanism
         rather than getting them all at once and using excessive memory.

    Attrs
     sess: TF session to predict within
     model: TF model to predict with
     seqs_1hot (Nx4XL array): one hot coded gene sequences
    ...

    '''

    def __init__(self, sess, model, seqs_1hot, seqs_snps_list, stream_length, rc):
        self.sess = sess
        self.model = model

        self.seqs_1hot = seqs_1hot

        self.seqs_snps_list = seqs_snps_list

        self.stream_length = stream_length
        self.stream_start = 0
        self.stream_end = 0

        if self.stream_length % self.model.batch_size != 0:
            print('Make the stream length a multiple of the batch size', file=sys.stderr)
            exit(1)

        self.rc = rc


    def __getitem__(self, i):
        # acquire predictions, if needed
        if i >= self.stream_end:
            self.stream_start = self.stream_end
            self.stream_end = min(self.stream_start + self.stream_length, len(self.seqs_snps_list))

            # construct sequences
            stream_seqs_1hot = []
            for ssi in range(self.stream_start, self.stream_end):
                seq_i, snp_pos, snp_nt = self.seqs_snps_list[ssi]
                stream_seqs_1hot.append(np.copy(self.seqs_1hot[seq_i]))
                if snp_pos is not None:
                    basenji.dna_io.hot1_set(stream_seqs_1hot[-1], snp_pos, snp_nt)

            stream_seqs_1hot = np.array(stream_seqs_1hot)

            # initialize batcher
            batcher = basenji.batcher.Batcher(stream_seqs_1hot, batch_size=self.model.batch_size)

            # predict
            self.stream_preds = self.model.predict(self.sess, batcher, rc_avg=self.rc)

        return self.stream_preds[i - self.stream_start]


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
    # pdb.runcall(main)
