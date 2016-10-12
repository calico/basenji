#!/usr/bin/env python
from optparse import OptionParser
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pysam
import seaborn as sns
import tensorflow as tf

import basenji.dna_io
import basenji.vcf

################################################################################
# basenji_sad.py
#
# Compute SNP Accessibility Difference (SAD) scores for SNPs in a VCF file.
################################################################################

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <params_file> <model_file> <vcf_file>'
    parser = OptionParser(usage)
    parser.add_option('-b', dest='batch_size', default=None, type='int', help='Batch size [Default: %default]')
    parser.add_option('-c', dest='csv', default=False, action='store_true', help='Print table as CSV [Default: %default]')
    parser.add_option('-e', dest='heatmaps', default=False, action='store_true', help='Draw score heatmaps, grouped by index SNP [Default: %default]')
    parser.add_option('-f', dest='genome_fasta', default='%s/data/genomes/hg19.fa'%os.environ['BASSETDIR'], help='Genome FASTA from which sequences will be drawn [Default: %default]')
    parser.add_option('-i', dest='index_snp', default=False, action='store_true', help='SNPs are labeled with their index SNP as column 6 [Default: %default]')
    parser.add_option('-l', dest='seq_len', type='int', default=1024, help='Sequence length provided to the model [Default: %default]')
    parser.add_option('-m', dest='min_limit', default=0.1, type='float', help='Minimum heatmap limit [Default: %default]')
    parser.add_option('-o', dest='out_dir', default='sad', help='Output directory for tables and plots [Default: %default]')
    parser.add_option('-s', dest='score', default=False, action='store_true', help='SNPs are labeled with scores as column 7 [Default: %default]')
    parser.add_option('-t', dest='targets_file', default=None, help='File specifying target indexes and labels in table format')
    (options,args) = parser.parse_args()

    if len(args) != 3:
        parser.error('Must provide parameters and model files and QTL VCF file')
    else:
        params_file = args[0]
        model_file = args[1]
        vcf_file = args[2]

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    #################################################################
    # setup model

    job = basenji.dna_io.read_job_params(params_file)
    job['batch_length'] = options.seq_len

    if 'num_targets' not in job:
        print("Must specify number of targets (num_targets) in the parameters file. I know, it's annoying. Sorry.", file=sys.stderr)
        exit(1)

    t0 = time.time()
    dr = basenji.rnn.RNN()
    dr.build(job)
    print('Model building time %f' % (time.time()-t0))

    if options.batch_size is not None:
        dr.batch_size = options.batch_size

    # initialize saver
    saver = tf.train.Saver()

    #################################################################
    # load SNPs

    snps = basenji.vcf.vcf_snps(vcf_file, options.index_snp, options.score)


    #################################################################
    # setup output

    if options.targets_file is None:
        target_labels = ['t%d' % ti for ti in range(job['num_targets'])]
    else:
        target_labels = [line.split()[0] for line in open(options.targets_file)]

    header_cols = ('rsid', 'index', 'score', 'ref', 'alt', 'target', 'ref_pred', 'alt pred', 'sad')
    if options.csv:
        sad_out = open('%s/sad_table.csv' % options.out_dir, 'w')
        print(','.join(header_cols), file=sad_out)
    else:
        sad_out = open('%s/sad_table.txt' % options.out_dir, 'w')
        print(' '.join(header_cols), file=sad_out)

    # hash by index snp
    sad_matrices = {}
    sad_labels = {}
    sad_scores = {}

    #################################################################
    # process

    # open genome FASTA
    genome_open = pysam.Fastafile(options.genome_fasta)

    snp_i = 0

    with tf.Session() as sess:
        # load variables into session
        saver.restore(sess, model_file)

        # construct first batch
        batch_1hot, batch_snps, snp_i = snps_next_batch(snps, snp_i, dr.batch_size, options.seq_len, genome_open)

        while len(batch_snps) > 0:
            ###################################################
            # predict

            # initialize batcher
            batcher = basenji.batcher.Batcher(batch_1hot, batch_size=dr.batch_size)

            # predict
            batch_preds = dr.predict(sess, batcher)

            ###################################################
            # collect and print SADs

            pi = 0
            for snp in batch_snps:
                # get reference prediction (LxT)
                ref_preds = batch_preds[pi]
                pi += 1

                for alt_al in snp.alt_alleles:
                    # get alternate prediction (LxT)
                    alt_preds = batch_preds[pi]
                    pi += 1

                    # normalize by reference and mean across length
                    alt_sad = (alt_preds - ref_preds).mean(axis=0)
                    sad_matrices.setdefault(snp.index_snp,[]).append(alt_sad)

                    # label as mutation from reference
                    alt_label = '%s_%s>%s' % (snp.rsid, basenji.vcf.cap_allele(snp.ref_allele), basenji.vcf.cap_allele(alt_al))
                    sad_labels.setdefault(snp.index_snp,[]).append(alt_label)

                    # save scores
                    sad_scores.setdefault(snp.index_snp,[]).append(snp.score)

                    # print table lines
                    for ti in range(len(alt_sad)):
                        # set index SNP
                        snp_is = '%-13s' % '.'
                        if options.index_snp:
                            snp_is = '%-13s' % snp.index_snp

                        # set score
                        snp_score = '%5s' % '.'
                        if options.score:
                            snp_score = '%5.3f' % snp.score

                        # print line
                        cols = (snp.rsid, snp_is, snp_score, basenji.vcf.cap_allele(snp.ref_allele), basenji.vcf.cap_allele(alt_al), target_labels[ti], ref_preds[ti].mean(), alt_preds[ti].mean(), alt_sad[ti])
                        if options.csv:
                            print(','.join([str(c) for c in cols]), file=sad_out)
                        else:
                            print('%-13s %s %5s %6s %6s %12s %6.4f %6.4f %7.4f' % cols, file=sad_out)

            ###################################################
            # construct next batch

            batch_1hot, batch_snps, snp_i = snps_next_batch(snps, snp_i, dr.batch_size, options.seq_len, genome_open)


    sad_out.close()

    #################################################################
    # plot SAD heatmaps
    #################################################################
    if options.heatmaps:
        for ii in sad_matrices:
            # convert fully to numpy arrays
            sad_matrix = abs(np.array(sad_matrices[ii]))
            print(ii, sad_matrix.shape)

            if sad_matrix.shape[0] > 1:
                vlim = max(options.min_limit, sad_matrix.max())
                score_mat = np.reshape(np.array(sad_scores[ii]), (-1, 1))

                # plot heatmap
                plt.figure(figsize=(20, 0.5 + 0.5*sad_matrix.shape[0]))

                if options.score:
                    # lay out scores
                    cols = 12
                    ax_score = plt.subplot2grid((1,cols), (0,0))
                    ax_sad = plt.subplot2grid((1,cols), (0,1), colspan=(cols-1))

                    sns.heatmap(score_mat, xticklabels=False, yticklabels=False, vmin=0, vmax=1, cmap='Reds', cbar=False, ax=ax_score)
                else:
                    ax_sad = plt.gca()

                sns.heatmap(sad_matrix, xticklabels=target_labels, yticklabels=sad_labels[ii], vmin=0, vmax=vlim, ax=ax_sad)

                for tick in ax_sad.get_xticklabels():
                    tick.set_rotation(-45)
                    tick.set_horizontalalignment('left')
                    tick.set_fontsize(5)

                plt.tight_layout()
                if ii == '.':
                    out_pdf = '%s/sad_heat.pdf' % options.out_dir
                else:
                    out_pdf = '%s/sad_%s_heat.pdf' % (options.out_dir, ii)
                plt.savefig(out_pdf)
                plt.close()


def snps_next_batch(snps, snp_i, batch_size, seq_len, genome_open):
    ''' Load the next batch of SNP sequence 1-hot. '''

    batch_1hot = []
    batch_snps = []

    while len(batch_1hot) < batch_size and snp_i < len(snps):
        # get SNP sequences
        snp_1hot = basenji.vcf.snp_seq1(snps[snp_i], seq_len, genome_open)

        # if it was valid
        if len(snp_1hot) > 0:
            # accumulate
            batch_1hot += snp_1hot
            batch_snps.append(snps[snp_i])

        # advance SNP index
        snp_i += 1

    # convert to array
    batch_1hot = np.array(batch_1hot)

    return batch_1hot, batch_snps, snp_i

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
