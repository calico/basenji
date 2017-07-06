#!/usr/bin/env python
from optparse import OptionParser
import os
import sys

import h5py
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

import basenji
import basenji_sat

'''
basenji_sat_vcf.py

Perform an in silico saturated mutagenesis of the sequences surrounding variants
given in a VCF file.
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <params_file> <model_file> <vcf_file>'
    parser = OptionParser(usage)
    parser.add_option('-f', dest='figure_width', default=20, type='float', help='Figure width [Default: %default]')
    parser.add_option('--f1', dest='genome1_fasta', default='%s/assembly/hg19.fa'%os.environ['HG19'], help='Genome FASTA which which major allele sequences will be drawn')
    parser.add_option('--f2', dest='genome2_fasta', default=None, help='Genome FASTA which which minor allele sequences will be drawn')
    parser.add_option('-g', dest='gain', default=False, action='store_true', help='Draw nucleotides proportional to the gain score [Default: %default]')
    parser.add_option('-l', dest='satmut_len', default=200, type='int', help='Length of centered sequence to mutate [Default: %default]')
    parser.add_option('-m', dest='min_limit', default=0.005, type='float', help='Minimum heatmap limit [Default: %default]')
    parser.add_option('-n', dest='load_sat_npy', default=False, action='store_true', help='Load the predictions from .npy files [Default: %default]')
    parser.add_option('-o', dest='out_dir', default='heat', help='Output directory [Default: %default]')
    parser.add_option('-s', dest='seq_len', default=131072, type='int', help='Input sequence length [Default: %default]')
    parser.add_option('-t', dest='targets', default='0', help='Comma-separated list of target indexes to plot (or -1 for all) [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 3:
        parser.error('Must provide parameters and model files and input sequences (as a FASTA file or test data in an HDF file')
    else:
        params_file = args[0]
        model_file = args[1]
        vcf_file = args[2]

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    # decide which targets to obtain
    target_indexes = [int(ti) for ti in options.targets.split(',')]

    #################################################################
    # prep SNP sequences
    #################################################################
    # load SNPs
    snps = basenji.vcf.vcf_snps(vcf_file)

    for si in range(len(snps)):
        print(snps[si])

    # get one hot coded input sequences
    if not options.genome2_fasta:
        seqs_1hot, seq_headers, snps, seqs = basenji.vcf.snps_seq1(snps, options.seq_len, options.genome1_fasta, return_seqs=True)
    else:
        seqs_1hot, seq_headers, snps, seqs = basenji.vcf.snps2_seq1(snps, options.seq_len, options.genome1_fasta, options.genome2_fasta, return_seqs=True)

    seqs_n = seqs_1hot.shape[0]


    #################################################################
    # setup model
    #################################################################
    job = basenji.dna_io.read_job_params(params_file)

    job['batch_length'] = seqs_1hot.shape[1]
    job['seq_depth'] = seqs_1hot.shape[2]

    if 'num_targets' not in job or 'target_pool' not in job:
        print('Must provide num_targets and target_pool in parameters file', file=sys.stderr)
        exit(1)

    # build model
    dr = basenji.seqnn.SeqNN()
    dr.build(job)

    # initialize saver
    saver = tf.train.Saver()


    #################################################################
    # predict and process
    #################################################################

    with tf.Session() as sess:
        # load variables into session
        saver.restore(sess, model_file)

        for si in range(seqs_n):
            header = seq_headers[si]
            header_fs = fs_clean(header)

            print('Mutating sequence %d / %d' % (si+1,seqs_n), flush=True)

            # write sequence
            fasta_out = open('%s/seq%d.fa' % (options.out_dir,si), 'w')
            end_len = (len(seqs[si]) - options.satmut_len) // 2
            print('>seq%d\n%s' % (si,seqs[si][end_len:-end_len]), file=fasta_out)
            fasta_out.close()

            #################################################################
            # predict modifications

            if options.load_sat_npy:
                sat_preds = np.load('%s/seq%d_preds.npy' % (options.out_dir,si))

            else:
                # supplement with saturated mutagenesis
                sat_seqs_1hot = basenji_sat.satmut_seqs(seqs_1hot[si:si+1], options.satmut_len)

                # initialize batcher
                batcher_sat = basenji.batcher.Batcher(sat_seqs_1hot, batch_size=dr.batch_size)

                # predict
                sat_preds = dr.predict(sess, batcher_sat, rc_avg=True, target_indexes=target_indexes)
                np.save('%s/seq%d_preds.npy' % (options.out_dir,si), sat_preds)

            #################################################################
            # compute delta, loss, and gain matrices

            # compute the matrix of prediction deltas: (4 x L_sm x T) array
            sat_delta = basenji_sat.delta_matrix(seqs_1hot[si], sat_preds, options.satmut_len)

            # sat_loss, sat_gain = loss_gain(sat_delta, sat_preds[si], options.satmut_len)
            sat_loss = sat_delta.min(axis=0)
            sat_gain = sat_delta.max(axis=0)

            ##############################################
            # plot

            for ti in range(len(target_indexes)):
                # setup plot
                sns.set(style='white', font_scale=1)
                spp = basenji_sat.subplot_params(sat_delta.shape[1])
                plt.figure(figsize=(options.figure_width,4))
                ax_logo = plt.subplot2grid((3,spp['heat_cols']), (0,spp['logo_start']), colspan=spp['logo_span'])
                ax_sad = plt.subplot2grid((3,spp['heat_cols']), (1,spp['sad_start']), colspan=spp['sad_span'])
                ax_heat = plt.subplot2grid((3,spp['heat_cols']), (2,0), colspan=spp['heat_cols'])

                # plot sequence logo w/ DeepLIFT
                if options.gain:
                    basenji_sat.plot_seqlogo(ax_logo, seqs_1hot[si], -sat_gain[:,ti], -sat_loss[:,ti])
                else:
                    basenji_sat.plot_seqlogo(ax_logo, seqs_1hot[si], sat_loss[:,ti], sat_gain[:,ti])

                # sat_delta_ti_pos = sat_delta[:,:,ti].clip(0,None)
                # sat_loss_4l = basenji_sat.expand_4l(sat_loss[:,ti], seqs_1hot[si])
                # st_freq = basenji_sat.choose_subtick_frequency(options.satmut_len)

                # plot SAD
                basenji_sat.plot_sad(ax_sad, sat_loss[:,ti], sat_gain[:,ti])

                # plot heat map
                basenji_sat.plot_heat(ax_heat, sat_delta[:,:,ti], options.min_limit)

                plt.tight_layout()
                plt.savefig('%s/%s_t%d.pdf' % (options.out_dir,header_fs,ti), dpi=600)
                plt.close()


def fs_clean(header):
    ''' Clean up the headers to valid filenames. '''
    header = header.replace(':','_')
    header = header.replace('>','_')
    return header


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
