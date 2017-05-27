#!/usr/bin/env python
from __future__ import print_function
from optparse import OptionParser
import os
import time

import hail
import tensorflow as tf

import basenji

'''
basenji_genotype_genes.py

Description...
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <params_file> <model_file> <genes_hdf5_file> <vds_file>'
    parser = OptionParser(usage)
    parser.add_option('-o', dest='out_dir', default='geno', help='Output directory for tables and plots [Default: %default]')
    parser.add_option('-p', dest='processes', default=None, type='int', help='Number of processes, passed by multi script')
    parser.add_option('--rc', dest='rc', default=False, action='store_true', help='Average the forward and reverse complement predictions when testing [Default: %default]')
    parser.add_option('-t', dest='target_indexes', help='Comma-separated list of target indexes to make predictions for.')
    (options,args) = parser.parse_args()

    # add an option to limit the targets studied

    if len(args) == 4:
        # single worker
        params_file = args[0]
        model_file = args[1]
        genes_hdf5_file = args[2]
        vds_file = args[3]

    else:
        parser.error('Must provide parameter and model files, genes HDF5 file, and genotypes VDS')

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    if options.target_indexes is not None:
        options.target_indexes = [int(ti) for ti in options.target_indexes.split(',')]

    #################################################################
    # reads in genes HDF5

    gene_data = basenji.genes.GeneData(genes_hdf5_file)

    # filter for worker sequences
    if options.processes is not None:
        gene_data.worker(worker_index, options.processes)

    #################################################################
    # setup model

    job = basenji.dna_io.read_job_params(params_file)

    job['batch_length'] = gene_data.seq_length
    job['seq_depth'] = gene_data.seq_depth
    job['target_pool'] = gene_data.pool_width

    if 'num_targets' not in job and gene_data.num_targets is not None:
        job['num_targets'] = gene_data.num_targets
    if 'num_targets' not in job:
        print("Must specify number of targets (num_targets) in the parameters file. I know, it's annoying. Sorry.", file=sys.stderr)
        exit(1)

    # build model
    model = basenji.rnn.RNN()
    model.build(job)


    #################################################################
    # process gene sequences

    # initialize saver
    saver = tf.train.Saver()

    # initialize VDS
    hc = hail.HailContext()
    vds_open = hc.read(vds_file)

    with tf.Session() as sess:
        # load variables into session
        saver.restore(sess, model_file)

        for gene_seq in gene_data.gene_seqs():

            # load genotypes
            gene_seq.genotypes(vds_open)

            # form a non redundant set of haplotypes
            gene_seq.haplotypes()

            # 1-hot code haplotypes
            haps_1hot = gene_seq.haplotypes_1hot()

            # initialize batcher
            batcher = basenji.batcher.Batcher(haps_1hot, batch_size=model.batch_size)

            # predict
            haps_preds = model.predict(sess, batcher, options.rc, options.target_indexes)

            # map haplotype predictions to sample transcript predictions
            sample_preds = gene_seq.gene_preds(haps_preds)

            # Returning a DataFrame here isn't going to work; it'll just be
            # too large. Instead, I should probably fill in a matrix of
            # individuals...

            # Well, idk. I'm not going to keep the whole damn thing in memory,
            # right? I can spit it out as we go. So the primary reason not to
            # use a DataFrame is that it will write the same damn gene name
            # over and over again.

            # Maybe save the results as an HDF5, with gene_preds, transcript_preds, and sample and target information.


            # write?
            # sample_preds.to_csv('%s/transcript_preds.txt'%options.out_dir, sep='\t', index=False)

            # map transcript predictions to gene predictions
            # sample_gene_preds = ''


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
