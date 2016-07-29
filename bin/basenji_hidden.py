#!/usr/bin/env python
from __future__ import print_function
from optparse import OptionParser
import os
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
import tensorflow as tf

import basenji

sns.set_style('ticks')

################################################################################
# basenji_hidden.py
#
# Visualize the hidden representations of the test set.
################################################################################

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <params_file> <model_file> <data_file>'
    parser = OptionParser(usage)
    parser.add_option('-b', dest='batch_size', default=None, type='int', help='Batch size')
    parser.add_option('-l', dest='layers', default=None, help='Comma-separated list of layers to plot')
    parser.add_option('-o', dest='out_dir', default='hidden', help='Output directory [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 3:
    	parser.error('Must provide paramters, model, and test data HDF5')
    else:
        params_file = args[0]
        model_file = args[1]
        data_file = args[2]

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    if options.layers is not None:
        options.layers = [int(li) for li in options.layers.split(',')]

    #######################################################
    # load data
    #######################################################
    data_open = h5py.File(data_file)
    test_seqs = data_open['test_in']
    test_targets = data_open['test_out']

    #######################################################
    # model parameters and placeholders
    #######################################################
    job = basenji.dna_io.read_job_params(params_file)

    job['batch_length'] = test_seqs.shape[1]
    job['seq_depth'] = test_seqs.shape[2]
    job['num_targets'] = test_targets.shape[2]

    t0 = time.time()
    dr = basenji.rnn.RNN()
    dr.build(job)

    if options.batch_size is None:
        options.batch_size = dr.batch_size

    #######################################################
    # test
    #######################################################
    # initialize batcher
    batcher_test = basenji.batcher.Batcher(test_seqs, test_targets, options.batch_size)

    # initialize saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # load variables into session
        saver.restore(sess, model_file)

        # get layer representations
        layer_reprs = dr.hidden(sess, batcher_test, options.layers)

        for li in range(len(layer_reprs)):
            layer_repr = layer_reprs[li]
            print(layer_repr.shape)

            # sample one nt per sequence
            nt_reprs = layer_repr[:,job['batch_length']//2,:]

            ########################################################
            # plot raw
            plt.figure()
            sns.clustermap(nt_reprs)
            plt.savefig('%s/l%d_reprs.pdf' % (options.out_dir,li))
            plt.close()

            ########################################################
            # visualize w/ TSNE
            model = TSNE()
            nt_2d = model.fit_transform(nt_reprs)

            for ti in range(job['num_targets']):
                nt_targets = test_targets[:,job['batch_length']//2,ti]

                plt.figure()
                plt.scatter(nt_2d[:,0], nt_2d[:,1], alpha=0.5, c=nt_targets, cmap='RdBu_r')
                plt.colorbar()
                ax = plt.gca()
                ax.grid(True, linestyle=':')
                plt.savefig('%s/l%d_nt2d_t%d.pdf' % (options.out_dir,li,ti))
                plt.close()

            ########################################################
            # plot neuron-neuron correlations
            neuron_cors = np.corrcoef(nt_reprs.T)
            plt.figure()
            sns.clustermap(neuron_cors)
            plt.savefig('%s/l%d_cor.pdf' % (options.out_dir,li))
            plt.close()

            ########################################################
            # plot neuron densities
            neuron_stats_out = open('%s/l%d_stats.txt' % (options.out_dir,li), 'w')

            for ni in range(nt_reprs.shape[1]):
                # print stats
                nu = nt_reprs[:,ni].mean()
                nstd = nt_reprs[:,ni].std()
                print('%3d  %6.3f  %6.3f' % (ni,nu,nstd), file=neuron_stats_out)

                # plot
                plt.figure()
                sns.distplot(nt_reprs[:,ni])
                plt.savefig('%s/l%d_dist%d.pdf' % (options.out_dir,li,ni))
                plt.close()

            neuron_stats_out.close()

    data_open.close()


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
