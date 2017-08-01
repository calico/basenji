#!/usr/bin/env python
# Copyright 2017 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from __future__ import print_function
from optparse import OptionParser
import os
import time

import h5py

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import tensorflow as tf

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import seaborn as sns

import basenji

'''
basenji_predict_var.py

Make predictions and assess variance.
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <params_file> <model_file> <test_hdf5_file>'
    parser = OptionParser(usage)
    parser.add_option('-d', dest='down_sample', default=1, type='int', help='Down sample test computation by taking uniformly spaced positions [Default: %default]')
    parser.add_option('-m', dest='mc_n', default=0, type='int', help='Monte carlo iterations [Default: %default]')
    parser.add_option('-o', dest='out_dir', default='predict_var', help='Output directory for test statistics [Default: %default]')
    parser.add_option('-s', dest='save', default=False, action='store_true', help='Save predictions and variance arrays [Default: %default]')
    parser.add_option('--rc', dest='rc', default=False, action='store_true', help='Average the forward and reverse complement predictions when testing [Default: %default]')
    parser.add_option('-t', dest='targets', default=None, help='Comma-separated list of target indexes to plot (or -1 for all) [Default: %default]')
    parser.add_option('-v', dest='valid', default=False, action='store_true', help='Process the validation set [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 3:
    	parser.error('Must provide parameters, model, and test data HDF5')
    else:
        params_file = args[0]
        model_file = args[1]
        test_hdf5_file = args[2]

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    #######################################################
    # load data
    #######################################################
    data_open = h5py.File(test_hdf5_file)

    if not options.valid:
        test_seqs = data_open['test_in']
        test_targets = data_open['test_out']
        test_na = None
        if 'test_na' in data_open:
            test_na = data_open['test_na']

    else:
        test_seqs = data_open['valid_in']
        test_targets = data_open['valid_out']
        test_na = None
        if 'test_na' in data_open:
            test_na = data_open['valid_na']

    target_labels = [tl.decode('UTF-8') for tl in data_open['target_labels']]

    # limit targets
    if options.targets is None:
        target_indexes = np.arange(len(target_labels))
    else:
        target_indexes = np.array([int(ti) for ti in options.targets.split(',')])

    #######################################################
    # model parameters and placeholders

    job = basenji.dna_io.read_job_params(params_file)

    job['batch_length'] = test_seqs.shape[1]
    job['seq_depth'] = test_seqs.shape[2]
    job['num_targets'] = test_targets.shape[2]
    job['target_pool'] = int(np.array(data_open.get('pool_width', 1)))

    t0 = time.time()
    model = basenji.seqnn.SeqNN()
    model.build(job)
    print('Model building time %ds' % (time.time()-t0))

    # adjust for fourier
    job['fourier'] = 'train_out_imag' in data_open
    if job['fourier']:
        test_targets_imag = data_open['test_out_imag']
        if options.valid:
            test_targets_imag = data_open['valid_out_imag']

    #######################################################
    # predict

    # initialize batcher
    if job['fourier']:
        batcher_test = basenji.batcher.BatcherF(test_seqs, test_targets, test_targets_imag, test_na, model.batch_size, model.target_pool)
    else:
        batcher_test = basenji.batcher.Batcher(test_seqs, test_targets, test_na, model.batch_size, model.target_pool)

    # initialize saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # load variables into session
        saver.restore(sess, model_file)

        # predict
        # preds, preds_var = model.predict(sess, batcher_test, rc_avg=options.rc, mc_n=options.mc_n, return_var=True, target_indexes=target_indexes, down_sample=options.down_sample)
        preds, preds_var, preds_all = model.predict(sess, batcher_test, rc_avg=options.rc, mc_n=options.mc_n, return_var=True, return_all=True, target_indexes=target_indexes, down_sample=options.down_sample)

        if options.save:
            np.save('%s/preds.npy'%options.out_dir, preds)
            np.save('%s/preds_var.npy'%options.out_dir, preds_var)
            np.save('%s/preds_all.npy'%options.out_dir, preds_all)

    #######################################################
    # plot

    # plot prediction versus variance
    for tii in range(len(target_indexes)):
        preds_ti = preds[:,:,tii].flatten().astype('float32')
        preds_var_ti = preds_var[:,:,tii].flatten().astype('float32')
        out_pdf = '%s/t%d.pdf' % (options.out_dir, tii)

        cor, p = spearmanr(preds_ti, preds_var_ti)
        print('%4d  %.4f  %.4f' % (target_indexes[tii], cor, p))

        basenji.plots.jointplot(preds_ti, preds_var_ti, out_pdf, x_label='Prediction', y_label='Variance', figsize=(7,7), sample=2000)

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
