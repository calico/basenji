#!/usr/bin/env python
from __future__ import print_function
from optparse import OptionParser
import os
import random
import sys
import time

import h5py
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
import tensorflow as tf

import basenji

################################################################################
# basenji_test.py
#
#
################################################################################

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <params_file> <model_file> <data_file>'
    parser = OptionParser(usage)
    parser.add_option('-b', dest='batch_size', default=None, type='int', help='Batch size')
    parser.add_option('-d', dest='down_sample', default=1, type='int', help='Down sample test computation by taking uniformly spaced positions [Default: %default]')
    parser.add_option('-o', dest='out_dir', default='test_out', help='Output directory for test statistics [Default: %default]')
    parser.add_option('-p', dest='peaks_hdf5', help='Compute AUC for sequence peak calls [Default: %default]')
    parser.add_option('-s', dest='scent_file', help='Dimension reduction model file')
    parser.add_option('-v', dest='valid', default=False, action='store_true', help='Process the validation set [Default: %default]')
    parser.add_option('-w', dest='pool_width', default=1, type='int', help='Max pool width for regressing nucleotide predictions to predict peak calls [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 3:
    	parser.error('Must provide paramters, model, and test data HDF5')
    else:
        params_file = args[0]
        model_file = args[1]
        data_file = args[2]

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    #######################################################
    # load data
    #######################################################
    data_open = h5py.File(data_file)

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
            test_na = data_open['test_na']


    #######################################################
    # model parameters and placeholders
    #######################################################
    job = basenji.dna_io.read_job_params(params_file)

    job['batch_length'] = test_seqs.shape[1]
    job['seq_depth'] = test_seqs.shape[2]
    job['num_targets'] = test_targets.shape[2]
    job['target_pool'] = int(np.array(data_open.get('pool_width', 1)))

    t0 = time.time()
    dr = basenji.rnn.RNN()
    dr.build(job)
    print('Model building time %ds' % (time.time()-t0))

    if options.batch_size is None:
        options.batch_size = dr.batch_size

    # adjust for fourier
    job['fourier'] = 'train_out_imag' in data_open
    if job['fourier']:
        test_targets_imag = data_open['test_out_imag']
        if options.valid:
            test_targets_imag = data_open['valid_out_imag']

    # adjust for factors
    if options.scent_file is not None:
        t0 = time.time()
        test_targets_full = data_open['test_out_full']
        model = joblib.load(options.scent_file)

    #######################################################
    # test
    #######################################################
    # initialize batcher
    if job['fourier']:
        batcher_test = basenji.batcher.BatcherF(test_seqs, test_targets, test_targets_imag, test_na, options.batch_size, job['target_pool'])
    else:
        batcher_test = basenji.batcher.Batcher(test_seqs, test_targets, test_na, options.batch_size, job['target_pool'])

    # initialize saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # load variables into session
        saver.restore(sess, model_file)

        # test
        t0 = time.time()
        test_loss, test_r2_list, test_preds = dr.test(sess, batcher_test, return_preds=True, down_sample=options.down_sample)
        print('RNN test: %ds' % (time.time()-t0))

        # print
        print('Test loss: %7.5f' % test_loss)
        print('Test R2:   %7.5f' % test_r2_list.mean())

        r2_out = open('%s/r2.txt' % options.out_dir, 'w')
        for ti in range(len(test_r2_list)):
            print('%4d  %.4f' % (ti,test_r2_list[ti]), file=r2_out)
        r2_out.close()

        if options.scent_file is not None:
            full_targets = test_targets_full.shape[2]

            # determine non-buffer region
            buf_start = dr.batch_buffer // dr.target_pool
            buf_end = (dr.batch_length - dr.batch_buffer) // dr.target_pool
            buf_len = buf_end - buf_start

            # uniformly sample indexes
            ds_indexes = np.arange(0, buf_len, options.down_sample)

            # filter down full test targets
            test_targets_full_ds = test_targets_full[:,buf_start+ds_indexes,:]
            test_na_ds = test_na[:,buf_start+ds_indexes]

            # inverse transform in length batches
            t0 = time.time()
            test_preds_full = np.zeros((test_preds.shape[0], test_preds.shape[1], full_targets), dtype='float16')
            for li in range(test_preds.shape[1]):
                test_preds_full[:,li,:] = model.inverse_transform(test_preds[:,li,:])
            print('PCA transform: %ds' % (time.time()-t0))

            print(test_preds_full.shape)
            print(test_targets_full_ds.shape)

            # compute R2 by target
            t0 = time.time()
            test_r2_full = np.zeros(full_targets)
            for ti in range(full_targets):
                # flatten
                # preds_ti = test_preds_full[:,:,ti].flatten()
                # targets_ti = test_targets_full_ds[:,:,ti].flatten()
                preds_ti = test_preds_full[test_na_ds,ti]
                targets_ti = test_targets_full_ds[test_na_ds,ti]

                # compute R2
                tmean = targets_ti.mean(dtype='float64')
                tvar = (targets_ti-tmean).var(dtype='float64')
                pvar = (targets_ti-preds_ti).var(dtype='float64')
                test_r2_full[ti] = 1.0 - pvar/tvar
            print('Compute full R2: %d' % (time.time()-t0))

            print('Test full R2: %7.5f' % test_r2_full.mean())

            r2_out = open('%s/r2_full.txt' % options.out_dir, 'w')
            for ti in range(len(test_r2_full)):
                print('%4d  %.4f' % (ti,test_r2_full[ti]), file=r2_out)
            r2_out.close()

        #######################################################
        # peaks AUC
        #######################################################
        if options.peaks_hdf5:
            # use validation set to train
            valid_seqs = data_open['valid_in']
            valid_targets = data_open['valid_out']

            # initialize batcher
            if job['fourier']:
                valid_targets_imag = data_open['valid_out_imag']
                batcher_valid = basenji.batcher.BatcherF(valid_seqs, valid_targets, valid_targets_imag, options.batch_size)
            else:
                batcher_valid = basenji.batcher.Batcher(valid_seqs, valid_targets, options.batch_size)

            # make predictions
            _, _, valid_preds = dr.test(sess, batcher_valid, return_preds=True, down_sample=options.down_sample)

            print(valid_preds.shape)

            # max pool
            if options.pool_width > 1:
                valid_preds = max_pool(valid_preds, options.pool_width)
                test_preds = max_pool(test_preds, options.pool_width)
                print(valid_preds.shape)

            # load peaks
            peaks_open = h5py.File(options.peaks_hdf5)
            valid_peaks = np.array(peaks_open['valid_out'])
            test_peaks = np.array(peaks_open['test_out'])
            peaks_open.close()

            # compute target AUCs
            target_aucs = []
            model_coefs = []
            for ti in range(test_peaks.shape[1]):
                # balance positive and negative examples
                # valid_preds_ti, valid_peaks_ti = balance(valid_preds, valid_peaks[:,ti])
                valid_preds_ti, valid_peaks_ti = valid_preds, valid_peaks[:,ti]

                # train a predictor for peak calls
                model = LogisticRegression()
                if options.scent_file is not None:
                    valid_preds_ti_flat = valid_preds_ti.reshape((valid_preds_ti.shape[0],-1))
                    model.fit(valid_preds_ti_flat, valid_peaks_ti)
                else:
                    model.fit(valid_preds_ti[:,:,ti], valid_peaks_ti)
                model_coefs.append(model.coef_)


                # predict peaks for test set
                if options.scent_file is not None:
                    test_preds_flat = test_preds.reshape((test_preds.shape[0],-1))
                    test_peaks_preds = model.predict_proba(test_preds_flat)[:,1]
                else:
                    test_peaks_preds = model.predict_proba(test_preds[:,:,ti])[:,1]

                # compute AUC
                auc = roc_auc_score(test_peaks[:,ti], test_peaks_preds)
                target_aucs.append(auc)
                print('%d AUC: %f' % (ti,auc))

            print('AUC: %f' % np.mean(target_aucs))

            model_coefs = np.vstack(model_coefs)
            np.save('%s/coefs.npy' % options.out_dir, model_coefs)

    data_open.close()

def balance(X, Y):
    # determine positive and negative indexes
    indexes_pos = [i for i in range(len(Y)) if Y[i] == 1]
    indexes_neg = [i for i in range(len(Y)) if Y[i] == 0]

    # sample down negative
    indexes_neg_sample = random.sample(indexes_neg, len(indexes_pos))

    # combine
    indexes = sorted(indexes_pos + indexes_neg_sample)

    return X[indexes], Y[indexes]


def max_pool(preds, pool):
    # group by pool
    preds_pool = preds.reshape((preds.shape[0], preds.shape[1]//pool, pool, preds.shape[2]), order='C')

    # max
    return preds_pool.max(axis=2)


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
