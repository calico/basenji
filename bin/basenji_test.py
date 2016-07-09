#!/usr/bin/env python
from __future__ import print_function
from optparse import OptionParser
import time

import h5py
import numpy as np
from sklearn.linear_model import LinearRegression
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
    parser.add_option('-p', dest='peaks_hdf5', help='Compute AUC for sequence peak calls [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 3:
    	parser.error('Must provide paramters, model, and test data HDF5')
    else:
        params_file = args[0]
        model_file = args[1]
        data_file = args[2]

    #######################################################
    # load data
    #######################################################
    data_open = h5py.File(data_file)
    test_seqs = data_open['test_in']
    test_targets = data_open['test_out']

    #######################################################
    # model parameters and placeholders
    #######################################################
    job = basenji.io.read_job_params(params_file)

    job['batch_length'] = test_seqs.shape[1]
    job['seq_depth'] = test_seqs.shape[2]
    job['num_targets'] = test_targets.shape[2]

    t0 = time.time()
    dr = basenji.rnn.RNN()
    dr.build(job)
    print('Model building time %f' % (time.time()-t0))

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

        # test
        test_loss, test_r2, test_preds = dr.test(sess, batcher_test, return_preds=True)

        # print
        print('Test loss: %7.5f' % test_loss)
        print('Test R2:   %7.5f' % test_r2)

        #######################################################
        # peaks AUC
        #######################################################
        if options.peaks_hdf5:
            # use validation set to train
            valid_seqs = data_open['valid_in']
            valid_targets = data_open['valid_out']

            # initialize batcher
            batcher_valid = basenji.batcher.Batcher(valid_seqs, valid_targets, options.batch_size)

            # make predictions
            _, _, valid_preds = dr.test(sess, batcher_valid, return_preds=True)
            valid_n = valid_preds.shape[0]

            print(valid_preds.shape)

            # load peaks
            peaks_open = h5py.File(options.peaks_hdf5)
            valid_peaks = np.array(peaks_open['valid_out'])
            test_peaks = np.array(peaks_open['test_out'])
            peaks_open.close()

            # compute target AUCs
            target_aucs = []
            for ti in range(test_peaks.shape[1]):
                # train a predictor for peak calls
                model = LinearRegression()
                model.fit(valid_preds[:,:,ti], valid_peaks[:valid_n,ti])

                # predict peaks for test set
                test_peaks_preds = model.predict(test_preds[:,:,ti])
                test_n = test_peaks_preds.shape[0]

                # compute AUC
                auc = roc_auc_score(test_peaks[:test_n,ti], test_peaks_preds)
                target_aucs.append(auc)
                print('%d AUC: %f' % (ti,auc))

            print('AUC: %f' % np.mean(target_aucs))

    data_open.close()


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
