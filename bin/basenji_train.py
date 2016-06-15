#!/usr/bin/env python
from optparse import OptionParser
import time

import h5py
import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow as tf

import basenji

################################################################################
# basenji_train.py
#
#
################################################################################


################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <data_file>'
    parser = OptionParser(usage)
    parser.add_option('-s', '--save', dest='save_prefix', default='rott')
    (options,args) = parser.parse_args()

    if len(args) != 1:
    	parser.error('Must provide data file')
    else:
    	data_file = args[0]

    #######################################################
    # load data
    #######################################################
    data_open = h5py.File(data_file)
    train_seqs = data_open['train_in']
    train_targets = data_open['train_out']
    valid_seqs = data_open['valid_in']
    valid_targets = data_open['valid_out']

    seq_len = train_seqs.shape[0]
    seq_depth = train_seqs.shape[1]
    num_targets = train_targets.shape[1]


    #######################################################
    # model parameters and placeholders
    #######################################################
    job = {}
    job['batch_size'] = 32
    job['batch_length'] = 128
    job['seq_depth'] = 4
    job['num_targets'] = 1
    job['hidden_units'] = [30,30]
    job['cell'] = 'lstm'

    dr = basenji.rnn.RNN()
    dr.build(job)

    #######################################################
    # train
    #######################################################
    # initialize batcher
    batcher_train = basenji.batcher.Batcher(train_seqs, train_targets, dr.batch_size, dr.batch_length)
    batcher_valid = basenji.batcher.Batcher(valid_seqs, valid_targets, dr.batch_size, dr.batch_length)

    # checkpoints
    saver = tf.train.Saver()

    with tf.Session() as sess:
        t0 = time.time()

        # initialize variables
        sess.run(tf.initialize_all_variables())
        print("Initialization time %f" % (time.time()-t0))

        best_r2 = -1000

        for epoch in range(1, 101):
            t0 = time.time()

            # train
            train_loss = dr.train_epoch(sess, batcher_train)

            # validate
            valid_loss, valid_r2 = dr.test(sess, batcher_valid)

            best_str = ''
            if valid_r2 > best_r2:
                best_r2 = valid_r2
                best_str = 'best!'

            # measure time
            et = time.time() - t0

            # print update
            print('Epoch %3d: Train loss: %11.4f, Valid loss: %11.4f, Valid R2: %7.5f, Time: %5d %s' % (epoch, train_loss, valid_loss, valid_r2, et, best_str))

            # Save the variables to disk.
            # saver.save(sess, '%s.ckpt' % options.save_prefix)


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
