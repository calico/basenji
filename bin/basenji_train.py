#!/usr/bin/env python
from optparse import OptionParser
import sys
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
    parser.add_option('-j', '--job', dest='job')
    parser.add_option('-o', '--output', dest='output_file', help='Print accuracy output to file')
    parser.add_option('-r', '--restart', dest='restart', help='Restart training this model')
    parser.add_option('-s', '--save', dest='save_prefix', default='houndrnn')
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


    #######################################################
    # model parameters and placeholders
    #######################################################
    job = basenji.io.read_job_params(options.job)

    job['batch_length'] = train_seqs.shape[1]
    job['seq_depth'] = train_seqs.shape[2]
    job['num_targets'] = train_targets.shape[2]
    job['early_stop'] = job.get('early_stop', 8)
    job['rate_drop'] = job.get('rate_drop', 3)

    t0 = time.time()
    dr = basenji.rnn.RNN()
    dr.build(job)
    print('Model building time %f' % (time.time()-t0))
    sys.stdout.flush()

    #######################################################
    # train
    #######################################################
    # initialize batcher
    batcher_train = basenji.batcher.Batcher(train_seqs, train_targets, dr.batch_size)
    batcher_valid = basenji.batcher.Batcher(valid_seqs, valid_targets, dr.batch_size)

    # checkpoints
    saver = tf.train.Saver()

    with tf.Session() as sess:
        t0 = time.time()

        if options.restart:
            # load variables into session
            saver.restore(sess, options.restart)
        else:
            # initialize variables
            sess.run(tf.initialize_all_variables())
            print("Initialization time %f" % (time.time()-t0))
            sys.stdout.flush()

        train_loss = None
        best_r2 = -1000
        early_stop_i = 0

        for epoch in range(200):
            if early_stop_i < job['early_stop']:
                t0 = time.time()

                # save previous
                train_loss_last = train_loss

                # train
                train_loss = dr.train_epoch(sess, batcher_train)

                # validate
                valid_loss, valid_r2 = dr.test(sess, batcher_valid)

                best_str = ''
                if valid_r2 > best_r2:
                    best_r2 = valid_r2
                    best_str = 'best!'
                    early_stop_i = 0
                    saver.save(sess, '%s_best.tf' % options.save_prefix)
                else:
                    early_stop_i += 1

                # measure time
                et = time.time() - t0
                if et < 600:
                    time_str = '%3ds' % et
                elif et < 6000:
                    time_str = '%3dm' % (et/60)
                else:
                    time_str = '%3.1fh' % (et/3600)

                # print update
                print('Epoch %3d: Train loss: %7.5f, Valid loss: %7.5f, Valid R2: %7.5f, Time: %s %s' % (epoch+1, train_loss, valid_loss, valid_r2, time_str, best_str))
                sys.stdout.flush()

                # if training or validation stagnant
                if train_loss_last is not None and (train_loss_last - train_loss) / train_loss_last < 0.001 or early_stop_i >= job['rate_drop']:
                    print(' Dropping the learning rate.')
                    dr.drop_rate()

                # save the variables to disk.
                # saver.save(sess, '%s_ckpt.tf' % options.save_prefix)

    # print result to file
    if options.output_file:
        output_open = open(options.output_file, 'w')
        print(best_r2, file=output_open)
        output_open.close()


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
