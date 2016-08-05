#!/usr/bin/env python
from optparse import OptionParser
import sys
import time

import h5py
import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow as tf

import basenji.dna_io
import basenji.batcher
import basenji.rnn

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
    parser.add_option('-d', dest='down_sample', default=1, type='int', help='Down sample test computation by taking uniformly spaced positions [Default: %default]')
    parser.add_option('-m', dest='params_file', help='Model parameters')
    parser.add_option('-o', dest='output_file', help='Print accuracy output to file')
    parser.add_option('-r', dest='restart', help='Restart training this model')
    parser.add_option('-s', dest='save_prefix', default='houndrnn')
    parser.add_option('-u', dest='summary', default=None, help='TensorBoard summary directory')
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
    job = basenji.dna_io.read_job_params(options.params_file)

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
    batcher_train = basenji.batcher.Batcher(train_seqs, train_targets, dr.batch_size, shuffle=True)
    batcher_valid = basenji.batcher.Batcher(valid_seqs, valid_targets, dr.batch_size)

    # checkpoints
    saver = tf.train.Saver()

    with tf.Session() as sess:
        t0 = time.time()

        if options.summary is None:
            train_writer = None
        else:
            train_writer = tf.train.SummaryWriter(options.summary + '/train', sess.graph)

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

        for epoch in range(1000):
            if early_stop_i < job['early_stop']:
                t0 = time.time()

                # save previous
                train_loss_last = train_loss

                # train
                train_loss = dr.train_epoch(sess, batcher_train, train_writer)

                # validate
                valid_loss, valid_r2 = dr.test(sess, batcher_valid, down_sample=options.down_sample)

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

                # if training stagnant
                # if train_loss_last is not None and train_loss > train_loss_last:
                #     print(' Dropping the learning rate.')
                #     dr.drop_rate()

        if options.summary is not None:
            train_writer.close()

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
