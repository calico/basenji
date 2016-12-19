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
    parser.add_option('-l', dest='learn_rate_drop', default=False, action='store_true', help='Drop learning rate when training loss stalls [Default: %default]')
    parser.add_option('-m', dest='params_file', help='Model parameters')
    parser.add_option('-o', dest='output_file', help='Print accuracy output to file')
    parser.add_option('-r', dest='restart', help='Restart training this model')
    parser.add_option('--rc', dest='rc', default=False, action='store_true', help='Average the forward and reverse complement predictions when testing [Default: %default]')
    parser.add_option('-s', dest='save_prefix', default='houndrnn')
    parser.add_option('--seed', dest='seed', type='float', default=1, help='RNG seed')
    parser.add_option('-u', dest='summary', default=None, help='TensorBoard summary directory')
    (options,args) = parser.parse_args()

    if len(args) != 1:
    	parser.error('Must provide data file')
    else:
    	data_file = args[0]

    np.random.seed(options.seed)

    #######################################################
    # load data
    #######################################################
    data_open = h5py.File(data_file)

    train_seqs = data_open['train_in']
    train_targets = data_open['train_out']
    train_na = None
    if 'train_na' in data_open:
        train_na = data_open['train_na']

    valid_seqs = data_open['valid_in']
    valid_targets = data_open['valid_out']
    valid_na = None
    if 'valid_na' in data_open:
        valid_na = data_open['valid_na']


    #######################################################
    # model parameters and placeholders
    #######################################################
    job = basenji.dna_io.read_job_params(options.params_file)

    job['batch_length'] = train_seqs.shape[1]
    job['seq_depth'] = train_seqs.shape[2]
    job['num_targets'] = train_targets.shape[2]
    job['target_pool'] = int(np.array(data_open.get('pool_width', 1)))
    job['early_stop'] = job.get('early_stop', 12)
    job['rate_drop'] = job.get('rate_drop', 3)

    t0 = time.time()
    dr = basenji.rnn.RNN()
    dr.build(job)
    print('Model building time %f' % (time.time()-t0))
    sys.stdout.flush()

    # adjust for fourier
    job['fourier'] = 'train_out_imag' in data_open
    if job['fourier']:
        train_targets_imag = data_open['train_out_imag']
        valid_targets_imag = data_open['valid_out_imag']


    #######################################################
    # train
    #######################################################
    # initialize batcher
    if job['fourier']:
        batcher_train = basenji.batcher.BatcherF(train_seqs, train_targets, train_targets_imag, train_na, dr.batch_size, dr.target_pool, shuffle=True)
        batcher_valid = basenji.batcher.BatcherF(valid_seqs, valid_targets, valid_targets_imag, valid_na, dr.batch_size, dr.target_pool)
    else:
        batcher_train = basenji.batcher.Batcher(train_seqs, train_targets, train_na, dr.batch_size, dr.target_pool, shuffle=True)
        batcher_valid = basenji.batcher.Batcher(valid_seqs, valid_targets, valid_na, dr.batch_size, dr.target_pool)
    print('Batcher initialized')
    sys.stdout.flush()


    # checkpoints
    saver = tf.train.Saver()

    with tf.Session() as sess:
        t0 = time.time()

        # set seed
        tf.set_random_seed(options.seed)

        if options.summary is None:
            train_writer = None
        else:
            train_writer = tf.train.SummaryWriter(options.summary + '/train', sess.graph)

        if options.restart:
            # load variables into session
            saver.restore(sess, options.restart)
        else:
            # initialize variables
            print('Initializing...')
            sys.stdout.flush()
            # sess.run(tf.initialize_all_variables())
            sess.run(tf.global_variables_initializer())
            print("Initialization time %f" % (time.time()-t0))
            sys.stdout.flush()

        train_loss = None
        best_loss = None
        early_stop_i = 0

        for epoch in range(1000):
            if early_stop_i < job['early_stop']:
                t0 = time.time()

                # save previous
                train_loss_last = train_loss

                # alternate forward and reverse batches
                rc_epoch = (epoch % 2) == 1

                # train
                train_loss = dr.train_epoch(sess, batcher_train, rc_epoch, train_writer)

                # validate
                valid_loss, valid_r2_list, _ = dr.test(sess, batcher_valid, rc_avg=options.rc, down_sample=options.down_sample)
                valid_r2 = valid_r2_list.mean()

                best_str = ''
                if best_loss is None or valid_loss < best_loss:
                    best_loss = valid_loss
                    best_str = ', best!'
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
                print('Epoch %3d: Train loss: %7.5f, Valid loss: %7.5f, Valid R2: %7.5f, Time: %s %s' % (epoch+1, train_loss, valid_loss, valid_r2, time_str, best_str), end='')

                # if training stagnant
                if options.learn_rate_drop and train_loss_last is not None and (train_loss_last - train_loss)/train_loss_last < 0.0005:
                    print(', rate drop', end='')
                    dr.drop_rate(2/3)

                print('')
                sys.stdout.flush()

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
