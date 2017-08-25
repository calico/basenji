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
import sys
import time

import h5py
import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow as tf

import basenji.dna_io
import basenji.batcher
import basenji.seqnn

'''
basenji_train.py

'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <params_file> <data_file>'
    parser = OptionParser(usage)
    parser.add_option('-d', dest='down_sample', default=1, type='int', help='Down sample test computation by taking uniformly spaced positions [Default: %default]')
    parser.add_option('-l', dest='learn_rate_drop', default=False, action='store_true', help='Drop learning rate when training loss stalls [Default: %default]')
    parser.add_option('--mc', dest='mc_n', default=0, type='int', help='Monte carlo test iterations [Default: %default]')
    parser.add_option('-o', dest='output_file', help='Print accuracy output to file')
    parser.add_option('-r', dest='restart', help='Restart training this model')
    parser.add_option('--rc', dest='rc', default=False, action='store_true', help='Average the forward and reverse complement predictions when testing [Default: %default]')
    parser.add_option('-s', dest='save_prefix', default='houndnn')
    parser.add_option('--seed', dest='seed', type='float', default=1, help='RNG seed')
    parser.add_option('-u', dest='summary', default=None, help='TensorBoard summary directory')
    parser.add_option('--log_device_placement', dest='log_device_placement', default=False, help='Log device placement (ie, CPU or GPU) [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 2:
        parser.error('Must provide parameters and data files')
    else:
        params_file = args[0]
        data_file = args[1]

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
    job = basenji.dna_io.read_job_params(params_file)

    job['batch_length'] = train_seqs.shape[1]
    job['seq_depth'] = train_seqs.shape[2]
    job['num_targets'] = train_targets.shape[2]
    job['target_pool'] = int(np.array(data_open.get('pool_width', 1)))
    job['early_stop'] = job.get('early_stop', 16)
    job['rate_drop'] = job.get('rate_drop', 3)

    t0 = time.time()
    dr = basenji.seqnn.SeqNN()
    dr.build(job)
    print('Model building time %f' % (time.time()-t0))

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


    # checkpoints
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    if options.log_device_placement:
        config.log_device_placement = True
    with tf.Session(config=config) as sess:
        t0 = time.time()

        # set seed
        tf.set_random_seed(options.seed)

        if options.summary is None:
            train_writer = None
        else:
            train_writer = tf.summary.FileWriter(options.summary + '/train', sess.graph)

        if options.restart:
            # load variables into session
            saver.restore(sess, options.restart)
        else:
            # initialize variables
            print('Initializing...')
            # sess.run(tf.initialize_all_variables())
            sess.run(tf.global_variables_initializer())
            print("Initialization time %f" % (time.time()-t0))

        train_loss = None
        best_loss = None
        early_stop_i = 0
        undroppable_counter = 3
        max_drops = 8
        num_drops = 0

        for epoch in range(1000):
            if early_stop_i < job['early_stop']:
                t0 = time.time()

                # save previous
                train_loss_last = train_loss

                # alternate forward and reverse batches
                if options.rc:
                    rc_epoch = (epoch % 2) == 1
                else:
                    rc_epoch = 0

                # train
                train_loss = dr.train_epoch(sess, batcher_train, rc_epoch, train_writer)

                # validate
                valid_acc = dr.test(sess, batcher_valid, mc_n=options.mc_n, rc_avg=options.rc, down_sample=options.down_sample)
                valid_loss = valid_acc.loss
                valid_r2 = valid_acc.r2().mean()
                del valid_acc

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
                print('Epoch %3d: Train loss: %7.5f, Valid loss: %7.5f, Valid R2: %7.5f, Time: %s%s' % (epoch+1, train_loss, valid_loss, valid_r2, time_str, best_str), end='')

                # if training stagnant
                if options.learn_rate_drop and num_drops < max_drops and undroppable_counter == 0 and (train_loss_last - train_loss)/train_loss_last < 0.0002:
                    print(', rate drop', end='')
                    dr.drop_rate(2/3)
                    undroppable_counter = 1
                    num_drops += 1
                else:
                    undroppable_counter = max(0, undroppable_counter-1)

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
