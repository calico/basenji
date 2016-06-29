#!/usr/bin/env python
from optparse import OptionParser
import time

import h5py
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
    usage = 'usage: %prog [options] <model_file> <data_file>'
    parser = OptionParser(usage)
    parser.add_option('-b', dest='batch_size', default=None, type='int', help='Batch size')
    parser.add_option('-j', '--job', dest='job')
    (options,args) = parser.parse_args()

    if len(args) != 2:
    	parser.error('Must provide model and test data HDF5')
    else:
        model_file = args[0]
        data_file = args[1]

    #######################################################
    # load data
    #######################################################
    data_open = h5py.File(data_file)
    test_seqs = data_open['test_in']
    test_targets = data_open['test_out']

    #######################################################
    # model parameters and placeholders
    #######################################################
    job = read_job_params(options.job)

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
    # train
    #######################################################
    # initialize batcher
    batcher_test = basenji.batcher.Batcher(test_seqs, test_targets, options.batch_size)

    # initialie saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # load variables into session
        saver.restore(sess, model_file)

        # test
        test_loss, test_r2 = dr.test(sess, batcher_test)

        # print
        print('Test loss: %7.5f' % test_loss)
        print('Test R2:   %7.5f' % test_r2)

    data_open.close()


def read_job_params(job_file):
    ''' Read job parameters from table. '''

    job = {}

    if job_file is not None:
        for line in open(job_file):
            param, val = line.split()

            # require a decimal for floats
            try:
                if val.find('.') == -1:
                    val = int(val)
                else:
                    val = float(val)
            except ValueError:
                pass

            if param in job:
                # change to a list
                if type(job[param]) != list:
                    job[param] = [job[param]]

                # append new value
                job[param].append(val)
            else:
                job[param] = val

        print(job)

    return job

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
