#!/usr/bin/env python
from optparse import OptionParser
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
    dr = basenji.rnn.RNN()
    dr.load(model_file)

    if options.batch_size is None:
        options.batch_size = dr.batch_size

    #######################################################
    # train
    #######################################################
    # initialize batcher
    batcher_test = basenji.batcher.Batcher(test_seqs, test_targets, options.batch_size)

    with tf.Session() as sess:
        # test
        test_loss, test_r2 = dr.test(sess, batcher_test)

        # print
        print('Test loss: %7.5f' % test_loss)
        print('Test R2:   %7.5f' % test_r2)

    data_open.close()


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
