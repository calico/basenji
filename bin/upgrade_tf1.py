#!/usr/bin/env python
from optparse import OptionParser

import h5py
import numpy as np
import pdb

import tensorflow as tf

from basenji import dataset
from basenji import params
from basenji import seqnn

'''
upgrade_tf1.py

Read in tf1 model and write into a preset tf2 hdf5.
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <params_tf1_file> <model_tf1_file> <model_tf2_h5>'
    parser = OptionParser(usage)
    parser.add_option('-f', dest='final_slice',
            default=None,
            help='Final dense layer target slice, e.g. "0:1000"')
    (options,args) = parser.parse_args()

    if len(args) != 3:
        parser.error('Must provide TF1 params and .tf files and TF2 .h5 file.')
    else:
        params_tf1_file = args[0]
        model_tf1_file = args[1]
        model_tf2_h5_file = args[2]

    ################################################################
    # params

    job = params.read_job_params(params_tf1_file, require=['seq_length','num_targets'])


    ################################################################
    # dummy data

    def dummy_gen():
        for i in range(16):
            yield i

    data_types = {'sequence': tf.float32}
    data_shapes = {'sequence': tf.TensorShape([tf.Dimension(job['seq_length']), tf.Dimension(4)])}

    dataset = tf.data.Dataset.from_generator(dummy_gen, output_types=data_types, output_shapes=data_shapes)
    dataset = dataset.batch(job['batch_size'])
    iterator = dataset.make_one_shot_iterator()
    data_ops = iterator.get_next()

    ################################################################
    # setup model

    model = seqnn.SeqNN()
    model.build_sad(job, data_ops)

    ################################################################
    # restore model and extract weights

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_tf1_file)
        model1_vars = tf.global_variables()
        model1_weights = sess.run(model1_vars)

    ################################################################
    # write into tf2 hdf5

    model_tf2_h5 = h5py.File(model_tf2_h5_file, 'r+')

    for weight1_name, weight1_val in zip(model1_vars, model1_weights):
        print(weight1_name.name, weight1_val.shape)

        # skip step
        if weight1_name.name == 'global_step:0':
            continue

        weight1_split = weight1_name.name.split('/')
        weight2_split = ['model_weights']

        if weight1_split[0] == 'final':
            weight2_split += [weight1_split[1]]*2

        else:
            li = int(weight1_split[0].replace('cnn',''))
            if li == 0:
                weight2_split += [weight1_split[1]]*2
            else:
                weight2_split += ['%s_%s' % (weight1_split[1],li)]*2

        weight2_split.append(weight1_split[-1])

        weight2_name = '/'.join(weight2_split)
        print(weight2_name, model_tf2_h5[weight2_name].shape, '\n')

        if weight1_split[0] == 'final' and options.final_slice is not None:
            fs, fe = options.final_slice.split(':')
            fs, fe = int(fs), int(fe)
            model_tf2_h5[weight2_name][...] = weight1_val[...,fs:fe]
        else:
            model_tf2_h5[weight2_name][...] = weight1_val

    model_tf2_h5.close()


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
