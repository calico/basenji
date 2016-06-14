#!/usr/bin/env python
import time

import numpy as np
from sklearn.metrics import r2_score
import tensorflow as tf

import rnn_batcher

class RNN:
    def __init__(self):
        pass


    def build(self, job):
        ###################################################
        # model parameters and placeholders
        ###################################################
        self.set_params(job)

        self.inputs = tf.placeholder(tf.float32, shape=(self.batch_size, self.batch_length, self.seq_depth))
        self.targets = tf.placeholder(tf.float32, shape=(self.batch_size, self.batch_length, self.num_targets))

        with tf.variable_scope('out'):
            out_weights = tf.get_variable(name='weights', shape=[2*self.hidden_units[-1], self.num_targets], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(uniform=True))
            out_biases = tf.Variable(tf.zeros(self.num_targets), name='bias')

        ###################################################
        # construct model
        ###################################################
        # tf needs batch_length in the front as a list
        rinput = tf.unpack(tf.transpose(self.inputs, [1, 0, 2]))

        for li in range(self.layers):
            with tf.variable_scope('layer%d' % li) as vs:
                # determine cell
                if self.cell == 'rnn':
                    cell = tf.nn.rnn_cell.BasicRNNCell(self.hidden_units[li])
                elif self.cell == 'gru':
                    cell = tf.nn.rnn_cell.GRUCell(self.hidden_units[li])
                elif self.cell == 'lstm':
                    cell = tf.nn.rnn_cell.LSTMCell(self.hidden_units[li], state_is_tuple=True, initializer=tf.contrib.layers.xavier_initializer(uniform=True))
                else:
                    print('Cannot recognize RNN cell type %s' % self.cell)
                    exit(1)

                # run bidirectional
                outputs, _, _ = tf.nn.bidirectional_rnn(cell, cell, rinput, dtype=tf.float32)

                # outputs become input to next layer
                rinput = outputs

        '''
        # individual layers
        layer_cells = []
        for li in range(self.layers):
            with tf.variable_scope('layer%d' % li):
                # single cell
                if self.cell == 'rnn':
                    cell = tf.nn.rnn_cell.BasicRNNCell(self.hidden_units[li])
                elif self.cell == 'gru':
                    cell = tf.nn.rnn_cell.GRUCell(self.hidden_units[li])
                elif self.cell == 'lstm':
                    cell = tf.nn.rnn_cell.LSTMCell(self.hidden_units[li], state_is_tuple=True, initializer=tf.contrib.layers.xavier_initializer(uniform=True))
                else:
                    print('Cannot recognize RNN cell type %s' % self.cell)
                    exit(1)

                # add dropout
                if self.dropouts[li] > 0:
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=(1.0-self.dropouts[li]))

                layer_cells.append(cell)

        # stack
        if self.cell == 'lstm':
            rnn_stack = tf.nn.rnn_cell.MultiRNNCell(layer_cells, state_is_tuple=True)
        else:
            rnn_stack = tf.nn.rnn_cell.MultiRNNCell(layer_cells)

        # run the RNN
        outputs, states_fwd, states_rev = tf.nn.bidirectional_rnn(rnn_stack, rnn_stack, rinput, dtype=tf.float32)
        '''

        # throw away a buffer region on the side
        outputs = outputs[self.batch_buffer:-self.batch_buffer]

        # make final predictions
        preds_length = []
        for li in range(self.batch_length - 2*self.batch_buffer):
            preds_length.append(tf.matmul(outputs[li], out_weights) + out_biases)

        # convert list to tensor
        preds = tf.pack(preds_length)

        # transpose back to batches in front
        self.preds = tf.transpose(preds, [1, 0, 2])

        for v in tf.all_variables():
            print(v.name, v.get_shape())


        ###################################################
        # loss and optimization
        ###################################################
        # define loss function
        self.loss_op = tf.reduce_mean(tf.squared_difference(self.preds, self.targets[:,self.batch_buffer:self.batch_length-self.batch_buffer,:]))

        # define optimization
        if self.optimization == 'adam':
            opt = tf.train.AdamOptimizer(self.learning_rate, self.adam_beta1, self.adam_beta2)
        else:
            print('Cannot recognize optimization algorithm %s' % self.optimization)
            exit(1)

        # clip gradients
        gvs = opt.compute_gradients(self.loss_op)
        if self.grad_clip is None:
            clip_gvs = gvs
        else:
            clip_gvs = [(tf.clip_by_value(g, -self.grad_clip, self.grad_clip), v) for g, v in gvs]
        self.step_op = opt.apply_gradients(clip_gvs)


    def load(self):
        pass


    def set_params(self, job):
        ''' Set RNN parameters. '''

        ###################################################
        # data attributes
        ###################################################
        self.seq_depth = job['seq_depth']
        self.num_targets = job['num_targets']

        ###################################################
        # batching
        ###################################################
        self.batch_size = job.get('batch_size', 4)
        self.batch_length = job.get('batch_length', 1024)
        self.batch_buffer = job.get('batch_buffer', 8)

        ###################################################
        # training
        ###################################################
        self.learning_rate = job.get('learning_rate', 0.015)
        self.adam_beta1 = job.get('adam_beta1', 0.9)
        self.adam_beta2 = job.get('adam_beta2', 0.95)
        self.optimization = job.get('optimization', 'adam').lower()
        self.grad_clip = job.get('grad_clip', 2)

        ###################################################
        # RNN params
        ###################################################
        self.hidden_units = job.get('hidden_units', [100])
        self.layers = len(self.hidden_units)
        self.cell = job.get('cell', 'lstm').lower()

        ###################################################
        # regularization
        ###################################################
        self.dropouts = layer_extend(job.get('dropout',[]), 0, self.layers)

        # batch normalization?


    def test(self, sess, batcher):
        ''' Compute model accuracy on a test set. '''

        batch_losses = []
        preds = []
        targets = []

        # get first batch
        Xb, Yb = batcher.next()

        while Xb is not None:
            # measure batch loss
            preds_batch, loss_batch = sess.run([self.preds, self.loss_op], feed_dict={self.inputs:Xb, self.targets:Yb})

            # accumulate loss
            batch_losses.append(loss_batch)

            # accumulate predictions and targets
            preds.append(preds_batch)
            targets.append(Yb[:,self.batch_buffer:self.batch_length-self.batch_buffer,:])

            # next batch
            Xb, Yb = batcher.next()

        # reset batcher
        batcher.reset()

        # compute R2
        preds = np.vstack(preds)
        targets = np.vstack(targets)
        r2 = r2_score(targets.flatten(), preds.flatten())

        return np.mean(batch_losses), r2


    def train_epoch(self, sess, batcher):
        ''' Execute one training epoch '''

        # initialize training loss
        train_loss = []

        # get first batch
        Xb, Yb = batcher.next()

        while Xb is not None:
            # run step
            loss_batch, _ = sess.run([self.loss_op, self.step_op], feed_dict={self.inputs:Xb, self.targets:Yb})

            # accumulate loss
            train_loss.append(loss_batch)

            # next batch
            Xb, Yb = batcher.next()

        # reset training batcher
        batcher.reset()

        return np.mean(train_loss)


def layer_extend(var, default, layers):
    ''' Process job input to extend for the
         proper number of layers. '''

    # if it's a number
    if type(var) != list:
        # change the default to that number
        default = var

        # make it a list
        var = [var]

    # extend for each layer
    while len(var) < layers:
        var.append(default)

    return var


