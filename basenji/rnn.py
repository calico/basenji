#!/usr/bin/env python
import sys
import time

import numpy as np

from sklearn.metrics import r2_score
import tensorflow as tf

class RNN:
    def __init__(self):
        pass


    def build(self, job):
        ###################################################
        # model parameters and placeholders
        ###################################################
        self.set_params(job)

        # batches
        self.inputs = tf.placeholder(tf.float32, shape=(self.batch_size, self.batch_length, self.seq_depth))
        self.targets = tf.placeholder(tf.float32, shape=(self.batch_size, self.batch_length//self.target_pool, self.num_targets))
        self.targets_na = tf.placeholder(tf.bool, shape=(self.batch_size, self.batch_length//self.target_pool))

        # dropout rates
        self.cnn_dropout_ph = []
        for li in range(self.cnn_layers):
            self.cnn_dropout_ph.append(tf.placeholder(tf.float32))
        self.dcnn_dropout_ph = []
        for li in range(self.dcnn_layers):
            self.dcnn_dropout_ph.append(tf.placeholder(tf.float32))
        self.rnn_dropout_ph = []
        for lin in range(self.rnn_layers):
            self.rnn_dropout_ph.append(tf.placeholder(tf.float32))

        # training conditional
        self.is_training = tf.placeholder(tf.bool)

        ###################################################
        # convolution layers
        ###################################################
        seq_length = self.batch_length
        seq_depth = self.seq_depth
        self.layer_reprs = []
        self.filter_weights = []

        # reshape for convolution
        cinput = tf.reshape(self.inputs, [self.batch_size, 1, seq_length, seq_depth])

        for li in range(self.cnn_layers):
            with tf.variable_scope('cnn%d' % li) as vs:
                # convolution params
                stdev = 1./np.sqrt(self.cnn_filters[li]*seq_depth)
                kernel = tf.Variable(tf.random_uniform([1, self.cnn_filter_sizes[li], seq_depth, self.cnn_filters[li]], minval=-stdev, maxval=stdev), name='kernel')
                biases = tf.Variable(tf.zeros([self.cnn_filters[li]]), name='bias')

                # maintain a pointer to the weights
                self.filter_weights.append(kernel)

                # convolution
                conv = tf.nn.conv2d(cinput, kernel, [1, 1, 1, 1], padding='SAME')

                # batch normalization
                # cinput = tf.contrib.layers.batch_norm(conv, center=True, scale=True, activation_fn=tf.nn.relu, is_training=True, updates_collections=None)

                # batch normalization (poor test performance)
                cinput = tf.contrib.layers.batch_norm(conv, decay=0.9, center=True, scale=True, activation_fn=tf.nn.relu, is_training=self.is_training, updates_collections=None)

                # batch normalization (poor test performance)
                # cinput = tf.cond(self.is_training,
                #             lambda: tf.contrib.layers.batch_norm(conv, is_training=True, center=True, scale=True, activation_fn=tf.nn.relu, updates_collections=None, scope=vs),
                #             lambda: tf.contrib.layers.batch_norm(conv, is_training=False, center=True, scale=True, activation_fn=tf.nn.relu, updates_collections=None, scope=vs, reuse=True))

                # nonlinearity
                # cinput = tf.nn.relu(tf.nn.bias_add(conv, biases), name='conv%d'%li)

                # pooling
                if self.cnn_pool[li] > 1:
                    cinput = tf.nn.max_pool(cinput, ksize=[1,1,self.cnn_pool[li],1], strides=[1,1,self.cnn_pool[li],1], padding='SAME', name='pool%d'%li)

                # dropout
                if self.cnn_dropout[li] > 0:
                    cinput = tf.nn.dropout(cinput, 1.0-self.cnn_dropout_ph[li])

                # updates size variables
                seq_length = seq_length // self.cnn_pool[li]
                seq_depth = self.cnn_filters[li]

                # save representation (not positive about this one)
                if self.save_reprs:
                    self.layer_reprs.append(tf.reshape(cinput, [self.batch_size, seq_length, seq_depth]))


        if self.cnn_layers > 0:
            # reshape for RNN
            # dinput = tf.reshape(cinput, [self.batch_size, seq_length, seq_depth])

            # pass to dilated CNN
            dinput = cinput

        else:
            # pass input along (assuming it's actually heading towards RNN layers)
            dinput = self.inputs

        # update batch buffer to reflect pooling
        pool_preds = self.batch_length // seq_length
        if self.batch_buffer % pool_preds != 0:
            print('Please make the batch_buffer %d divisible by the CNN pooling %d' % (self.batch_buffer, pool_preds), file=sys.stderr)
            exit(1)
        self.batch_buffer_pool = self.batch_buffer // pool_preds

        ###################################################
        # dilated convolution layers
        ###################################################

        # assuming dinput has been reshaped by convolution layers

        for li in range(self.dcnn_layers):
            with tf.variable_scope('dcnn%d' % li) as vs:
                # convolution params
                stdev = 1./np.sqrt(self.dcnn_filters[li]*seq_depth)
                kernel = tf.Variable(tf.random_uniform([1, self.dcnn_filter_sizes[li], seq_depth, self.dcnn_filters[li]], minval=-stdev, maxval=stdev), name='kernel')
                biases = tf.Variable(tf.zeros([self.dcnn_filters[li]]), name='bias')

                # convolution
                dinput = tf.nn.atrous_conv2d(dinput, kernel, rate=np.power(2,li), padding='SAME')

                # batch normalization and ReLU
                dinput = tf.contrib.layers.batch_norm(dinput, center=True, scale=True, activation_fn=tf.nn.relu, is_training=True, updates_collections=None)

                # dropout
                if self.dcnn_dropout[li] > 0:
                    dinput = tf.nn.dropout(dinput, 1.0-self.dcnn_dropout_ph[li])

                # update size variables
                seq_depth = self.dcnn_filters[li]

        # prep for RNN
        if self.cnn_layers + self.dcnn_layers > 0:
            # reshape
            rinput = tf.reshape(dinput, [self.batch_size, seq_length, seq_depth])

        else:
            # pass input along
            rinput = dinput


        ###################################################
        # recurrent layers
        ###################################################

        # move batch_length to the front as a list
        rinput = tf.unpack(tf.transpose(rinput, [1, 0, 2]))

        # initialize norm stabilizer
        norm_stabilizer = 0

        for li in range(self.rnn_layers):
            with tf.variable_scope('rnn%d' % li) as vs:
                # determine cell
                if self.cell == 'rnn':
                    cell = tf.nn.rnn_cell.BasicRNNCell(self.rnn_units[li], activation=self.activation)
                elif self.cell == 'gru':
                    cell = tf.nn.rnn_cell.GRUCell(self.rnn_units[li], activation=self.activation)
                elif self.cell == 'lstm':
                    cell = tf.nn.rnn_cell.LSTMCell(self.rnn_units[li], state_is_tuple=True, initializer=tf.contrib.layers.xavier_initializer(uniform=True), activation=self.activation)
                elif self.cell == 'block':
                    cell = tf.contrib.rnn.LSTMBlockCell(self.rnn_units[li])
                else:
                    print('Cannot recognize RNN cell type %s' % self.cell)
                    exit(1)

                # dropout
                if li < len(self.rnn_dropout) and self.rnn_dropout[li] > 0:
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=(1-self.rnn_dropout_ph[li]))

                # run bidirectional
                if self.cnn_layers == 0 and li == 0:
                    # outputs, _, _ = tf.nn.bidirectional_rnn(cell, cell, rinput, dtype=tf.float32)
                    outputs, _, _ = bidirectional_rnn_rc(cell, cell, rinput, dtype=tf.float32)
                else:
                    outputs, _, _ = bidirectional_rnn_tied(cell, cell, rinput, dtype=tf.float32)

                # update depth
                seq_depth = 2*self.rnn_units[li]

                # accumulate norm stablizer
                if self.norm_stabilizer[li] > 0:
                    output_norms = tf.sqrt(tf.reduce_sum(tf.square(outputs), reduction_indices=2))
                    output_norms_diff = tf.squared_difference(output_norms[self.batch_buffer_pool:,:], output_norms[:seq_length-self.batch_buffer_pool,:])
                    norm_stabilizer += self.norm_stabilizer[li]*tf.reduce_mean(output_norms_diff)

                # pooling
                if self.rnn_pool[li] > 1:
                    # pack into a tensor
                    outputs = tf.pack(outputs)

                    # transpose batch to the front
                    outputs = tf.transpose(outputs, [1, 0, 2])

                    # reshape to pretend 4D
                    outputs = tf.reshape(outputs, [self.batch_size, 1, seq_length, seq_depth])

                    if self.activation == 'tanh':
                        # max pool
                        max_pool = tf.nn.max_pool(outputs, ksize=[1,1,self.rnn_pool[li],1], strides=[1,1,self.rnn_pool[li],1], padding='SAME', name='pool%d'%li)

                        # abs max pool
                        abs_max_pool = tf.nn.max_pool(tf.abs(outputs), ksize=[1,1,self.rnn_pool[li],1], strides=[1,1,self.rnn_pool[li],1], padding='SAME', name='abs_pool%d'%li)

                        # construct matrix of 1/-1 for abs max
                        abs_max_mask1 =  tf.to_float(tf.equal(max_pool, abs_max_pool))
                        abs_max_mask = abs_max_mask1*2 - 1

                        outputs = abs_max_mask * abs_max_pool

                    else:
                        # pool
                        outputs = tf.nn.max_pool(outputs, ksize=[1,1,self.rnn_pool[li],1], strides=[1,1,self.rnn_pool[li],1], padding='SAME', name='pool%d'%li)

                    # updates size variable
                    seq_length = seq_length // self.rnn_pool[li]

                    # reshape to real 3D
                    outputs = tf.reshape(outputs, [self.batch_size, seq_length, seq_depth])

                    # transpose length to the front
                    outputs = tf.transpose(outputs, [1, 0, 2])

                    # unpack into a list
                    outputs = tf.unpack(outputs)

                    # update batch buffer to reflect pooling
                    pool_preds = self.batch_length // seq_length
                    if self.batch_buffer % pool_preds != 0:
                        print('Please make the batch_buffer %d divisible by the pooling %d' % (self.batch_buffer, pool_preds), file=sys.stderr)
                        exit(1)
                    self.batch_buffer_pool = self.batch_buffer // pool_preds

                # save representation
                if self.save_reprs:
                    self.layer_reprs.append(tf.transpose(tf.pack(outputs), [1, 0, 2]))

                # outputs become input to next layer
                rinput = outputs

        # prep for target prediction
        if self.rnn_layers == 0:
            outputs = rinput

        ###################################################
        # output layers
        ###################################################
        with tf.variable_scope('out'):
            out_weights = tf.get_variable(name='weights', shape=[seq_depth, self.num_targets], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(uniform=True))
            out_biases = tf.Variable(tf.zeros(self.num_targets), name='bias')

        # make final predictions
        preds_length = []
        for li in range(self.batch_buffer_pool, seq_length-self.batch_buffer_pool):
            preds_length.append(tf.matmul(outputs[li], out_weights) + out_biases)

        # convert list to tensor
        preds = tf.pack(preds_length)

        # transpose back to batches in front
        self.preds_op = tf.transpose(preds, [1, 0, 2], name='predictions')

        # clip predictions
        if self.target_space == 'positive':
            self.preds_op = tf.nn.relu(self.preds_op)

        # repeat if pooling
        pool_repeat = pool_preds // self.target_pool
        if pool_repeat > 1:
            tlength = (self.batch_length-2*self.batch_buffer) // self.target_pool
            self.preds_op = tf.reshape(tf.tile(tf.reshape(self.preds_op, (-1,self.num_targets)), (1,pool_repeat)), (self.batch_size, tlength, self.num_targets))

        # print variables
        for v in tf.all_variables():
            print(v.name, v.get_shape())


        ###################################################
        # loss and optimization
        ###################################################

        # remove buffer
        tstart = self.batch_buffer // self.target_pool
        tend = (self.batch_length - self.batch_buffer) // self.target_pool
        self.targets_op = self.targets[:,tstart:tend,:]

        if self.target_space == 'integer':
            # move negatives into exponential space and align positives
            #  clipping the negatives prevents overflow that TF dislikes
            self.preds_op = tf.select(self.preds_op > 0, self.preds_op + 1, tf.exp(tf.clip_by_value(self.preds_op,-50,50)))

            # Poisson loss
            self.loss_op = tf.nn.log_poisson_loss(tf.log(self.preds_op), self.targets_op, compute_full_loss=False)
            self.loss_op = tf.reduce_mean(self.loss_op)

        else:
            # clip targets
            if self.target_space == 'positive':
                self.targets_op = tf.nn.relu(self.targets_op)

            # take square difference
            sq_diff = tf.squared_difference(self.preds_op, self.targets_op)

            # set NaN's to zero
            # sq_diff = tf.boolean_mask(sq_diff, tf.logical_not(self.targets_na[:,tstart:tend]))

            # take the mean
            self.loss_op = tf.reduce_mean(sq_diff, name='r2_loss') + norm_stabilizer

        # track
        tf.scalar_summary('loss', self.loss_op)

        # define optimization
        if self.optimization == 'adam':
            self.opt = tf.train.AdamOptimizer(self.learning_rate, beta1=self.adam_beta1, beta2=self.adam_beta2, epsilon=self.adam_eps)
        else:
            print('Cannot recognize optimization algorithm %s' % self.optimization)
            exit(1)

        # clip gradients
        self.gvs = self.opt.compute_gradients(self.loss_op)
        if self.grad_clip is None:
            clip_gvs =  self.gvs
        else:
            # self.gvs = [(tf.clip_by_value(g, -self.grad_clip, self.grad_clip), v) for g, v in self.gvs]

            # batch norm introduces these None values that we have to dodge
            clip_gvs = []
            for i in range(len(self.gvs)):
                g,v = self.gvs[i]
                if g is None:
                    clip_gvs.append(self.gvs[i])
                else:
                    clip_gvs.append((tf.clip_by_value(g, -self.grad_clip, self.grad_clip), v))

        # apply gradients
        self.step_op = self.opt.apply_gradients(clip_gvs)


        ###################################################
        # summary
        ###################################################
        self.merged_summary = tf.merge_all_summaries()

        # initialize steps
        self.step = 0


    def drop_rate(self):
        ''' Drop the optimizer learning rate. '''
        self.opt._lr *= 2/3


    def hidden(self, sess, batcher, layers=None):
        ''' Compute hidden representations for a test set. '''

        if layers is None:
            layers = list(range(self.cnn_layers+self.rnn_layers))

        # initialize layer representation data structure
        layer_reprs = []
        for li in range(1+np.max(layers)):
            layer_reprs.append([])
        preds = []

        # setup feed dict for dropout
        fd = {}
        for li in range(self.cnn_layers):
            fd[self.cnn_dropout_ph[li]] = 0
        for li in range(self.dcnn_layers):
            fd[self.dcnn_dropout_ph[li]] = 0
        for li in range(self.rnn_layers):
            fd[self.rnn_dropout_ph[li]] = 0

        # get first batch
        Xb, _, _, Nb = batcher.next()

        while Xb is not None:
            # update feed dict
            fd[self.inputs] = Xb

            # compute predictions
            layer_reprs_batch, preds_batch = sess.run([self.layer_reprs, self.preds_op], feed_dict=fd)

            # accumulate representations
            for li in layers:
                layer_reprs[li].append(layer_reprs_batch[li][:Nb])
            preds.append(preds_batch[:Nb])

            # next batch
            Xb, _, _, Nb = batcher.next()

        # reset batcher
        batcher.reset()

        # accumulate representations
        for li in layers:
            layer_reprs[li] = np.vstack(layer_reprs[li])
        preds = np.vstack(preds)

        return layer_reprs, preds


    def predict(self, sess, batcher, target_indexes=None):
        ''' Compute predictions on a test set. '''

        preds = []

        # setup feed dict for dropout
        fd = {self.is_training:False}
        for li in range(self.cnn_layers):
            fd[self.cnn_dropout_ph[li]] = 0
        for li in range(self.dcnn_layers):
            fd[self.dcnn_dropout_ph[li]] = 0
        for li in range(self.rnn_layers):
            fd[self.rnn_dropout_ph[li]] = 0

        # determine non-buffer region
        buf_start = self.batch_buffer // self.target_pool
        buf_end = (self.batch_length - self.batch_buffer) // self.target_pool
        buf_len = buf_end - buf_start

        # initialize prediction arrays
        num_targets = self.num_targets
        if target_indexes is not None:
            num_targets = len(target_indexes)

        preds = np.zeros((batcher.num_seqs, buf_len, num_targets), dtype='float16')

        si = 0

        # get first batch
        Xb, _, _, Nb = batcher.next()

        while Xb is not None:
            # update feed dict
            fd[self.inputs] = Xb

            # compute predictions
            preds_batch = sess.run(self.preds_op, feed_dict=fd)

            # filter for specific targets
            if target_indexes is not None:
                preds_batch = preds_batch[:,:,target_indexes]

            # accumulate predictions
            preds[si:si+Nb,:,:] = preds_batch[:Nb,:,:]

            # update sequence index
            si += Nb

            # next batch
            Xb, _, _, Nb = batcher.next()

        # reset batcher
        batcher.reset()

        return preds


    def set_params(self, job):
        ''' Set RNN parameters. '''

        ###################################################
        # data attributes
        ###################################################
        self.seq_depth = job.get('seq_depth', 4)
        self.num_targets = job['num_targets']
        self.target_pool = job.get('target_pool', 1)

        ###################################################
        # batching
        ###################################################
        self.batch_size = job.get('batch_size', 64)
        self.batch_length = job.get('batch_length', 1024)
        self.batch_buffer = job.get('batch_buffer', 64)

        ###################################################
        # training
        ###################################################
        self.learning_rate = job.get('learning_rate', 0.001)
        self.adam_beta1 = job.get('adam_beta1', 0.9)
        self.adam_beta2 = job.get('adam_beta2', 0.999)
        self.adam_eps = job.get('adam_eps', 1e-8)
        self.optimization = job.get('optimization', 'adam').lower()
        self.grad_clip = job.get('grad_clip', None)

        ###################################################
        # CNN params
        ###################################################
        self.cnn_filters = np.atleast_1d(job.get('cnn_filters', []))
        self.cnn_filter_sizes = np.atleast_1d(job.get('cnn_filter_sizes', []))
        self.cnn_layers = len(self.cnn_filters)
        self.cnn_pool = layer_extend(job.get('cnn_pool', []), 1, self.cnn_layers)

        ###################################################
        # dilated CNN params
        ###################################################
        self.dcnn_filters = np.atleast_1d(job.get('dcnn_filters', []))
        self.dcnn_filter_sizes = np.atleast_1d(job.get('dcnn_filter_sizes', []))
        self.dcnn_layers = len(self.dcnn_filters)

        ###################################################
        # RNN params
        ###################################################
        self.rnn_units = np.atleast_1d(job.get('rnn_units', []))
        self.rnn_layers = len(self.rnn_units)
        self.rnn_pool = layer_extend(job.get('rnn_pool', []), 1, self.rnn_layers)

        self.cell = job.get('cell', 'lstm').lower()
        self.activation = job.get('activation','tanh').lower()
        if self.activation == 'relu':
            self.activation = tf.nn.relu
        elif self.activation == 'tanh':
            self.activation = tf.tanh
        else:
            print('Activation %s not implemented' % self.activation, file=sys.stderr)
            exit(1)

        ###################################################
        # regularization
        ###################################################
        self.cnn_dropout = layer_extend(job.get('cnn_dropout', []), 0, self.cnn_layers)
        self.dcnn_dropout = layer_extend(job.get('dcnn_dropout', []), 0, self.dcnn_layers)
        self.rnn_dropout = layer_extend(job.get('rnn_dropout', []), 0, self.rnn_layers)
        self.norm_stabilizer = layer_extend(job.get('norm_stabilizer', []), 0, self.rnn_layers)

        # batch normalization?

        ###################################################
        # loss
        ###################################################
        self.target_space = job.get('target_space', 'real')
        if self.target_space not in ['real', 'positive', 'integer']:
            print('target_space: %s invalid. Must be one of real, positive, or integer' % self.target_space, file=sys.stderr)
            exit(1)

        ###################################################
        # other
        ###################################################
        self.save_reprs = job.get('save_reprs', False)


    def test(self, sess, batcher, return_preds=False, down_sample=1):
        ''' Compute model accuracy on a test set.

        Args:
          sess:         TensorFlow session
          batcher:      Batcher object to provide data
          return_preds: Bool indicating whether to return predictions
          down_sample:  Int specifying to consider uniformly spaced sampled positions

        Returns:
          mean_loss:    Mean loss across targets
          mean_r2:      Mean R^2 across targets
          preds:        Predictions
        '''

        batch_losses = []

        # determine non-buffer region
        buf_start = self.batch_buffer // self.target_pool
        buf_end = (self.batch_length - self.batch_buffer) // self.target_pool
        buf_len = buf_end - buf_start

        # uniformly sample indexes
        ds_indexes = np.arange(0, buf_len, down_sample)

        # initialize prediction and target arrays
        preds = np.zeros((batcher.num_seqs, len(ds_indexes), self.num_targets), dtype='float16')
        targets = np.zeros((batcher.num_seqs, len(ds_indexes), self.num_targets), dtype='float16')
        targets_na = np.zeros((batcher.num_seqs, len(ds_indexes)), dtype='bool')
        si = 0

        # setup feed dict for dropout
        fd = {self.is_training:False}
        for li in range(self.cnn_layers):
            fd[self.cnn_dropout_ph[li]] = 0
        for li in range(self.dcnn_layers):
            fd[self.dcnn_dropout_ph[li]] = 0
        for li in range(self.rnn_layers):
            fd[self.rnn_dropout_ph[li]] = 0

        # get first batch
        Xb, Yb, NAb, Nb = batcher.next()

        while Xb is not None:
            # update feed dict
            fd[self.inputs] = Xb
            fd[self.targets] = Yb
            fd[self.targets_na] = NAb

            # measure batch loss
            preds_batch, targets_batch, loss_batch = sess.run([self.preds_op, self.targets_op, self.loss_op], feed_dict=fd)

            # accumulate predictions and targets
            preds[si:si+Nb,:,:] = preds_batch[:Nb,ds_indexes,:]
            targets[si:si+Nb,:,:] = targets_batch[:Nb,ds_indexes,:]
            # targets[si:si+Nb,:,:] = Yb[:Nb,buf_start+ds_indexes,:]
            # targets_na[si:si+Nb,:] = NAb[:Nb,buf_start+ds_indexes]

            # accumulate loss
            # avail_sum = np.logical_not(targets_na[si:si+Nb,:]).sum()
            # batch_losses.append(loss_batch / avail_sum)
            batch_losses.append(loss_batch)

            # update sequence index
            si += Nb

            # next batch
            Xb, Yb, NAb, Nb = batcher.next()

        # reset batcher
        batcher.reset()

        # compute R2 per target
        r2 = np.zeros(self.num_targets)
        for ti in range(self.num_targets):
            preds_ti = preds[np.logical_not(targets_na),ti]
            targets_ti = targets[np.logical_not(targets_na),ti]

            # compute R2
            tmean = targets_ti.mean(dtype='float64')
            tvar = (targets_ti-tmean).var(dtype='float64')
            pvar = (targets_ti-preds_ti).var(dtype='float64')
            r2[ti] = 1.0 - pvar/tvar

        if return_preds:
            return np.mean(batch_losses), r2, preds
        else:
            return np.mean(batch_losses), r2


    def train_epoch(self, sess, batcher, sum_writer):
        ''' Execute one training epoch '''

        # initialize training loss
        train_loss = []

        # setup feed dict for dropout
        fd = {self.is_training:True}
        for li in range(self.cnn_layers):
            fd[self.cnn_dropout_ph[li]] = self.cnn_dropout[li]
        for li in range(self.dcnn_layers):
            fd[self.dcnn_dropout_ph[li]] = self.dcnn_dropout[li]
        for li in range(self.rnn_layers):
            fd[self.rnn_dropout_ph[li]] = self.rnn_dropout[li]

        # get first batch
        Xb, Yb, NAb, Nb = batcher.next()

        while Xb is not None and Nb == self.batch_size:
            # update feed dict
            fd[self.inputs] = Xb
            fd[self.targets] = Yb
            fd[self.targets_na] = NAb

            summary, loss_batch, _ = sess.run([self.merged_summary, self.loss_op, self.step_op], feed_dict=fd)

            # pull gradients
            # gvs_batch = sess.run([g for (g,v) in self.gvs if g is not None], feed_dict=fd)

            # add summary
            if sum_writer is not None:
                sum_writer.add_summary(summary, self.step)

            # accumulate loss
            # avail_sum = np.logical_not(NAb[:Nb,:]).sum()
            # train_loss.append(loss_batch / avail_sum)
            train_loss.append(loss_batch)

            # next batch
            Xb, Yb, NAb, Nb = batcher.next()
            self.step += 1

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

###################################################################################################
# TensorFlow adjustments
###################################################################################################
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.rnn import _reverse_seq

################################################################################
def bidirectional_rnn_tied(cell_fw, cell_bw, inputs, initial_state_fw=None, initial_state_bw=None, dtype=None, sequence_length=None, scope=None):
    name = scope or "BiRNN"
    with vs.variable_scope(name) as fw_scope:
        # Forward direction
        output_fw, output_state_fw = tf.nn.rnn(cell_fw, inputs, initial_state_fw, dtype, sequence_length, scope=fw_scope)

    with vs.variable_scope(name, reuse=True) as bw_scope:
        # Backward direction
        tmp, output_state_bw = tf.nn.rnn(cell_bw, _reverse_seq(inputs, sequence_length),
                 initial_state_bw, dtype, sequence_length, scope=bw_scope)

    output_bw = _reverse_seq(tmp, sequence_length)

    # Concat each of the forward/backward outputs
    outputs = [array_ops.concat(1, [fw, bw]) for fw, bw in zip(output_fw, output_bw)]

    return (outputs, output_state_fw, output_state_bw)

################################################################################
def bidirectional_rnn_rc(cell_fw, cell_bw, inputs, initial_state_fw=None, initial_state_bw=None, dtype=None, sequence_length=None, scope=None):
    name = scope or "BiRNN"
    with vs.variable_scope(name) as fw_scope:
        # Forward direction
        output_fw, output_state_fw = tf.nn.rnn(cell_fw, inputs, initial_state_fw, dtype, sequence_length, scope=fw_scope)

    with vs.variable_scope(name, reuse=True) as bw_scope:
        # Backward direction
        tmp, output_state_bw = tf.nn.rnn(cell_bw, _reverse_complement(inputs, sequence_length),
                 initial_state_bw, dtype, sequence_length, scope=bw_scope)

    output_bw = _reverse_seq(tmp, sequence_length)

    # Concat each of the forward/backward outputs
    outputs = [array_ops.concat(1, [fw, bw]) for fw, bw in zip(output_fw, output_bw)]

    return (outputs, output_state_fw, output_state_bw)

################################################################################
def _reverse_complement(input_seq, lengths):
    """Reverse complement a list of one hot coded nucleotide Tensors.
    Args:
    input_seq: Sequence of seq_len tensors of dimension (batch_size, 4)
    lengths:   A `Tensor` of dimension batch_size, containing lengths for each
               sequence in the batch. If "None" is specified, simply reverse
               complements the list.
    Returns:
    reverse complemented sequence
    """
    if lengths is not None:
        print('Not yet implemented', file=sys.stderr)
        exit(1)
    else:
        nt_rc = tf.constant([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]], dtype='float32')
        return [tf.matmul(ris,nt_rc) for ris in reversed(input_seq)]
