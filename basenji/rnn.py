#!/usr/bin/env python
import sys
import time

import numpy as np
import pyBigWig
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
import tensorflow as tf

from basenji.dna_io import hot1_rc

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

        print('Targets pooled by %d to length %d' % (self.target_pool, self.batch_length//self.target_pool))

        # dropout rates
        self.cnn_dropout_ph = []
        for li in range(self.cnn_layers):
            self.cnn_dropout_ph.append(tf.placeholder(tf.float32))
        self.dcnn_dropout_ph = []
        for li in range(self.dcnn_layers):
            self.dcnn_dropout_ph.append(tf.placeholder(tf.float32))
        self.rnn_dropout_ph = []
        for li in range(self.rnn_layers):
            self.rnn_dropout_ph.append(tf.placeholder(tf.float32))
        self.full_dropout_ph = []
        for li in range(self.full_layers):
            self.full_dropout_ph.append(tf.placeholder(tf.float32))

        # training conditional
        self.is_training = tf.placeholder(tf.bool)

        ###################################################
        # convolution layers
        ###################################################
        seq_length = self.batch_length
        seq_depth = self.seq_depth
        self.layer_reprs = []
        self.filter_weights = []

        if self.save_reprs:
            self.layer_reprs.append(self.inputs)

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
                print('Convolution w/ %d %dx%d filters' % (self.cnn_filters[li], seq_depth, self.cnn_filter_sizes[li]))

                # batch normalization
                cinput = tf.contrib.layers.batch_norm(conv, decay=0.9, center=True, scale=True, activation_fn=tf.nn.relu, is_training=self.is_training, updates_collections=None)
                print('Batch normalization')
                print('ReLU')

                # nonlinearity (w/o batch norm)
                # cinput = tf.nn.relu(tf.nn.bias_add(conv, biases), name='conv%d'%li)

                # pooling
                if self.cnn_pool[li] > 1:
                    cinput = tf.nn.max_pool(cinput, ksize=[1,1,self.cnn_pool[li],1], strides=[1,1,self.cnn_pool[li],1], padding='SAME', name='pool%d'%li)
                    print('Max pool %d' % self.cnn_pool[li])

                # dropout
                if self.cnn_dropout[li] > 0:
                    cinput = tf.nn.dropout(cinput, 1.0-self.cnn_dropout_ph[li])
                    print('Dropout w/ probability %.3f' % self.cnn_dropout[li])

                # updates size variables
                seq_length = seq_length // self.cnn_pool[li]
                seq_depth = self.cnn_filters[li]

                # save representation (not positive about this one)
                if self.save_reprs:
                    self.layer_reprs.append(cinput)


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

                # let the last convolution layer handle the rate 1 pass
                drate = np.power(2,li+1)

                # convolution
                doutput = tf.nn.atrous_conv2d(dinput, kernel, rate=drate, padding='SAME')
                print('Dilated convolution w/ %d %dx%d rate %d filters' % (self.dcnn_filters[li], seq_depth, self.dcnn_filter_sizes[li], drate))


                # batch normalization and ReLU
                doutput = tf.contrib.layers.batch_norm(doutput, decay=0.9, center=True, scale=True, activation_fn=tf.nn.relu, is_training=self.is_training, updates_collections=None)
                print('Batch normalization')
                print('ReLU')

                # dropout
                if self.dcnn_dropout[li] > 0:
                    doutput = tf.nn.dropout(doutput, 1.0-self.dcnn_dropout_ph[li])
                    print('Dropout w/ probability %.3f' % self.dcnn_dropout[li])

                if self.dense_dilate:
                    # concat to dinput
                    dinput = tf.concat(3, [dinput, doutput])

                    # update size variables
                    seq_depth += self.dcnn_filters[li]
                else:
                    # move doutput to dinput
                    dinput = doutput

                    # update size variables
                    seq_depth = self.dcnn_filters[li]

                # save representation (not positive about this one)
                if self.save_reprs:
                    self.layer_reprs.append(dinput)

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

        if self.rnn_layers == 0:
            outputs = rinput

        else:
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
                        # cannot take gradients w.r.t. this,
                        #  but I'm not in an RNN mindset to fix it right now
                        self.layer_reprs.append(tf.transpose(tf.pack(outputs), [1, 0, 2]))

                    # outputs become input to next layer
                    rinput = outputs

            # move batch_length back to the middle as a tensor
            outputs = tf.pack(tf.transpose(outputs, [1, 0, 2]))


        ###################################################
        # fully connected layers
        ###################################################

        # slice out buffer regions
        outputs = outputs[:,self.batch_buffer_pool:seq_length-self.batch_buffer_pool,:]
        seq_length -= 2*self.batch_buffer_pool

        # reshape to make every position an element
        outputs = tf.reshape(outputs, (self.batch_size*seq_length, seq_depth))

        for li in range(self.full_layers):
            with tf.variable_scope('full%d' % li):
                # linear transform
                full_weights = tf.get_variable(name='weights', shape=[seq_depth, self.full_units[li]], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(uniform=True))
                full_biases = tf.Variable(tf.zeros(self.full_units), name='bias')

                outputs = tf.matmul(outputs, full_weights) + full_biases
                print('Linear transformation %dx%d' % (seq_depth, self.full_units[li]))

                # batch normalization
                outputs = tf.contrib.layers.batch_norm(outputs, decay=0.9, center=True, scale=True, activation_fn=tf.nn.relu, is_training=self.is_training, updates_collections=None)
                print('Batch normalization')
                print('ReLU')

                # dropout
                if self.full_dropout[li] > 0:
                    outputs = tf.nn.dropout(outputs, 1.0-self.full_dropout_ph[li])
                    print('Dropout w/ probability %.3f' % self.full_dropout[li])

                # update
                seq_depth = self.full_units[li]

        ###################################################
        # final layer
        ###################################################

        with tf.variable_scope('final'):
            # linear transform
            final_weights = tf.get_variable(name='weights', shape=[seq_depth, self.num_targets], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(uniform=True))
            final_biases = tf.Variable(tf.zeros(self.num_targets), name='bias')

            self.preds_op = tf.matmul(outputs, final_weights) + final_biases
            print('Linear transform %dx%d' % (seq_depth, self.num_targets))

        # expand length back out
        self.preds_op = tf.reshape(self.preds_op, (self.batch_size, seq_length, self.num_targets))

        # clip predictions
        if self.target_space == 'positive':
            self.preds_op = tf.nn.relu(self.preds_op)

        # repeat if pooling
        # pool_repeat = pool_preds // self.target_pool
        # if pool_repeat > 1:
        #     tlength = (self.batch_length-2*self.batch_buffer) // self.target_pool
        #     self.preds_op = tf.reshape(tf.tile(tf.reshape(self.preds_op, (-1,self.num_targets)), (1,pool_repeat)), (self.batch_size, tlength, self.num_targets))

        ###################################################
        # loss and optimization
        ###################################################

        # slice out buffer regions
        tstart = self.batch_buffer // self.target_pool
        tend = (self.batch_length - self.batch_buffer) // self.target_pool
        self.targets_op = self.targets[:,tstart:tend,:]

        # work-around for specifying my own predictions
        self.preds_adhoc = tf.placeholder(tf.float32, shape=self.preds_op.get_shape())

        if self.target_space == 'integer':
            # move negatives into exponential space and align positives
            #  clipping the negatives prevents overflow that TF dislikes
            self.preds_op = tf.select(self.preds_op > 0, self.preds_op + 1, tf.exp(tf.clip_by_value(self.preds_op,-50,50)))

            # Poisson loss
            self.loss_op = tf.nn.log_poisson_loss(tf.log(self.preds_op), self.targets_op, compute_full_loss=True)
            self.loss_op = tf.reduce_mean(self.loss_op)

            # work-around for computing loss from my own predictions
            self.loss_adhoc = tf.nn.log_poisson_loss(tf.log(self.preds_adhoc), self.targets_op, compute_full_loss=True)
            self.loss_adhoc = tf.reduce_mean(self.loss_adhoc)

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

            # work-around for computing loss from my own predictions
            self.loss_adhoc = tf.reduce_mean(tf.squared_difference(self.preds_adhoc, self.targets_op)) + norm_stabilizer

        # track
        tf.scalar_summary('loss', self.loss_op)

        # define optimization
        if self.optimization == 'adam':
            self.opt = tf.train.AdamOptimizer(self.learning_rate, beta1=self.adam_beta1, beta2=self.adam_beta2, epsilon=self.adam_eps)
        else:
            print('Cannot recognize optimization algorithm %s' % self.optimization)
            exit(1)

        # clip gradients
        self.gvs = self.opt.compute_gradients(self.loss_op, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
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


    def drop_rate(self, drop_mult=0.5):
        ''' Drop the optimizer learning rate. '''
        self.opt._lr *= drop_mult


    def gradients(self, sess, batcher, target_indexes=None, layers=None, return_preds=False):
        ''' Compute predictions on a test set.

        In
         sess: TensorFlow session
         batcher: Batcher class with sequence(s)
         target_indexes: Optional target subset list
         layers: Optional layer subset list

        Out
         grads: [S (sequences) x Li (layer i shape) x T (targets) array] * (L layers)
         preds:
        '''

        # initialize target_indexes
        if target_indexes is None:
            target_indexes = np.array(range(self.num_targets))
        elif type(target_indexes) != np.ndarray:
            target_indexes = np.array(target_indexes)

        # initialize gradients
        #  (I need a list for layers because the sizes are different within)
        #  (I'm using a list for targets because I need to compute them individually)
        layer_grads = []
        for lii in range(len(layers)):
            layer_grads.append([])
            for tii in range(len(target_indexes)):
                layer_grads[-1].append([])

        # initialize layers
        if layers is None:
            layers = range(1+self.cnn_layers+self.dcnn_layers+self.rnn_layers)
        elif type(layers) != list:
            layers = [layers]

        # initialize predictions
        preds = None
        if return_preds:
            # determine non-buffer region
            buf_start = self.batch_buffer // self.target_pool
            buf_end = (self.batch_length - self.batch_buffer) // self.target_pool
            buf_len = buf_end - buf_start

            # initialize predictions
            preds = np.zeros((batcher.num_seqs, buf_len, len(target_indexes)), dtype='float16')

            # sequence index
            si = 0

        # setup feed dict for dropout
        fd = self.set_mode('test')

        # get first batch
        Xb, _, _, Nb = batcher.next()

        while Xb is not None:
            # update feed dict
            fd[self.inputs] = Xb

            # predict
            preds_batch = sess.run(self.preds_op, feed_dict=fd)

            # compute gradients for each target individually
            for tii in range(len(target_indexes)):
                ti = target_indexes[tii]

                # compute gradients
                grads_op = tf.gradients(self.preds_op[:,:,ti], [self.layer_reprs[li] for li in layers])
                grads_batch_raw = sess.run(grads_op, feed_dict=fd)

                for lii in range(len(layers)):
                    # clean up
                    grads_batch = grads_batch_raw[lii][:Nb].astype('float16')
                    if grads_batch.shape[1] == 1:
                        grads_batch = grads_batch.squeeze(axis=1)

                    # save
                    layer_grads[lii][tii].append(grads_batch)

            if return_preds:
                # filter for specific targets
                if target_indexes is not None:
                    preds_batch = preds_batch[:,:,target_indexes]

                # accumulate predictions
                preds[si:si+Nb,:,:] = preds_batch[:Nb,:,:]

                # update sequence index
                si += Nb

            # next batch
            Xb, _, _, Nb = batcher.next()

        # reset training batcher
        batcher.reset()

        # stack into arrays
        for lii in range(len(layers)):
            for tii in range(len(target_indexes)):
                # stack sequences
                layer_grads[lii][tii] = np.vstack(layer_grads[lii][tii])

            # transpose targets to back
            layer_grads[lii] = np.array(layer_grads[lii])
            if layer_grads[lii].ndim == 4:
                # length dimension
                layer_grads[lii] = np.transpose(layer_grads[lii], [1,2,3,0])
            else:
                # no length dimension
                layer_grads[lii] = np.transpose(layer_grads[lii], [1,2,0])


        if return_preds:
            return layer_grads, preds
        else:
            return layer_grads


    def hidden(self, sess, batcher, layers=None):
        ''' Compute hidden representations for a test set. '''

        if layers is None:
            layers = list(range(self.cnn_layers+self.dcnn_layers+self.rnn_layers))

        # initialize layer representation data structure
        layer_reprs = []
        for li in range(1+np.max(layers)):
            layer_reprs.append([])
        preds = []

        # setup feed dict
        fd = self.set_mode('test')

        # get first batch
        Xb, _, _, Nb = batcher.next()

        while Xb is not None:
            # update feed dict
            fd[self.inputs] = Xb

            # compute predictions
            layer_reprs_batch, preds_batch = sess.run([self.layer_reprs, self.preds_op], feed_dict=fd)

            # accumulate representations
            for li in layers:
                # squeeze (conv_2d-expanded) second dimension
                if layer_reprs_batch[li].shape[1] == 1:
                    layer_reprs_batch[li] = layer_reprs_batch[li].squeeze(axis=1)

                # append
                layer_reprs[li].append(layer_reprs_batch[li][:Nb].astype('float16'))

            # accumualte predictions
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


    def predict(self, sess, batcher, rc_avg=False, target_indexes=None):
        ''' Compute predictions on a test set.

        In
         sess: TensorFlow session
         batcher: Batcher class with transcript-covering sequences
         rc_avg: Average predictions from the forward and reverse complement sequences
         target_indexes: Optional target subset list

        Out
         preds: S (sequences) x L (unbuffered length) x T (targets) array
        '''

        # setup feed dict
        fd = self.set_mode('test')

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

            if rc_avg:
                # compute reverse complement prediction
                fd[self.inputs] = hot1_rc(Xb)
                preds_batch_rc = sess.run(self.preds_op, feed_dict=fd)

                # average with forward prediction
                preds_batch += preds_batch_rc[:,::-1,:]
                preds_batch /= 2

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


    def predict_genes(self, sess, batcher, transcript_map, rc_avg=False, target_indexes=None):
        ''' Compute predictions on a test set.

        In
         sess: TensorFlow session
         batcher: Batcher class with transcript-covering sequences
         transcript_map: OrderedDict mapping transcript id's to (sequence index, position) tuples marking TSSs.
         rc_avg: Average predictions from the forward and reverse complement sequences
         target_indexes: Optional target subset list

        Out
         transcript_preds: G (gene transcripts) X T (targets) array
        '''

        # setup feed dict
        fd = self.set_mode('test')

        # initialize prediction arrays
        num_targets = self.num_targets
        if target_indexes is not None:
            num_targets = len(target_indexes)

        # initialize gene target predictions
        num_genes = len(transcript_map)
        gene_preds = np.zeros((num_genes, num_targets), dtype='float16')

        # construct an inverse map
        sequence_pos_transcripts = []
        txi = 0
        for transcript in transcript_map:
            si, pos = transcript_map[transcript]

            # extend sequence list
            while len(sequence_pos_transcripts) <= si:
                sequence_pos_transcripts.append({})

            # add gene to position set
            sequence_pos_transcripts[si].setdefault(pos,set()).add(txi)

            txi += 1

        '''
        sequence_transcripts = []
        txi = 0
        for transcript in transcript_map:
            tsi, tpos = transcript_map[transcript]
            while len(sequence_transcripts) <= tsi:
                sequence_transcripts.append([])
            sequence_transcripts[tsi].append((txi,tpos))
            txi += 1
        '''

        si = 0

        # get first batch
        Xb, _, _, Nb = batcher.next()

        while Xb is not None:
            # update feed dict
            fd[self.inputs] = Xb

            # compute predictions
            preds_batch = sess.run(self.preds_op, feed_dict=fd)

            if rc_avg:
                # compute reverse complement prediction
                fd[self.inputs] = hot1_rc(Xb)
                preds_batch_rc = sess.run(self.preds_op, feed_dict=fd)

                # average with forward prediction
                preds_batch += preds_batch_rc[:,::-1,:]
                preds_batch /= 2

            # filter for specific targets
            if target_indexes is not None:
                preds_batch = preds_batch[:,:,target_indexes]

            # for each sequence in the batch
            for pi in range(Nb):
                '''
                # for each transcript in the sequence
                for txi, tpos in sequence_transcripts[si+pi]:
                    # adjust for the buffer
                    ppos = tpos - self.batch_buffer//self.target_pool

                    # save transcript prediction
                    transcript_preds[txi,:] = preds_batch[pi,ppos,:]
                '''

                for tpos in sequence_pos_transcripts[si+pi]:
                    for txi in sequence_pos_transcripts[si+pi][tpos]:
                        # adjust for the buffer
                        ppos = tpos - self.batch_buffer//self.target_pool

                        # add prediction
                        gene_preds[txi,:] += preds_batch[pi,ppos,:]

            # update sequence index
            si += Nb

            # next batch
            Xb, _, _, Nb = batcher.next()

        # reset batcher
        batcher.reset()

        return gene_preds


    def set_mode(self, mode):
        ''' Construct a feed dictionary to specify the model's mode. '''
        fd = {}

        if mode in ['train', 'training']:
            fd[self.is_training] = True
            for li in range(self.cnn_layers):
                fd[self.cnn_dropout_ph[li]] = self.cnn_dropout[li]
            for li in range(self.dcnn_layers):
                fd[self.dcnn_dropout_ph[li]] = self.dcnn_dropout[li]
            for li in range(self.rnn_layers):
                fd[self.rnn_dropout_ph[li]] = self.rnn_dropout[li]
            for li in range(self.full_layers):
                fd[self.full_dropout_ph[li]] = self.full_dropout[li]

        elif mode in ['test', 'testing', 'evaluate']:
            fd[self.is_training] = False
            for li in range(self.cnn_layers):
                fd[self.cnn_dropout_ph[li]] = 0
            for li in range(self.dcnn_layers):
                fd[self.dcnn_dropout_ph[li]] = 0
            for li in range(self.rnn_layers):
                fd[self.rnn_dropout_ph[li]] = 0
            for li in range(self.full_layers):
                fd[self.full_dropout_ph[li]] = 0

        else:
            print('Cannot recognize mode %s' % mode)
            exit(1)

        return fd


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
        self.dense_dilate = bool(job.get('dense_dilate',False))
        self.dense_dilate = bool(job.get('dense',self.dense_dilate))

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
        # fully connected params
        ###################################################
        self.full_units = np.atleast_1d(job.get('full_units', []))
        self.full_layers = len(self.full_units)

        ###################################################
        # regularization
        ###################################################
        self.cnn_dropout = layer_extend(job.get('cnn_dropout', []), 0, self.cnn_layers)
        self.dcnn_dropout = layer_extend(job.get('dcnn_dropout', []), 0, self.dcnn_layers)
        self.rnn_dropout = layer_extend(job.get('rnn_dropout', []), 0, self.rnn_layers)
        self.full_dropout = layer_extend(job.get('full_dropout', []), 0, self.full_layers)
        self.norm_stabilizer = layer_extend(job.get('norm_stabilizer', []), 0, self.rnn_layers)

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


    def test(self, sess, batcher, rc_avg=False, return_preds=False, down_sample=1):
        ''' Compute model accuracy on a test set.

        Args:
          sess:         TensorFlow session
          batcher:      Batcher object to provide data
          rc_avg:       Average predictions from the forward and reverse complement sequences
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

        # setup feed dict
        fd = self.set_mode('test')

        # get first batch
        Xb, Yb, NAb, Nb = batcher.next()

        while Xb is not None:
            # update feed dict
            fd[self.inputs] = Xb
            fd[self.targets] = Yb
            fd[self.targets_na] = NAb

            # measure batch loss
            preds_batch, targets_batch, loss_batch = sess.run([self.preds_op, self.targets_op, self.loss_op], feed_dict=fd)

            if rc_avg:
                # compute reverse complement prediction
                fd[self.inputs] = hot1_rc(Xb)
                preds_batch_rc = sess.run(self.preds_op, feed_dict=fd)

                # average with forward prediction
                preds_batch += preds_batch_rc[:,::-1,:]
                preds_batch /= 2
                fd[self.preds_adhoc] = preds_batch
                loss_batch = sess.run(self.loss_adhoc, feed_dict=fd)

            # accumulate predictions and targets
            preds[si:si+Nb,:,:] = preds_batch[:Nb,ds_indexes,:]
            targets[si:si+Nb,:,:] = targets_batch[:Nb,ds_indexes,:]

            # accumulate loss
            batch_losses.append(loss_batch)

            # update sequence index
            si += Nb

            # next batch
            Xb, Yb, NAb, Nb = batcher.next()

        # reset batcher
        batcher.reset()

        # compute R2 per target
        r2 = np.zeros(self.num_targets)
        cor = np.zeros(self.num_targets)
        for ti in range(self.num_targets):
            preds_ti = preds[np.logical_not(targets_na),ti]
            targets_ti = targets[np.logical_not(targets_na),ti]

            # compute R2
            tmean = targets_ti.mean(dtype='float64')
            tvar = (targets_ti-tmean).var(dtype='float64')
            pvar = (targets_ti-preds_ti).var(dtype='float64')
            r2[ti] = 1.0 - pvar/tvar

            # compute Spearman correlation
            scor, _ = spearmanr(targets_ti, preds_ti)
            cor[ti] = scor

        if return_preds:
            return np.mean(batch_losses), r2, cor, preds
        else:
            return np.mean(batch_losses), r2, cor


    def train_epoch(self, sess, batcher, rc=False, sum_writer=None):
        ''' Execute one training epoch '''

        # initialize training loss
        train_loss = []

        # setup feed dict
        fd = self.set_mode('train')

        # get first batch
        Xb, Yb, NAb, Nb = batcher.next(rc)

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
            Xb, Yb, NAb, Nb = batcher.next(rc)
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
