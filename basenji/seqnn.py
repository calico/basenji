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

import gc
import pdb
import sys
import time

import numpy as np
import tensorflow as tf


from basenji.dna_io import hot1_rc, hot1_augment
import basenji.ops

class SeqNN:
    def __init__(self):
        pass

    def build(self, job, target_subset=None):
        ###################################################
        # model parameters and placeholders
        ###################################################
        self.set_params(job)

        # batches
        self.inputs = tf.placeholder(tf.float32, shape=(self.batch_size, self.batch_length, self.seq_depth), name='inputs')
        if self.target_classes == 1:
            self.targets = tf.placeholder(tf.float32, shape=(self.batch_size, self.batch_length//self.target_pool, self.num_targets), name='targets')
        else:
            self.targets = tf.placeholder(tf.int32, shape=(self.batch_size, self.batch_length//self.target_pool, self.num_targets), name='targets')
        self.targets_na = tf.placeholder(tf.bool, shape=(self.batch_size, self.batch_length//self.target_pool))

        print('Targets pooled by %d to length %d' % (self.target_pool, self.batch_length//self.target_pool))

        # dropout rates
        self.cnn_dropout_ph = []
        for li in range(self.cnn_layers):
            self.cnn_dropout_ph.append(tf.placeholder(tf.float32))

        self.global_step = tf.train.create_global_step()
        if self.batch_renorm:
            RMAX_decay = basenji.ops.adjust_max(5000, 50000, 1, 2, name='RMAXDECAY')
            DMAX_decay = basenji.ops.adjust_max(5000, 50000, 0, 3, name='DMAXDECAY')
            renorm_clipping = {'rmin':1./RMAX_decay, 'rmax':RMAX_decay, 'dmax':DMAX_decay}
            tf.summary.scalar('renorm_rmax', RMAX_decay)
            tf.summary.scalar('renorm_dmax', DMAX_decay)
        else:
            renorm_clipping = {}

        # training conditional
        self.is_training = tf.placeholder(tf.bool)

        ###################################################
        # convolution layers
        ###################################################
        seq_length = self.batch_length
        seq_depth = self.seq_depth

        weights_regularizers = 0
        self.layer_reprs = [self.inputs]
        self.filter_weights = []

        # reshape for convolution
        # seqs_repr = tf.reshape(self.inputs, [self.batch_size, 1, seq_length, seq_depth])
        seqs_repr = self.inputs

        for li in range(self.cnn_layers):
            with tf.variable_scope('cnn%d' % li) as vs:

                seqs_repr_next = tf.layers.conv1d(seqs_repr, filters=self.cnn_filters[li], kernel_size=[self.cnn_filter_sizes[li]], strides=self.cnn_strides[li], padding='same', dilation_rate=[self.cnn_dilation[li]], use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(), kernel_regularizer=None)
                print('Convolution w/ %d %dx%d filters strided %d, dilated %d' % (self.cnn_filters[li], seq_depth, self.cnn_filter_sizes[li], self.cnn_strides[li], self.cnn_dilation[li]))

                # regularize
                # if self.cnn_l2[li] > 0:
                #    weights_regularizers += self.cnn_l2[li]*tf.reduce_mean(tf.nn.l2_loss(kernel))

                # maintain a pointer to the weights
                # self.filter_weights.append(kernel)

                # batch normalization
                seqs_repr_next = tf.layers.batch_normalization(seqs_repr_next, momentum=0.9, training=self.is_training, renorm=self.batch_renorm, renorm_clipping=renorm_clipping, renorm_momentum=0.9)
                print('Batch normalization')

                # ReLU
                seqs_repr_next = tf.nn.relu(seqs_repr_next)
                print('ReLU')

                # pooling
                if self.cnn_pool[li] > 1:
                    seqs_repr_next = tf.layers.max_pooling1d(seqs_repr_next, pool_size=self.cnn_pool[li], strides=self.cnn_pool[li], padding='same')
                    print('Max pool %d' % self.cnn_pool[li])

                # dropout
                if self.cnn_dropout[li] > 0:
                    seqs_repr_next = tf.nn.dropout(seqs_repr_next, 1.0-self.cnn_dropout_ph[li])
                    # seqs_repr = tf.layers.dropout(seqs_repr, rate=self.cnn_dropout[li], training=self.is_training)
                    print('Dropout w/ probability %.3f' % self.cnn_dropout[li])

                # updates size variables
                seq_length = seq_length // self.cnn_pool[li]

                if self.cnn_dense[li]:
                    # concat layer repr
                    seqs_repr = tf.concat(values=[seqs_repr, seqs_repr_next], axis=2)

                    # update size variables
                    seq_depth += self.cnn_filters[li]
                else:
                    # update layer repr
                    seqs_repr = seqs_repr_next

                    # update size variables
                    seq_depth = self.cnn_filters[li]

                # save representation
                self.layer_reprs.append(seqs_repr)

        # update batch buffer to reflect pooling
        pool_preds = self.batch_length // seq_length
        if self.batch_buffer % pool_preds != 0:
            print('Please make the batch_buffer %d divisible by the CNN pooling %d' % (self.batch_buffer, pool_preds), file=sys.stderr)
            exit(1)
        self.batch_buffer_pool = self.batch_buffer // pool_preds


        ###################################################
        # slice out side buffer
        ###################################################

        # predictions
        seqs_repr = seqs_repr[:,self.batch_buffer_pool:seq_length-self.batch_buffer_pool,:]
        seq_length -= 2*self.batch_buffer_pool
        self.preds_length = seq_length

        # save penultimate representation
        self.penultimate_op = seqs_repr

        # targets
        tstart = self.batch_buffer // self.target_pool
        tend = (self.batch_length - self.batch_buffer) // self.target_pool
        self.targets_op = tf.identity(self.targets[:,tstart:tend,:], name='targets_op')

        if target_subset is not None:
            self.targets_op = tf.gather(self.targets_op, target_subset, axis=2)

        ###################################################
        # final layer
        ###################################################
        with tf.variable_scope('final'):
            final_filters = self.num_targets*self.target_classes

            final_repr = tf.layers.conv1d(seqs_repr, filters=final_filters, kernel_size=[1], padding='same', use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), kernel_regularizer=None)
            print('Convolution w/ %d %dx1 filters to final targets' % (final_filters, seq_depth))

            # if self.final_l1 > 0:
            #     weights_regularizers += self.final_l1*tf.reduce_mean(tf.abs(final_weights))

            if target_subset is not None:
                # get convolution parameters
                filters_full = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'final/conv1d/kernel')[0]
                bias_full = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'final/conv1d/bias')[0]

                # subset to specific targets
                filters_subset = tf.gather(filters_full, target_subset, axis=2)
                bias_subset = tf.gather(bias_full, target_subset, axis=0)

                # substitute a new limited convolution
                final_repr = tf.nn.conv1d(seqs_repr, filters_subset, stride=1, padding='SAME')
                final_repr = tf.nn.bias_add(final_repr, bias_subset)

                # update # targets
                self.num_targets = len(target_subset)

            # switch back to expected name
            seqs_repr = final_repr

        # expand length back out
        if self.target_classes > 1:
            seqs_repr = tf.reshape(seqs_repr, (self.batch_size, seq_length, self.num_targets, self.target_classes))


        ###################################################
        # loss and optimization
        ###################################################

        # work-around for specifying my own predictions
        self.preds_adhoc = tf.placeholder(tf.float32, shape=seqs_repr.get_shape())

        # choose link
        if self.link in ['identity','linear']:
            self.preds_op = tf.identity(seqs_repr, name='preds')

        elif self.link == 'relu':
            self.preds_op = tf.relu(seqs_repr, name='preds')

        elif self.link == 'exp':
            self.preds_op = tf.exp(tf.clip_by_value(seqs_repr,-50,50), name='preds')

        elif self.link == 'exp_linear':
            self.preds_op = tf.where(seqs_repr > 0, seqs_repr + 1, tf.exp(tf.clip_by_value(seqs_repr,-50,50)), name='preds')

        elif self.link == 'softplus':
            self.preds_op = tf.nn.softplus(tf.clip_by_value(seqs_repr,-50,50), name='preds')

        elif self.link == 'softmax':
            # performed in the loss function, but saving probabilities
            self.preds_prob = tf.nn.softmax(seqs_repr, name='preds')

        else:
            print('Unknown link function %s' % self.link, file=sys.stderr)
            exit(1)

        # clip
        if self.target_clip is not None:
            self.preds_op = tf.clip_by_value(self.preds_op, 0, self.target_clip)
            self.targets_op = tf.clip_by_value(self.targets_op, 0, self.target_clip)

        # sqrt
        if self.target_sqrt:
            self.preds_op = tf.sqrt(self.preds_op)
            self.targets_op = tf.sqrt(self.targets_op)

        # choose loss
        if self.loss == 'gaussian':
            self.loss_op = tf.squared_difference(self.preds_op, self.targets_op)
            self.loss_adhoc = tf.squared_difference(self.preds_adhoc, self.targets_op)

        elif self.loss == 'poisson':
            self.loss_op = tf.nn.log_poisson_loss(self.targets_op, tf.log(self.preds_op), compute_full_loss=True)
            self.loss_adhoc = tf.nn.log_poisson_loss(self.targets_op, tf.log(self.preds_adhoc), compute_full_loss=True)

        elif self.loss == 'cross_entropy':
            self.loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=(self.targets_op-1), logits=self.preds_op)
            self.loss_adhoc = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=(self.targets_op-1), logits=self.preds_adhoc)

        else:
            print('Cannot identify loss function %s' % self.loss)
            exit(1)

        self.loss_op = tf.check_numerics(self.loss_op, 'Invalid loss', name='loss_check')

        # reduce lossses by batch and position
        self.loss_op = tf.reduce_mean(self.loss_op, axis=[0,1], name='target_loss')
        self.loss_adhoc = tf.reduce_mean(self.loss_adhoc, axis=[0,1], name='target_loss_adhoc')
        tf.summary.histogram('target_loss', self.loss_op)
        for ti in np.linspace(0,self.num_targets-1,10).astype('int'):
            tf.summary.scalar('loss_t%d'%ti, self.loss_op[ti])
        self.target_losses = self.loss_op
        self.target_losses_adhoc = self.loss_adhoc

        # fully reduce
        self.loss_op = tf.reduce_mean(self.loss_op, name='loss')
        self.loss_adhoc = tf.reduce_mean(self.loss_adhoc, name='loss_adhoc')

        # add extraneous terms
        self.loss_op += weights_regularizers
        self.loss_adhoc += weights_regularizers

        # track
        tf.summary.scalar('loss', self.loss_op)

        # define optimization
        if self.optimization == 'adam':
            self.opt = tf.train.AdamOptimizer(self.learning_rate, beta1=self.adam_beta1, beta2=self.adam_beta2, epsilon=self.adam_eps)
        elif self.optimization == 'rmsprop':
            self.opt = tf.train.RMSPropOptimizer(self.learning_rate, decay=self.decay, momentum=self.momentum)
        elif self.optimization in ['sgd','momentum']:
            self.opt = tf.train.MomentumOptimizer(self.learning_rate, momentum=self.momentum)
        else:
            print('Cannot recognize optimization algorithm %s' % self.optimization)
            exit(1)

        # compute gradients
        self.gvs = self.opt.compute_gradients(self.loss_op, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

        # clip gradients
        if self.grad_clip is not None:
            gradients, variables = zip(*self.gvs)
            gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
            self.gvs = zip(gradients, variables)

        # apply gradients
        self.step_op = self.opt.apply_gradients(self.gvs, global_step=self.global_step)

        # batch norm helper
        # if self.batch_renorm:
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)


        # summary
        self.merged_summary = tf.summary.merge_all()


    def build_grads(self, layers=[0]):
        ''' Build gradient ops for predictions summed across the sequence for
             each target with respect to some set of layers.

        In
          layers: Optional layer subset list
        '''

        self.grad_layers = layers
        self.grad_ops = []

        for ti in range(self.num_targets):
            grad_ti_op = tf.gradients(self.preds_op[:,:,ti], [self.layer_reprs[li] for li in self.grad_layers])
            self.grad_ops.append(grad_ti_op)


    def build_grads_genes(self, gene_seqs, layers=[0]):
        ''' Build gradient ops for TSS position-specific predictions
             for each target with respect to some set of layers.

        In
          gene_seqs:  GeneSeq list, from which to extract TSS positions
          layers:     Layer subset list.
        '''

        # save layer indexes
        self.grad_layers = layers

        # initialize ops
        self.grad_pos_ops = []

        # determine TSS positions
        tss_pos = set()
        for gene_seq in gene_seqs:
            for tss in gene_seq.tss_list:
                tss_pos.add(tss.seq_bin(width=self.target_pool, pred_buffer=self.batch_buffer))

        # for each position
        preds_length = self.preds_op.shape[1].value
        for pi in range(preds_length):
            self.grad_pos_ops.append([])

            # if it's a TSS position
            if pi in tss_pos:
                # build position-specific, target-specific gradient ops
                for ti in range(self.num_targets):
                    grad_piti_op = tf.gradients(self.preds_op[:,pi,ti], [self.layer_reprs[li] for li in self.grad_layers])
                    self.grad_pos_ops[-1].append(grad_piti_op)


    def drop_rate(self, drop_mult=0.5):
        ''' Drop the optimizer learning rate. '''
        self.opt._lr *= drop_mult


    def gradients(self, sess, batcher, rc=False, shifts=[0], mc_n=0, return_all=False):
        ''' Compute predictions on a test set.

        In
         sess:          TensorFlow session
         batcher:       Batcher class with sequence(s)
         rc:            Average predictions from the forward and reverse complement sequences.
         return_all:    Return all ensemble predictions.

        Out
         layer_grads: [S (sequences) x T (targets) x P (seq position) x U (Units layer i) array] * (L layers)
         layer_reprs: [S (sequences) x P (seq position) x U (Units layer i) array] * (L layers)
         preds:
        '''

        #######################################################################
        # determine ensemble iteration parameters

        ensemble_fwdrc = []
        ensemble_shifts = []
        for shift in shifts:
            ensemble_fwdrc.append(True)
            ensemble_shifts.append(shift)
            if rc:
                ensemble_fwdrc.append(False)
                ensemble_shifts.append(shift)

        if mc_n > 0:
            # setup feed dict
            fd = self.set_mode('test_mc')

        else:
            # setup feed dict
            fd = self.set_mode('test')

            # co-opt the variable to represent
            # iterations per fwdrc/shift.
            mc_n = 1

        # total ensemble predictions
        all_n = mc_n * len(ensemble_fwdrc)


        #######################################################################
        # initialize data structures

        # initialize gradients
        #  (I need a list for layers because the sizes are different within)
        #  (Targets up front, because I need to run their ops one by one)
        layer_reprs = []
        layer_grads = []
        layer_reprs_all = []
        layer_grads_all = []

        for lii in range(len(self.grad_layers)):
            li = self.grad_layers[lii]
            layer_seq_len = self.layer_reprs[li].shape[1].value
            layer_units = self.layer_reprs[li].shape[2].value

            lr = np.zeros((batcher.num_seqs, layer_seq_len, layer_units), dtype='float16')
            layer_reprs.append(lr)

            lg = np.zeros((self.num_targets, batcher.num_seqs, layer_seq_len, layer_units), dtype='float16')
            layer_grads.append(lg)

            if return_all:
                lra = np.zeros((batcher.num_seqs, layer_seq_len, layer_units, all_n), dtype='float16')
                layer_reprs_all.append(lra)

                lgr = np.zeros((self.num_targets, batcher.num_seqs, layer_seq_len, layer_units, all_n), dtype='float16')
                layer_grads_all.append(lgr)


        # initialize predictions
        preds_length = self.preds_op.shape[1].value
        preds = np.zeros((batcher.num_seqs, preds_length, self.num_targets), dtype='float16')

        if return_all:
            preds_all = np.zeros((batcher.num_seqs, preds_length, self.num_targets, all_n), dtype='float16')


        #######################################################################
        # compute

        # sequence index
        si = 0

        # get first batch
        Xb, _, _, Nb = batcher.next()

        while Xb is not None:
            # ensemble predict
            preds_batch, layer_reprs_batch, layer_grads_batch = self._gradients_ensemble(sess, fd, Xb, ensemble_fwdrc, ensemble_shifts, mc_n, return_all=return_all)

            # unpack
            if return_all:
                preds_batch, preds_batch_all = preds_batch
                layer_reprs_batch, layer_reprs_batch_all = layer_reprs_batch
                layer_grads_batch, layer_grads_batch_all = layer_grads_batch

            # accumulate predictions
            preds[si:si+Nb,:,:] = preds_batch[:Nb,:,:]
            if return_all:
                preds_all[si:si+Nb,:,:,:] = preds_batch_all[:Nb,:,:,:]

            # accumulate representations
            for lii in range(len(self.grad_layers)):
                layer_reprs[lii][si:si+Nb] = layer_reprs_batch[lii][:Nb]
                if return_all:
                    layer_reprs_all[lii][si:si+Nb] = layer_reprs_batch_all[lii][:Nb]

            # accumulate gradients
            for lii in range(len(self.grad_layers)):
                for ti in range(self.num_targets):
                    layer_grads[lii][ti,si:si+Nb,:,:] = layer_grads_batch[lii][ti,:Nb,:,:]
                    if return_all:
                        layer_grads_all[lii][ti,si:si+Nb,:,:,:] = layer_grads_batch_all[lii][ti,:Nb,:,:,:]

            # update sequence index
            si += Nb

            # next batch
            Xb, _, _, Nb = batcher.next()

        # reset training batcher
        batcher.reset()

        #######################################################################
        # modify and return

        # move sequences to front
        for lii in range(len(self.grad_layers)):
            layer_grads[lii] = np.transpose(layer_grads[lii], [1,0,2,3])
            if return_all:
                layer_grads_all[lii] = np.transpose(layer_grads_all[lii], [1,0,2,3,4])

        if return_all:
            return layer_grads, layer_reprs, preds, layer_grads_all, layer_reprs_all, preds_all
        else:
            return layer_grads, layer_reprs, preds


    def _gradients_ensemble(self, sess, fd, Xb, ensemble_fwdrc, ensemble_shifts, mc_n, return_var=False, return_all=False):

        # initialize batch predictions
        preds = np.zeros((Xb.shape[0], self.preds_length, self.num_targets), dtype='float32')

        # initialize layer representations and gradients
        layer_reprs = []
        layer_grads = []
        for lii in range(len(self.grad_layers)):
            li = self.grad_layers[lii]
            layer_seq_len = self.layer_reprs[li].shape[1].value
            layer_units = self.layer_reprs[li].shape[2].value

            lr = np.zeros((Xb.shape[0], layer_seq_len, layer_units), dtype='float16')
            layer_reprs.append(lr)

            lg = np.zeros((self.num_targets, Xb.shape[0], layer_seq_len, layer_units), dtype='float16')
            layer_grads.append(lg)


        # initialize variance
        if return_var:
            preds_var = np.zeros(preds.shape, dtype='float32')

            layer_reprs_var = []
            layer_grads_var = []
            for lii in range(len(self.grad_layers)):
                layer_reprs_var.append(np.zeros(layer_reprs.shape, dtype='float32'))
                layer_grads_var.append(np.zeros(layer_grads.shape, dtype='float32'))
        else:
            preds_var = None
            layer_grads_var = [None]*len(self.grad_layers)


        # initialize all-saving arrays
        if return_all:
            all_n = mc_n * len(ensemble_fwdrc)
            preds_all = np.zeros((Xb.shape[0], self.preds_length, self.num_targets, all_n), dtype='float16')

            layer_reprs_all = []
            layer_grads_all = []
            for lii in range(len(self.grad_layers)):
                ls = tuple(list(layer_reprs[lii].shape) + [all_n])
                layer_reprs_all.append(np.zeros(ls, dtype='float16'))

                ls = tuple(list(layer_grads[lii].shape) + [all_n])
                layer_grads_all.append(np.zeros(ls, dtype='float16'))
        else:
            preds_all = None
            layer_grads_all = [None]*len(self.grad_layers)



        running_i = 0

        for ei in range(len(ensemble_fwdrc)):
            # construct sequence
            Xb_ensemble = hot1_augment(Xb, ensemble_fwdrc[ei], ensemble_shifts[ei])

            # update feed dict
            fd[self.inputs] = Xb_ensemble

            # for each monte carlo (or non-mc single) iteration
            for mi in range(mc_n):
                # print('ei=%d, mi=%d, fwdrc=%d, shifts=%d' % (ei, mi, ensemble_fwdrc[ei], ensemble_shifts[ei]), flush=True)

                ##################################################
                # prediction

                # predict
                preds_ei, layer_reprs_ei = sess.run([self.preds_op, self.layer_reprs], feed_dict=fd)

                # reverse
                if ensemble_fwdrc[ei] is False:
                    preds_ei = preds_ei[:,::-1,:]

                # save previous mean
                preds1 = preds

                # update mean
                preds = running_mean(preds1, preds_ei, running_i+1)

                # update variance sum
                if return_var:
                    preds_var = running_varsum(preds_var, preds_ei, preds1, preds)

                # save iteration
                if return_all:
                    preds_all[:,:,:,running_i] = preds_ei[:,:,:]


                ##################################################
                # representations

                for lii in range(len(self.grad_layers)):
                    li = self.grad_layers[lii]

                    # reverse
                    if ensemble_fwdrc[ei] is False:
                        layer_reprs_ei[li] = layer_reprs_ei[li][:,::-1,:]

                    # save previous mean
                    layer_reprs_lii1 = layer_reprs[lii]

                    # update mean
                    layer_reprs[lii] = running_mean(layer_reprs_lii1, layer_reprs_ei[li], running_i+1)

                    # update variance sum
                    if return_var:
                        layer_reprs_var[lii] = running_varsum(layer_reprs_var[lii], layer_reprs_ei[li], layer_reprs_lii1, layer_reprs[lii])

                    # save iteration
                    if return_all:
                        layer_reprs_all[lii][:,:,:,running_i] = layer_reprs_ei[li]


                ##################################################
                # gradients

                # compute gradients for each target individually
                for ti in range(self.num_targets):
                    # compute gradients
                    layer_grads_ti_ei = sess.run(self.grad_ops[ti], feed_dict=fd)

                    for lii in range(len(self.grad_layers)):
                        # reverse
                        if ensemble_fwdrc[ei] is False:
                            layer_grads_ti_ei[lii] = layer_grads_ti_ei[lii][:,::-1,:]

                        # save predious mean
                        layer_grads_lii_ti1 = layer_grads[lii][ti]

                        # update mean
                        layer_grads[lii][ti] = running_mean(layer_grads_lii_ti1, layer_grads_ti_ei[lii], running_i+1)

                        # update variance sum
                        if return_var:
                            layer_grads_var[lii][ti] = running_varsum(layer_grads_var[lii][ti], layer_grads_ti_ei[lii], layer_grads_lii_ti1, layer_grads[lii][ti])

                        # save iteration
                        if return_all:
                            layer_grads_all[lii][ti,:,:,:,running_i] = layer_grads_ti_ei[lii]

                # update running index
                running_i += 1

        if return_var:
            return (preds, preds_var), (layer_reprs, layer_reprs_var), (layer_grads, layer_grads_var)
        elif return_all:
            return (preds, preds_all), (layer_reprs, layer_reprs_all), (layer_grads, layer_grads_all)
        else:
            return preds, layer_reprs, layer_grads


    def gradients_genes(self, sess, batcher, gene_seqs, rc=False):
        ''' Compute predictions on a test set.

        In
         sess:       TensorFlow session
         batcher:    Batcher class with sequence(s)
         gene_seqs:  List of GeneSeq instances specifying gene positions in sequences.
         rc:         Average predictions from the forward and reverse complement sequences.

        Out
         layer_grads: [G (TSSs) x T (targets) x P (seq position) x U (Units layer i) array] * (L layers)
         layer_reprs: [S (sequences) x P (seq position) x U (Units layer i) array] * (L layers)

        Notes
         -Reverse complements aren't implemented yet. They're trickier here, because
          I'd need to build more gradient ops to match the flipped positions.
        '''

        # count TSSs
        tss_num = 0
        for gene_seq in gene_seqs:
            tss_num += len(gene_seq.tss_list)

        # initialize gradients and representations
        #  (I need a list for layers because the sizes are different within)
        #  (TSSxTargets up front, because I need to run their ops one by one)
        layer_grads = []
        layer_reprs = []
        for lii in range(len(self.grad_layers)):
            li = self.grad_layers[lii]
            layer_seq_len = self.layer_reprs[li].shape[1].value
            layer_units = self.layer_reprs[li].shape[2].value

            # gradients
            lg = np.zeros((tss_num, self.num_targets, layer_seq_len, layer_units), dtype='float16')
            layer_grads.append(lg)

            # representations
            lr = np.zeros((batcher.num_seqs, layer_seq_len, layer_units), dtype='float16')
            layer_reprs.append(lr)

        # setup feed dict for dropout
        fd = self.set_mode('test')

        # TSS index
        tss_i = 0

        # sequence index
        si = 0

        # get first batch
        Xb, _, _, Nb = batcher.next()

        while Xb is not None:
            # update feed dict
            fd[self.inputs] = Xb

            # predict
            reprs_batch, _ = sess.run([self.layer_reprs, self.preds_op], feed_dict=fd)

            # save representations
            for lii in range(len(self.grad_layers)):
                li = self.grad_layers[lii]
                layer_reprs[lii][si:si+Nb] = reprs_batch[li][:Nb]

            # compute gradients for each TSS position individually
            for bi in range(Nb):
                for tss in gene_seqs[si+bi].tss_list:
                    # get TSS prediction bin position
                    pi = tss.seq_bin(width=self.target_pool, pred_buffer=self.batch_buffer)

                    for ti in range(self.num_targets):
                        # compute gradients over all positions
                        grads_batch = sess.run(self.grad_pos_ops[pi][ti], feed_dict=fd)

                        # accumulate gradients
                        for lii in range(len(self.grad_layers)):
                            layer_grads[lii][tss_i,ti,:,:] = grads_batch[lii][bi]

                    # update TSS index
                    tss_i += 1

            # update sequence index
            si += Nb

            # next batch
            Xb, _, _, Nb = batcher.next()

        # reset training batcher
        batcher.reset()

        return layer_grads, layer_reprs


    def hidden(self, sess, batcher, layers=None):
        ''' Compute hidden representations for a test set. '''

        if layers is None:
            layers = list(range(self.cnn_layers))

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


    def _predict_ensemble(self, sess, fd, Xb, ensemble_fwdrc, ensemble_shifts, mc_n, ds_indexes=None, target_indexes=None, return_var=False, return_all=False, penultimate=False):

        # determine predictions length
        preds_length = self.preds_length
        if ds_indexes is not None:
            preds_length = len(ds_indexes)

        # determine num targets
        if penultimate:
            num_targets = self.cnn_filters[-1]
        else:
            num_targets = self.num_targets
            if target_indexes is not None:
                num_targets = len(target_indexes)

        # initialize batch predictions
        preds_batch = np.zeros((Xb.shape[0], preds_length, num_targets), dtype='float32')

        if return_var:
            preds_batch_var = np.zeros(preds_batch.shape, dtype='float32')
        else:
            preds_batch_var = None

        if return_all:
            all_n = mc_n * len(ensemble_fwdrc)
            preds_all = np.zeros((Xb.shape[0], preds_length, num_targets, all_n), dtype='float16')
        else:
            preds_all = None

        running_i = 0

        for ei in range(len(ensemble_fwdrc)):
            # construct sequence
            Xb_ensemble = hot1_augment(Xb, ensemble_fwdrc[ei], ensemble_shifts[ei])

            # update feed dict
            fd[self.inputs] = Xb_ensemble

            # for each monte carlo (or non-mc single) iteration
            for mi in range(mc_n):
                # print('ei=%d, mi=%d, fwdrc=%d, shifts=%d' % (ei, mi, ensemble_fwdrc[ei], ensemble_shifts[ei]), flush=True)

                # predict
                if penultimate:
                    preds_ei = sess.run(self.penultimate_op, feed_dict=fd)
                else:
                    preds_ei = sess.run(self.preds_op, feed_dict=fd)

                # reverse
                if ensemble_fwdrc[ei] is False:
                    preds_ei = preds_ei[:,::-1,:]

                # down-sample
                if ds_indexes is not None:
                    preds_ei = preds_ei[:,ds_indexes,:]
                if target_indexes is not None:
                    preds_ei = preds_ei[:,:,target_indexes]

                # save previous mean
                preds_batch1 = preds_batch

                # update mean
                preds_batch = running_mean(preds_batch1, preds_ei, running_i+1)

                # update variance sum
                if return_var:
                    preds_batch_var = running_varsum(preds_batch_var, preds_ei, preds_batch1, preds_batch)

                # save iteration
                if return_all:
                    preds_all[:,:,:,running_i] = preds_ei[:,:,:]

                # update running index
                running_i += 1

        return preds_batch, preds_batch_var, preds_all


    def predict(self, sess, batcher, rc=False, shifts=[0], mc_n=0, target_indexes=None, return_var=False, return_all=False, down_sample=1, penultimate=False):
        ''' Compute predictions on a test set.

        In
         sess:           TensorFlow session
         batcher:        Batcher class with transcript-covering sequences.
         rc:             Average predictions from the forward and reverse complement sequences.
         shifts:         Average predictions from sequence shifts left/right.
         mc_n:           Monte Carlo iterations per rc/shift.
         target_indexes: Optional target subset list
         return_var:     Return variance estimates
         down_sample:    Int specifying to consider uniformly spaced sampled positions
         penultimate:    Predict the penultimate layer.

        Out
         preds: S (sequences) x L (unbuffered length) x T (targets) array
        '''

        # uniformly sample indexes
        ds_indexes = None
        preds_length = self.preds_length
        if down_sample != 1:
            ds_indexes = np.arange(0, self.preds_length, down_sample)
            preds_length = len(ds_indexes)

        # initialize prediction arrays
        if penultimate:
            num_targets = self.cnn_filters[-1]
        else:
            num_targets = self.num_targets
            if target_indexes is not None:
                num_targets = len(target_indexes)

        # determine ensemble iteration parameters
        ensemble_fwdrc = []
        ensemble_shifts = []
        for shift in shifts:
            ensemble_fwdrc.append(True)
            ensemble_shifts.append(shift)
            if rc:
                ensemble_fwdrc.append(False)
                ensemble_shifts.append(shift)

        if mc_n > 0:
            # setup feed dict
            fd = self.set_mode('test_mc')

        else:
            # setup feed dict
            fd = self.set_mode('test')

            # co-opt the variable to represent
            # iterations per fwdrc/shift.
            mc_n = 1

        # total ensemble predictions
        all_n = mc_n * len(ensemble_fwdrc)

        # initialize prediction data structures
        preds = np.zeros((batcher.num_seqs, preds_length, num_targets), dtype='float16')
        if return_var:
            if all_n == 1:
                print('Cannot return prediction variance. Add rc, shifts, or mc.', file=sys.stderr)
                exit(1)
            preds_var = np.zeros((batcher.num_seqs, preds_length, num_targets), dtype='float16')
        if return_all:
            preds_all = np.zeros((batcher.num_seqs, preds_length, num_targets, all_n), dtype='float16')

        # sequence index
        si = 0

        # get first batch
        Xb, _, _, Nb = batcher.next()

        while Xb is not None:
            # make ensemble predictions
            preds_batch, preds_batch_var, preds_batch_all = self._predict_ensemble(sess, fd, Xb, ensemble_fwdrc, ensemble_shifts, mc_n, ds_indexes, target_indexes, return_var, return_all, penultimate)

            # accumulate predictions
            preds[si:si+Nb,:,:] = preds_batch[:Nb,:,:]
            if return_var:
                preds_var[si:si+Nb,:,:] = preds_batch_var[:Nb,:,:] / (all_n-1)
            if return_all:
                preds_all[si:si+Nb,:,:,:] = preds_batch_all[:Nb,:,:,:]

            # update sequence index
            si += Nb

            # next batch
            Xb, _, _, Nb = batcher.next()

        # reset batcher
        batcher.reset()

        if return_var:
            if return_all:
                return preds, preds_var, preds_all
            else:
                return preds, preds_var
        else:
            return preds


    def predict_genes(self, sess, batcher, gene_seqs, rc=False, shifts=[0], mc_n=0, target_indexes=None, tss_radius=0, penultimate=False):
        ''' Compute predictions on a test set.

        In
         sess:            TensorFlow session
         batcher:         Batcher class with transcript-covering sequences
         gene_seqs        List of GeneSeq instances specifying gene positions in sequences.
         rc:              Average predictions from the forward and reverse complement sequences.
         shifts:          Average predictions from sequence shifts left/right.
         mc_n:            Monte Carlo iterations per rc/shift.
         target_indexes:  Optional target subset list
         tss_radius:      Radius of bins to quantify TSS.
         penultimate:     Predict the penultimate layer.

        Out
         transcript_preds: G (gene transcripts) X T (targets) array
        '''

        # predict gene sequences
        gseq_preds = self.predict(sess, batcher, rc=rc, shifts=shifts, mc_n=mc_n, target_indexes=target_indexes, penultimate=penultimate)

        # count TSSs
        tss_num = 0
        for gene_seq in gene_seqs:
            tss_num += len(gene_seq.tss_list)

        # initialize TSS predictions
        tss_preds = np.zeros((tss_num, gseq_preds.shape[-1]), dtype='float16')

        # slice TSSs
        tss_i = 0
        for si in range(len(gene_seqs)):
            for tss in gene_seqs[si].tss_list:
                bi = tss.seq_bin(width=self.target_pool, pred_buffer=self.batch_buffer)
                tss_preds[tss_i,:] = gseq_preds[si,bi-tss_radius:bi+1+tss_radius,:].sum(axis=0)
                tss_i += 1

        # reset batcher
        batcher.reset()

        return tss_preds


    def set_mode(self, mode):
        ''' Construct a feed dictionary to specify the model's mode. '''
        fd = {}

        if mode in ['train', 'training']:
            fd[self.is_training] = True
            for li in range(self.cnn_layers):
                fd[self.cnn_dropout_ph[li]] = self.cnn_dropout[li]

        elif mode in ['test', 'testing', 'evaluate']:
            fd[self.is_training] = False
            for li in range(self.cnn_layers):
                fd[self.cnn_dropout_ph[li]] = 0

        elif mode in ['test_mc', 'testing_mc', 'evaluate_mc', 'mc_test', 'mc_testing', 'mc_evaluate']:
            fd[self.is_training] = False
            for li in range(self.cnn_layers):
                fd[self.cnn_dropout_ph[li]] = self.cnn_dropout[li]

        else:
            print('Cannot recognize mode %s' % mode)
            exit(1)

        return fd


    def set_params(self, job):
        ''' Set model parameters. '''

        ###################################################
        # data attributes
        ###################################################
        self.seq_depth = job.get('seq_depth', 4)
        self.num_targets = job['num_targets']
        self.target_classes = job.get('target_classes', 1)
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
        self.momentum = job.get('momentum', 0)
        self.decay = job.get('decay', 0.9)
        self.optimization = job.get('optimization', 'adam').lower()
        self.grad_clip = job.get('grad_clip', None)

        ###################################################
        # CNN params
        ###################################################
        self.cnn_filters = np.atleast_1d(job.get('cnn_filters', []))
        self.cnn_filter_sizes = np.atleast_1d(job.get('cnn_filter_sizes', []))
        self.cnn_layers = len(self.cnn_filters)

        self.cnn_pool = layer_extend(job.get('cnn_pool', []), 1, self.cnn_layers)
        self.cnn_strides = layer_extend(job.get('cnn_strides', []), 1, self.cnn_layers)
        self.cnn_dense = layer_extend(job.get('cnn_dense', []), False, self.cnn_layers)
        self.cnn_dilation = layer_extend(job.get('cnn_dilation', []), 1, self.cnn_layers)

        ###################################################
        # regularization
        ###################################################
        self.cnn_dropout = layer_extend(job.get('cnn_dropout', []), 0, self.cnn_layers)
        self.cnn_l2 = layer_extend(job.get('cnn_l2', []), 0, self.cnn_layers)

        self.final_l1 = job.get('final_l1', 0)

        self.batch_renorm = bool(job.get('batch_renorm', False))
        self.batch_renorm = bool(job.get('renorm', self.batch_renorm))

        ###################################################
        # loss
        ###################################################
        self.link = job.get('link', 'exp_linear')
        self.loss = job.get('loss', 'poisson')
        self.target_clip = job.get('target_clip', None)
        self.target_sqrt = bool(job.get('target_sqrt', False))


    def test(self, sess, batcher, rc=False, shifts=[0], mc_n=0):
        ''' Compute model accuracy on a test set.

        Args:
          sess:         TensorFlow session
          batcher:      Batcher object to provide data
          rc:             Average predictions from the forward and reverse complement sequences.
          shifts:         Average predictions from sequence shifts left/right.
          mc_n:           Monte Carlo iterations per rc/shift.

        Returns:
          acc:          Accuracy object
        '''

        # determine ensemble iteration parameters
        ensemble_fwdrc = []
        ensemble_shifts = []
        for shift in shifts:
            ensemble_fwdrc.append(True)
            ensemble_shifts.append(shift)
            if rc:
                ensemble_fwdrc.append(False)
                ensemble_shifts.append(shift)

        if mc_n > 0:
            # setup feed dict
            fd = self.set_mode('test_mc')

        else:
            # setup feed dict
            fd = self.set_mode('test')

            # co-opt the variable to represent
            # iterations per fwdrc/shift.
            mc_n = 1

        # initialize prediction and target arrays
        preds = np.zeros((batcher.num_seqs, self.preds_length, self.num_targets), dtype='float16')

        targets = np.zeros((batcher.num_seqs, self.preds_length, self.num_targets), dtype='float16')
        targets_na = np.zeros((batcher.num_seqs, self.preds_length), dtype='bool')

        batch_losses = []
        batch_target_losses = []

        # sequence index
        si = 0

        # get first batch
        Xb, Yb, NAb, Nb = batcher.next()

        while Xb is not None:
            # make ensemble predictions
            preds_batch, preds_batch_var, preds_all = self._predict_ensemble(sess, fd, Xb, ensemble_fwdrc, ensemble_shifts, mc_n)

            # add target info
            fd[self.targets] = Yb
            fd[self.targets_na] = NAb

            # recompute loss w/ ensembled prediction
            fd[self.preds_adhoc] = preds_batch
            targets_batch, loss_batch, target_losses_batch = sess.run([self.targets_op, self.loss_adhoc, self.target_losses_adhoc], feed_dict=fd)

            # accumulate predictions and targets
            if preds_batch.ndim == 3:
                preds[si:si+Nb,:,:] = preds_batch[:Nb,:,:]
                targets[si:si+Nb,:,:] = targets_batch[:Nb,:,:]

            else:
                for qi in range(preds_batch.shape[3]):
                    # TEMP, ideally this will be in the HDF5 and set previously
                    self.quantile_means = np.geomspace(0.1, 256, 16)

                    # softmax
                    preds_batch_norm = np.expand_dims(np.sum(np.exp(preds_batch[:Nb,:,:,:]),axis=3),axis=3)
                    pred_probs_batch = np.exp(preds_batch[:Nb,:,:,:]) / preds_batch_norm

                    # expectation over quantile medians
                    preds[si:si+Nb,:,:] = np.dot(pred_probs_batch, self.quantile_means)

                    # compare to quantile median
                    targets[si:si+Nb,:,:] = self.quantile_means[targets_batch[:Nb,:,:]-1]

            # accumulate loss
            batch_losses.append(loss_batch)
            batch_target_losses.append(target_losses_batch)

            # update sequence index
            si += Nb

            # next batch
            Xb, Yb, NAb, Nb = batcher.next()

        # reset batcher
        batcher.reset()

        # mean across batches
        batch_losses = np.mean(batch_losses)
        batch_target_losses = np.array(batch_target_losses).mean(axis=0)

        # instantiate accuracy object
        acc = basenji.accuracy.Accuracy(targets, preds, targets_na, batch_losses, batch_target_losses)

        return acc


    def train_epoch(self, sess, batcher, fwdrc=True, shift=0, sum_writer=None):
        ''' Execute one training epoch '''

        # initialize training loss
        train_loss = []

        # setup feed dict
        fd = self.set_mode('train')

        # get first batch
        Xb, Yb, NAb, Nb = batcher.next(fwdrc, shift)

        while Xb is not None and Nb == self.batch_size:
            # update feed dict
            fd[self.inputs] = Xb
            fd[self.targets] = Yb
            fd[self.targets_na] = NAb

            run_returns = sess.run([self.global_step, self.merged_summary, self.loss_op, self.step_op]+self.update_ops, feed_dict=fd)
            gstep, summary, loss_batch = run_returns[:3]

            # pull gradients
            # gvs_batch = sess.run([g for (g,v) in self.gvs if g is not None], feed_dict=fd)

            # add summary
            if sum_writer is not None:
                sum_writer.add_summary(summary, gstep)

            # accumulate loss
            # avail_sum = np.logical_not(NAb[:Nb,:]).sum()
            # train_loss.append(loss_batch / avail_sum)
            train_loss.append(loss_batch)

            # next batch
            Xb, Yb, NAb, Nb = batcher.next(fwdrc, shift)

        # reset training batcher
        batcher.reset()

        return np.mean(train_loss), self.global_step


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

def running_mean(u_k1, x_k, k):
    return u_k1 + (x_k - u_k1) / k

def running_varsum(v_k1, x_k, m_k1, m_k):
    ''' Computing the running variance numerator.

    Ref: https://www.johndcook.com/blog/standard_deviation/
    '''
    return v_k1 + (x_k - m_k1)*(x_k - m_k)
