#!/usr/bin/env python
import sys
import time

import numpy as np

from sklearn.metrics import r2_score
import tensorflow as tf


class AE:
    def __init__(self, job):
        ###################################################
        # model parameters and placeholders
        ###################################################
        self.set_params(job)

        # batches
        self.y = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_targets))

        # training indicator
        self.train = tf.placeholder(tf.float32, shape=[])

        # dropout rates
        # self.dropout_ph = []
        # for li in range(self.layers):
        #     self.dropout_ph.append(tf.placeholder(tf.float32))

        ###################################
        # encoder
        ###################################
        layer_rep = self.y
        for li in range(len(self.encoder_units)):
            with tf.variable_scope('encode%d' % li) as vs:
                # initialize parameters
                weights = tf.get_variable(name='weights', shape=(layer_rep.get_shape()[1], self.encoder_units[li]), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(uniform=True))
                bias = tf.Variable(tf.zeros(self.encoder_units[li]), name='bias')

                # linear transformation
                layer_rep = tf.matmul(layer_rep, weights) + bias

                # batch normalization
                layer_rep = tf.contrib.layers.python.layers.batch_norm(layer_rep, center=True, scale=True, is_training=True)

                # nonlinearity
                layer_rep = tf.nn.relu(layer_rep)

        ###################################
        # latent
        ###################################
        # mu
        with tf.variable_scope('latent_mu') as vs:
            # initialize parameters
            weights = tf.get_variable(name='weights', shape=(layer_rep.get_shape()[1], self.latent_dim), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(uniform=True))
            bias = tf.Variable(tf.zeros(self.latent_dim), name='bias')

            # compute
            self.mu = tf.matmul(layer_rep, weights) + bias

        # std
        if self.variational:
            with tf.variable_scope('latent_std') as vs:
                # initialize parameters
                weights = tf.get_variable(name='weights', shape=(layer_rep.get_shape()[1], self.latent_dim), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(uniform=True))
                bias = tf.Variable(tf.zeros(self.latent_dim), name='bias')

                # compute
                logvar = tf.matmul(layer_rep, weights) + bias
                self.std = tf.exp(0.5*logvar)

        # sample latent variables
        x = self.mu
        if self.variational:
            epsilon = tf.random_normal(tf.shape(logvar), stddev=0.005, dtype=tf.float32, name='epsilon')
            x += tf.scalar_mul(self.train, tf.mul(self.std, epsilon))

        ###################################
        # decoder
        ###################################
        layer_rep = x
        for li in range(len(self.decoder_units)):
            with tf.variable_scope('decode%d' % li) as vs:
                # initialize parameters
                weights = tf.get_variable(name='weights', shape=(layer_rep.get_shape()[1], self.decoder_units[li]), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(uniform=True))
                bias = tf.Variable(tf.zeros(self.decoder_units[li]), name='bias')

                # linear transformation
                layer_rep = tf.matmul(layer_rep, weights) + bias

                # batch normalization
                # layer_rep = tf.contrib.layers.python.layers.batch_norm(layer_rep, center=True, scale=True, is_training=True)

                # nonlinearity
                layer_rep = tf.nn.relu(layer_rep)

        with tf.variable_scope('reconstruct') as vs:
            # initialize parameters
            weights = tf.get_variable(name='weights', shape=(layer_rep.get_shape()[1], self.num_targets), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(uniform=True))
            bias = tf.Variable(tf.zeros(self.num_targets), name='bias')

            # compute
            self.preds = tf.matmul(layer_rep, weights) + bias

        ###################################
        # loss
        ###################################
        # L2 loss
        self.loss = tf.reduce_mean(tf.squared_difference(self.y, self.preds), reduction_indices=1)

        # KL divergence
        if self.variational:
            kld = -0.5 * tf.reduce_sum(1 + logvar - tf.square(self.mu) - tf.exp(logvar), reduction_indices=1)
            self.loss += kld

        # finalize
        self.loss = tf.reduce_mean(self.loss)

        # print variables
        for v in tf.all_variables():
            print(v.name, v.get_shape())

        ###################################
        # optimization
        ###################################
        self.opt = tf.train.AdamOptimizer(self.learning_rate, self.adam_beta1, self.adam_beta2, self.adam_eps)

        self.step_op = self.opt.minimize(self.loss)


    def drop_rate(self):
        ''' Drop the optimizer learning rate. '''
        self.opt._lr /= 1.5


    def set_params(self, job):
        ''' Set RNN parameters. '''

        ###################################################
        # data attributes
        ###################################################
        self.num_targets = job['num_targets']

        ###################################################
        # batching
        ###################################################
        self.batch_size = job.get('batch_size', 64)

        ###################################################
        # training
        ###################################################
        self.learning_rate = job.get('learning_rate', 0.001)
        self.adam_beta1 = job.get('adam_beta1', 0.9)
        self.adam_beta2 = job.get('adam_beta2', 0.999)
        self.adam_eps = job.get('adam_eps', 1e-8)
        self.optimization = job.get('optimization', 'adam').lower()

        ###################################################
        # neural net
        ###################################################
        self.variational = bool(job.get('variational', 0))
        self.encoder_units = np.atleast_1d(job.get('encoder_units', [300]))
        self.decoder_units = np.atleast_1d(job.get('decoder_units', self.encoder_units))
        self.latent_dim = job.get('latent_dim', 20)

        ###################################################
        # regularization?
        ###################################################
        self.early_stop = job.get('early_stop', 10)
        self.dropout = layer_extend(job.get('dropout', []), 0, len(self.encoder_units))


    def latent(self, sess, batcher):
        ''' Compute latent representation '''

        latents = []

        # setup feed dict for dropout
        fd = {self.train:0}

        # get first batch
        Yb, Nb = batcher.next()

        while Yb is not None:
            # update feed dict
            fd[self.y] = Yb

            # measure batch loss
            latents_batch = sess.run(self.mu, feed_dict=fd)

            # accumulate predictions and targets
            latents.append(latents_batch[:Nb])

            # next batch
            Yb, Nb = batcher.next()

        # reset batcher
        batcher.reset()

        # accumulate predictions
        latents = np.vstack(latents)

        return latents


    def predict(self, sess, batcher):
        ''' Compute predictions on a test set. '''

        preds = []

        # setup feed dict for dropout
        fd = {self.train:0}

        # get first batch
        Yb, Nb = batcher.next()

        while Yb is not None:
            # update feed dict
            fd[self.y] = Yb

            # measure batch loss
            preds_batch = sess.run(self.preds, feed_dict=fd)

            # accumulate predictions and targets
            preds.append(preds_batch[:Nb])

            # next batch
            Yb, Nb = batcher.next()

        # reset batcher
        batcher.reset()

        # accumulate predictions
        preds = np.vstack(preds)

        return preds


    def test(self, sess, batcher, return_preds=False):
        ''' Compute model accuracy on a test set. '''

        batch_losses = []
        preds = []
        targets = []

        # setup feed dict for dropout
        fd = {self.train:0}
        # for li in range(self.layers):
        #     fd[self.dropout_ph[li]] = 0

        # get first batch
        Yb, Nb = batcher.next()

        while Yb is not None:
            # update feed dict
            fd[self.y] = Yb

            # measure batch loss
            mu_batch, preds_batch, loss_batch = sess.run([self.mu, self.preds, self.loss], feed_dict=fd)

            # accumulate loss
            batch_losses.append(loss_batch)

            # accumulate predictions and targets
            preds.append(preds_batch[:Nb])
            targets.append(Yb[:Nb])

            # next batch
            Yb, Nb = batcher.next()

        # reset batcher
        batcher.reset()

        # accumulate predictions
        preds = np.vstack(preds)
        targets = np.vstack(targets)

        # compute R2 per target
        r2 = np.zeros(self.num_targets)
        for ti in range(self.num_targets):
            # index
            preds_ti = preds[:,ti]
            targets_ti = targets[:,ti]

            # compute R2
            tmean = targets_ti.mean(dtype='float64')
            tvar = (targets_ti-tmean).var(dtype='float64')
            pvar = (targets_ti-preds_ti).var(dtype='float64')
            r2[ti] = 1.0 - pvar/tvar

        if return_preds:
            return np.mean(batch_losses), np.mean(r2), preds
        else:
            return np.mean(batch_losses), np.mean(r2)


    def train_epoch(self, sess, batcher):
        ''' Execute one training epoch '''

        # initialize training loss
        train_loss = []

        # setup feed dict for dropout
        fd = {self.train:1}
        # for li in range(self.layers):
        #     fd[self.dropout_ph[li]] = self.dropout[li]

        # get first batch
        Yb, Nb = batcher.next()

        while Yb is not None and Nb == self.batch_size:
            # update feed dict
            fd[self.y] = Yb

            # run step
            loss_batch, _ = sess.run([self.loss, self.step_op], feed_dict=fd)

            # accumulate loss
            train_loss.append(loss_batch)

            # next batch
            Yb, Nb = batcher.next()

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
