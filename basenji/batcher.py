#!/usr/bin/env python

import h5py
import numpy as np
import random

class Batcher:
    ''' Batcher

    Class to manage batches.
    '''
    def __init__(self, Xf, Yf=None, batch_size=8):
        self.Xf = Xf
        self.num_seqs = self.Xf.shape[0]
        self.seq_len = self.Xf.shape[1]
        self.seq_depth = self.Xf.shape[2]

        self.Yf = Yf
        self.num_targets = self.Yf.shape[2]

        self.batch_size = batch_size

        self.reset()


    def next(self):
        ''' Load the next batch from the HDF5. '''
        Xb = None
        Yb = None
        Nb = 0

        stop = self.start + self.batch_size
        if stop <= self.num_seqs:
            # full batch

            # copy data
            Xb = np.array(self.Xf[self.start:stop], dtype='float32')
            if self.Yf is not None:
                Yb = np.nan_to_num(np.array(self.Yf[self.start:stop], dtype='float32'))

            # specify full batch
            Nb = self.batch_size

        elif self.start < self.num_seqs:
            # partial batch

            # initialize full batch of zeros
            Xb = np.zeros((self.batch_size, self.Xf.shape[1], self.Xf.shape[2]), dtype='float32')
            if self.Yf is not None:
                Yb = np.zeros((self.batch_size, self.Yf.shape[1], self.Yf.shape[2]), dtype='float32')

            # copy data
            Nb = self.num_seqs - self.start
            Xb[:Nb] = self.Xf[self.start:self.start+Nb]
            if self.Yf is not None:
                Yb[:Nb] = np.nan_to_num(self.Yf[self.start:self.start+Nb])

        # update start
        self.start = stop

        return Xb, Yb, Nb


    def next_shuffle(self):
        ''' Load the next batch from the HDF5. '''
        Xb = None
        Yb = None
        Nb = 0

        stop = self.start + self.batch_size
        if stop <= self.num_seqs:
            # full batch

            # initialize full batch of zeros
            Xb = np.zeros((self.batch_size, self.Xf.shape[1], self.Xf.shape[2]), dtype='float32')
            if self.Yf is not None:
                Yb = np.zeros((self.batch_size, self.Yf.shape[1], self.Yf.shape[2]), dtype='float32')

            # copy data
            for i in range(self.batch_size):
                si = self.shuffled[self.start+i]
                Xb[i] = self.Xf[si]
                if self.Yf is not None:
                    Yb[i] = np.nan_to_num(self.Yf[si])

            # specify full batch
            Nb = self.batch_size

        # update start
        self.start = stop

        return Xb, Yb, Nb


    def reset(self):
        self.start = 0

    def shuffle(self):
        self.shuffled = list(range(self.num_seqs))
        random.shuffle(self.shuffled)
