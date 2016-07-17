#!/usr/bin/env python

import h5py
import numpy as np
import random

class BatcherT:
    ''' BatcherT

    Class to manage batches for nucleotide target prediction.
    '''
    def __init__(self, Yf, batch_size=64, shuffle=False):
        self.Yf = Yf
        self.num_nts = self.Yf.shape[0]
        self.num_targets = self.Yf.shape[1]

        self.batch_size = batch_size

        self.shuffle = shuffle

        self.reset()


    def next(self):
        ''' Load the next batch from the HDF5. '''
        Yb = None
        Nb = 0

        stop = self.start + self.batch_size
        if stop <= self.num_nts:
            # full batch

            # initialize
            Yb = np.zeros((self.batch_size, self.num_targets), dtype='float32')

            # specify full batch
            Nb = self.batch_size

            # copy data
            for i in range(Nb):
                si = self.order[self.start+i]
                Yb[i] = self.Yf[si]

        elif self.start < self.num_nts:
            # initialize full batch of zeros
            Yb = np.zeros((self.batch_size, self.num_targets), dtype='float32')

            # specify partial batch
            Nb = self.num_nts - self.start

            # copy data
            for i in range(Nb):
                si = self.order[self.start+i]
                Yb[i] = self.Yf[si]

        # update start
        self.start = stop

        return Yb, Nb


    def reset(self):
        self.start = 0
        self.order = list(range(self.num_nts))
        if self.shuffle:
            random.shuffle(self.order)
