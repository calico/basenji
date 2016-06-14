#!/usr/bin/env python
import h5py
import numpy as np

class Batcher:
    ''' Batcher
    Class to manage batches.
    '''
    def __init__(self, Xf, Yf, batch_size=8, batch_length=16000):
        self.Xf = Xf
        self.seq_length = self.Xf.shape[0]
        self.seq_depth = self.Xf.shape[1]

        self.Yf = Yf
        self.num_targets = self.Yf.shape[1]

        self.batch_size = batch_size
        self.batch_length = batch_length

        # divide sequence
        self.epoch_length = self.seq_length // self.batch_size

        # determine epoch starts
        self.epoch_starts = np.zeros(self.batch_size, dtype='int')
        for ei in range(1,self.batch_size):
            self.epoch_starts[ei] = self.epoch_starts[ei-1] + self.epoch_length

        self.reset()


    def next(self):
        Xb = None
        Yb = None

        stops = self.starts + self.batch_length
        if stops[0] < self.epoch_starts[1]:
            # initialize batch tensors
            Xb = np.zeros((self.batch_size, self.batch_length, self.seq_depth))
            Yb = np.zeros((self.batch_size, self.batch_length, self.num_targets))

            # copy batch
            for bi in range(self.batch_size):
                Xb[bi,:,:] = self.Xf[self.starts[bi]:stops[bi],:]
                Yb[bi,:,:] = self.Yf[self.starts[bi]:stops[bi],:]

            # update starts for next batch
            for bi in range(self.batch_size):
                self.starts[bi] += self.batch_length

        return Xb, Yb


    def reset(self):
        self.starts = np.copy(self.epoch_starts)
