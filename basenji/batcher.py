#!/usr/bin/env python

import h5py
import numpy as np

class Batcher:
    ''' Batcher

    Class to manage batches.
    '''
    def __init__(self, Xf, Yf=None, batch_size=8):
        self.Xf = Xf
        self.num_seqs = self.Xf.shape[0]

        self.Yf = Yf

        self.batch_size = batch_size

        self.reset()


    def next(self):
        ''' Load the next batch from the HDF5. '''
        Xb = None
        Yb = None

        stop = self.start + self.batch_size
        if stop <= self.num_seqs:
            # copy data
            Xb = np.array(self.Xf[self.start:stop], dtype='float32')
            if self.Yf is not None:
                Yb = np.nan_to_num(np.array(self.Yf[self.start:stop], dtype='float32'))

            # update start
            self.start = stop

        if self.Yf is None:
            return Xb
        else:
            return Xb, Yb


    def reset(self):
        self.start = 0
