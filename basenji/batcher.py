#!/usr/bin/env python

import h5py
import numpy as np

class Batcher:
    ''' Batcher

    Class to manage batches.
    '''
    def __init__(self, Xf, Yf, batch_size=8):
        self.Xf = Xf
        self.num_seqs = self.Xf.shape[0]
        self.seq_length = self.Xf.shape[1]
        self.seq_depth = self.Xf.shape[2]

        self.Yf = Yf
        self.num_targets = self.Yf.shape[1]

        self.batch_size = batch_size

        self.reset()


    def next(self):
        ''' Load the next batch from the HDF5

        I'm making an incorrect simplifying assumption that grab the whole
        sequence and it'll match the batch length, but ultimately I'll need
        to be smarter here.
        '''
        Xb = None
        Yb = None

        stop = self.start + self.batch_size
        if stop <= self.num_seqs:
            # copy data
            Xb = self.Xf[self.start:stop,:,:]
            Yb = self.Yf[self.start:stop,:]

            # update start
            self.start = stop

        return Xb, Yb


    def reset(self):
        self.start = 0
