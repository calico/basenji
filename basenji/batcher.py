#!/usr/bin/env python

import h5py
import numpy as np
import random

class Batcher:
    ''' Batcher

    Class to manage batches.
    '''
    def __init__(self, Xf, Yf=None, batch_size=64, shuffle=False):
        self.Xf = Xf
        self.num_seqs = self.Xf.shape[0]
        self.seq_len = self.Xf.shape[1]
        self.seq_depth = self.Xf.shape[2]

        self.Yf = Yf
        self.num_targets = self.Yf.shape[2]

        self.batch_size = batch_size

        self.shuffle = shuffle

        self.reset()


    def next_float(self):
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


    def next(self):
        ''' Load the next batch from the HDF5. '''
        Xb = None
        Yb = None
        Nb = 0

        stop = self.start + self.batch_size
        if stop <= self.num_seqs:
            # full batch

            # initialize
            Xb = np.zeros((self.batch_size, self.seq_len, self.seq_depth), dtype='float32')
            if self.Yf is not None:
                Yb = np.zeros((self.batch_size, self.seq_len, self.num_targets), dtype='float32')

            # copy data
            for i in range(self.batch_size):
                si = self.order[self.start+i]
                Xb[i] = self.Xf[si]

                # fix N positions
                Xbi_n = (Xb[i].sum(axis=1) == 0)
                Xb[i] = Xb[i] + (1/self.seq_depth)*Xbi_n.repeat(self.seq_depth).reshape(self.seq_len,self.seq_depth)

                if self.Yf is not None:
                    Yb[i] = np.nan_to_num(self.Yf[si])

            # specify full batch
            Nb = self.batch_size

        # update start
        self.start = stop

        return Xb, Yb, Nb


    def reset(self):
        self.start = 0
        self.order = list(range(self.num_seqs))
        if self.shuffle:
            random.shuffle(self.order)


class BatcherF:
    ''' BatcherF

    Class to manage batches of data in fourier space.
    '''
    def __init__(self, Xf, Yf_real, Yf_imag, batch_size=64, shuffle=False):
        self.Xf = Xf
        self.num_seqs = self.Xf.shape[0]
        self.seq_len = self.Xf.shape[1]
        self.seq_depth = self.Xf.shape[2]

        self.Yf_real = Yf_real
        self.Yf_imag = Yf_imag
        self.num_targets = self.Yf.shape[2]

        self.batch_size = batch_size

        self.shuffle = shuffle

        self.reset()


    def next(self):
        ''' Load the next batch from the HDF5. '''
        Xb = None
        Yb = None
        Nb = 0

        stop = self.start + self.batch_size
        if stop <= self.num_seqs:
            # full batch

            # initialize
            Xb = np.zeros((self.batch_size, self.seq_len, self.seq_depth), dtype='float32')
            if self.Yf is not None:
                Yb = np.zeros((self.batch_size, self.seq_len, self.num_targets), dtype='float32')

            # copy data
            for i in range(self.batch_size):
                si = self.order[self.start+i]
                Xb[i] = self.Xf[si]

                # fix N positions
                Xbi_n = (Xb[i].sum(axis=1) == 0)
                Xb[i] = Xb[i] + (1/self.seq_depth)*Xbi_n.repeat(self.seq_depth).reshape(self.seq_len,self.seq_depth)

                # inverse fourier transform
                ybi_fourier = self.Yf_real[si] + self.Yf_imag[si]*1j
                for ti in range(self.num_targets):
                    Yb[i,:,ti] = np.fft.irfft(ybi_fourier[:,ti], self.seq_len)

            # specify full batch
            Nb = self.batch_size

        # update start
        self.start = stop

        return Xb, Yb, Nb


    def reset(self):
        self.start = 0
        self.order = list(range(self.num_seqs))
        if self.shuffle:
            random.shuffle(self.order)


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
                Yb[i] = np.nan_to_num(self.Yf[si])

        elif self.start < self.num_nts:
            # initialize full batch of zeros
            Yb = np.zeros((self.batch_size, self.num_targets), dtype='float32')

            # specify partial batch
            Nb = self.num_nts - self.start

            # copy data
            for i in range(Nb):
                si = self.order[self.start+i]
                Yb[i] = np.nan_to_num(self.Yf[si])

        # update start
        self.start = stop

        return Yb, Nb


    def reset(self):
        self.start = 0
        self.order = list(range(self.num_nts))
        if self.shuffle:
            random.shuffle(self.order)
