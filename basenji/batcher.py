# Copyright 2017 Calico LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from __future__ import print_function

import h5py
import numpy as np
import random
import sys

import basenji

class Batcher:
    ''' Batcher

    Class to manage batches.
    '''
    def __init__(self, Xf, Yf=None, NAf=None, batch_size=64, pool_width=1, shuffle=False):
        self.Xf = Xf
        self.num_seqs = self.Xf.shape[0]
        self.seq_len = self.Xf.shape[1]
        self.seq_depth = self.Xf.shape[2]

        self.Yf = Yf
        if self.Yf is not None:
            self.num_targets = self.Yf.shape[2]

        self.NAf = NAf

        self.batch_size = batch_size
        self.pool_width = pool_width
        if self.seq_len % self.pool_width != 0:
            print('Pool width %d does not evenly divide the sequence length %d' % (self.pool_width,self.seq_len), file=sys.stderr)
            exit(1)

        self.shuffle = shuffle

        self.reset()


    def next(self, fwdrc=True, shift=0):
        ''' Load the next batch from the HDF5. '''
        Xb = None
        Yb = None
        NAb = None
        Nb = 0

        stop = self.start + self.batch_size
        if self.start < self.num_seqs:
            # full or partial batch
            if stop <= self.num_seqs:
                Nb = self.batch_size
            else:
                Nb = self.num_seqs - self.start

            # initialize
            Xb = np.zeros((self.batch_size, self.seq_len, self.seq_depth), dtype='float32')
            if self.Yf is not None:
                if self.Yf.dtype == np.uint8:
                    ytype = 'int32'
                else:
                    ytype = 'float32'

                Yb = np.zeros((self.batch_size, self.seq_len//self.pool_width, self.num_targets), dtype=ytype)
                NAb = np.zeros((self.batch_size, self.seq_len//self.pool_width), dtype='bool')

            # copy data
            for i in range(Nb):
                si = self.order[self.start+i]
                Xb[i] = self.Xf[si]

                # fix N positions
                Xbi_n = (Xb[i].sum(axis=1) == 0)
                Xb[i] = Xb[i] + (1/self.seq_depth)*Xbi_n.repeat(self.seq_depth).reshape(self.seq_len,self.seq_depth)

                if self.Yf is not None:
                    Yb[i] = np.nan_to_num(self.Yf[si])

                    if self.NAf is not None:
                        NAb[i] = self.NAf[si]

        # reverse complement and shift
        if Xb is not None:
            Xb = basenji.dna_io.hot1_augment(Xb, fwdrc, shift)
        if not fwdrc:
            if Yb is not None:
                Yb = Yb[:,::-1,:]
            if NAb is not None:
                NAb = NAb[:,::-1]

        # update start
        self.start = stop

        return Xb, Yb, NAb, Nb


    def reset(self):
        self.start = 0
        self.order = list(range(self.num_seqs))
        if self.shuffle:
            random.shuffle(self.order)


class BatcherF:
    ''' BatcherF

    Class to manage batches of data in fourier space.
    '''
    def __init__(self, Xf, Yf_real, Yf_imag, NAf=None, batch_size=64, pool_width=1, shuffle=False):
        self.Xf = Xf
        self.num_seqs = self.Xf.shape[0]
        self.seq_len = self.Xf.shape[1]
        self.seq_depth = self.Xf.shape[2]

        self.Yf_real = Yf_real
        self.Yf_imag = Yf_imag
        self.num_targets = self.Yf_real.shape[2]

        self.NAf = NAf

        self.batch_size = batch_size
        self.pool_width = pool_width
        if self.seq_len % self.pool_width != 0:
            print('Pool width %d does not evenly divide the sequence length %d' % (self.pool_width,self.seq_len), file=sys.stderr)
            exit(1)

        self.shuffle = shuffle

        self.reset()


    def next(self):
        ''' Load the next batch from the HDF5. '''
        Xb = None
        Yb = None
        NAb = None
        Nb = 0

        stop = self.start + self.batch_size
        if self.start < self.num_seqs:
            # full or partial batch
            if stop <= self.num_seqs:
                Nb = self.batch_size
            else:
                Nb = self.num_seqs - self.start

            # initialize
            Xb = np.zeros((self.batch_size, self.seq_len, self.seq_depth), dtype='float32')
            Yb = np.zeros((self.batch_size, self.seq_len//self.pool_width, self.num_targets), dtype='float32')
            NAb = np.zeros((self.batch_size, self.seq_len//self.pool_width), dtype='bool')

            # copy data
            for i in range(Nb):
                si = self.order[self.start+i]
                Xb[i] = self.Xf[si]

                # fix N positions
                Xbi_n = (Xb[i].sum(axis=1) == 0)
                Xb[i] = Xb[i] + (1/self.seq_depth)*Xbi_n.repeat(self.seq_depth).reshape(self.seq_len,self.seq_depth)

                # inverse fourier transform
                ybi_fourier = self.Yf_real[si] + self.Yf_imag[si]*1j
                for ti in range(self.num_targets):
                    Yb[i,:,ti] = np.fft.irfft(ybi_fourier[:,ti], self.seq_len//self.pool_width)

                if self.NAf is not None:
                    NAb[i] = self.NAf[si]


        # update start
        self.start = stop

        return Xb, Yb, NAb, Nb


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
