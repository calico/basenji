#!/usr/bin/env python
import sys

import numpy as np

################################################################################
# io.py
#
# Methods to load the training data.
################################################################################

def dna_1hot(seq, seq_len=None):
    ''' dna_1hot

    Args:
      seq: nucleotide sequence.
      seq_len: length to extend sequences to.

    Returns:
      seq_code: length by nucleotides array representation.
    '''
    if seq_len is None:
        seq_len = len(seq)
        seq_start = 0
    else:
        if seq_len <= len(seq):
            # trim the sequence
            seq_trim = (len(seq)-seq_len)//2
            seq = seq[seq_trim:seq_trim+seq_len]
            seq_start = 0
        else:
            seq_start = (seq_len-len(seq))//2

    seq = seq.upper()

    # map nt's to a matrix len(seq)x4 of 0's and 1's.
    seq_code = np.zeros((seq_len,4), dtype='bool')
    for i in range(seq_len):
        if i >= seq_start and i-seq_start < len(seq):
            nt = seq[i-seq_start]
            if nt == 'A':
                seq_code[i,0] = 1
            elif nt == 'C':
                seq_code[i,1] = 1
            elif nt == 'G':
                seq_code[i,2] = 1
            elif nt == 'T':
                seq_code[i,3] = 1

    return seq_code


def dna_1hot_float(seq, seq_len=None):
    ''' dna_1hot

    Args:
      seq: nucleotide sequence.
      seq_len: length to extend sequences to.

    Returns:
      seq_code: length by nucleotides array representation.
    '''
    if seq_len is None:
        seq_len = len(seq)
        seq_start = 0
    else:
        if seq_len <= len(seq):
            # trim the sequence
            seq_trim = (len(seq)-seq_len)//2
            seq = seq[seq_trim:seq_trim+seq_len]
            seq_start = 0
        else:
            seq_start = (seq_len-len(seq))//2

    seq = seq.upper()
    seq = seq.replace('A','0')
    seq = seq.replace('C','1')
    seq = seq.replace('G','2')
    seq = seq.replace('T','3')

    # map nt's to a matrix len(seq)x4 of 0's and 1's.
    #  dtype='int8' fails for N's
    seq_code = np.zeros((seq_len,4), dtype='float16')
    for i in range(seq_len):
        if i < seq_start:
            seq_code[i,:] = 0.25
        else:
            try:
                seq_code[i,int(seq[i-seq_start])] = 1
            except:
                seq_code[i,:] = 0.25

    return seq_code


def hot1_dna(seqs_1hot):
    ''' Convert 1-hot coded sequences to ACGTN. '''

    singleton = False
    if seqs_1hot.ndim == 2:
        singleton = True
        seqs_1hot = np.expand_dims(seqs_1hot, 0)

    seqs = []
    for si in range(seqs_1hot.shape[0]):
        seq_list = ['A']*seqs_1hot.shape[1]
        for li in range(seqs_1hot.shape[1]):
            if seqs_1hot[si,li,0] == 1:
                seq_list[li] = 'A'
            elif seqs_1hot[si,li,1] == 1:
                seq_list[li] = 'C'
            elif seqs_1hot[si,li,2] == 1:
                seq_list[li] = 'G'
            elif seqs_1hot[si,li,3] == 1:
                seq_list[li] = 'T'
            else:
                seq_list[li] = 'N'

        seqs.append(''.join(seq_list))

    if singleton:
        seqs = seqs[0]

    return seqs


def hot1_rc(seqs_1hot):
    ''' Reverse complement a batch of one hot coded sequences '''

    # reverse
    seqs_1hot = seqs_1hot[:,::-1,:]

    # swap A and T
    seqs_1hot[:,:,[0,3]] = seqs_1hot[:,:,[3,0]]

    # swap C and G
    seqs_1hot[:,:,[1,2]] = seqs_1hot[:,:,[2,1]]

    return seqs_1hot


def hot1_set(seqs_1hot, pos, nt):
    # reset
    seqs_1hot[pos,:] = 0

    # set
    if nt == 'A':
        seqs_1hot[pos,0] = 1
    elif nt == 'C':
        seqs_1hot[pos,1] = 1
    elif nt == 'G':
        seqs_1hot[pos,2] = 1
    elif nt == 'T':
        seqs_1hot[pos,3] = 1
    else:
        print('Invalid nucleotide set %s' % nt, file=sys.stderr)


def read_job_params(job_file):
    ''' Read job parameters from table. '''

    job = {}

    if job_file is not None:
        for line in open(job_file):
            param, val = line.split()

            # require a decimal for floats
            try:
                if val.find('e') != -1:
                    val = float(val)
                elif val.find('.') == -1:
                    val = int(val)
                else:
                    val = float(val)
            except ValueError:
                pass

            if param in job:
                # change to a list
                if type(job[param]) != list:
                    job[param] = [job[param]]

                # append new value
                job[param].append(val)
            else:
                job[param] = val

        print(job)

    return job
