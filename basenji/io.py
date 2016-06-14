#!/usr/bin/env python
import sys
from collections import OrderedDict

import numpy as np
import numpy.random as npr
from sklearn import preprocessing

################################################################################
# dna_io.py
#
# Methods to load the training data.
################################################################################


def align_seqs_scores_1hot(seq_vecs, seq_scores, sort=False):
    ''' align_seqs_scores

    Align entries from input dicts into numpy matrices ready for analysis.

    Args
      seq_vecs:      Dict mapping headers to sequence vectors.
      seq_scores:    Dict mapping headers to score vectors.
      sort:          Sort by headers.

    Returns
      train_seqs:    Matrix with sequence vector rows.
      train_scores:  Matrix with score vector rows.
    '''
    if sort:
        seq_headers = sorted(seq_vecs.keys())
    else:
        seq_headers = seq_vecs.keys()

    # construct lists of vectors
    train_scores = []
    train_seqs = []
    for header in seq_headers:
        train_seqs.append(seq_vecs[header])
        train_scores.append(seq_scores[header])

    # stack into matrices
    train_seqs = np.array(train_seqs)
    train_scores = np.array(train_scores)

    return train_seqs, train_scores


################################################################################
# check_order
#
# Check that the order of sequences in a matrix of vectors matches the order
# in the given fasta file
################################################################################
def check_order(seq_vecs, fasta_file):
    # reshape into seq x 4 x len
    seq_mats = np.reshape(seq_vecs, (seq_vecs.shape[0], 4, seq_vecs.shape[1]/4))

    # generate sequences
    real_seqs = []
    for i in range(seq_mats.shape[0]):
        seq_list = ['']*seq_mats.shape[2]
        for j in range(seq_mats.shape[2]):
            if seq_mats[i,0,j] == 1:
                seq_list[j] = 'A'
            elif seq_mats[i,1,j] == 1:
                seq_list[j] = 'C'
            elif seq_mats[i,2,j] == 1:
                seq_list[j] = 'G'
            elif seq_mats[i,3,j] == 1:
                seq_list[j] = 'T'
            else:
                seq_list[j] = 'N'
        real_seqs.append(''.join(seq_list))

    # load FASTA sequences
    fasta_seqs = []
    for line in open(fasta_file):
        if line[0] == '>':
            fasta_seqs.append('')
        else:
            fasta_seqs[-1] += line.rstrip()

    # check
    assert(len(real_seqs) == len(fasta_seqs))

    for i in range(len(fasta_seqs)):
        try:
            assert(fasta_seqs[i] == real_seqs[i])
        except:
            print(fasta_seqs[i])
            print(real_seqs[i])
            exit()


def dna_one_hot(seq, seq_len=None):
    ''' dna_one_hot

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


################################################################################
# fasta2dict
#
# Read a multifasta file into a dict.  Taking the whole line as the key.
#
# I've found this can be quite slow for some reason, even for a single fasta
# entry.
################################################################################
def fasta2dict(fasta_file):
    fasta_dict = OrderedDict()
    header = ''

    for line in open(fasta_file):
        if line[0] == '>':
            #header = line.split()[0][1:]
            header = line[1:].rstrip()
            fasta_dict[header] = ''
        else:
            fasta_dict[header] += line.rstrip()

    return fasta_dict


def hash_scores(scores_file):
    ''' hash_scores

    Args
      scores_file:

    Returns
      seq_scores:  Dict mapping FASTA headers to score vectors.
    '''
    seq_scores = {}

    for line in open(scores_file):
        a = line.split()

        try:
            seq_scores[a[0]] = np.array([float(a[i]) for i in range(1,len(a))])
        except:
            print('Ignoring header line', file=sys.stderr)

    # consider converting the scores to integers
    int_scores = True
    for header in seq_scores:
        if not np.equal(np.mod(seq_scores[header], 1), 0).all():
            print(seq_scores[header])
            int_scores = False
            break

    if int_scores:
        for header in seq_scores:
            seq_scores[header] = seq_scores[header].astype('int8')

        '''
        for header in seq_scores:
            if seq_scores[header] > 0:
                seq_scores[header] = np.array([0, 1], dtype=np.min_scalar_type(1))
            else:
                seq_scores[header] = np.array([1, 0], dtype=np.min_scalar_type(1))
        '''

    return seq_scores


################################################################################
# hash_sequences_1hot
#
# Input
#  fasta_file:  Input FASTA file.
#  extend_len:  Extend the sequences to this length.
#
# Output
#  seq_vecs:    Dict mapping FASTA headers to sequence representation vectors.
################################################################################
def hash_sequences_1hot(fasta_file, extend_len=None):
    # determine longest sequence
    if extend_len is not None:
        seq_len = extend_len
    else:
        seq_len = 0
        seq = ''
        for line in open(fasta_file):
            if line[0] == '>':
                if seq:
                    seq_len = max(seq_len, len(seq))

                header = line[1:].rstrip()
                seq = ''
            else:
                seq += line.rstrip()

        if seq:
            seq_len = max(seq_len, len(seq))

    # load and code sequences
    seq_vecs = OrderedDict()
    seq = ''
    for line in open(fasta_file):
        if line[0] == '>':
            if seq:
                seq_vecs[header] = dna_one_hot(seq, seq_len)

            header = line[1:].rstrip()
            seq = ''
        else:
            seq += line.rstrip()

    if seq:
        seq_vecs[header] = dna_one_hot(seq, seq_len)

    return seq_vecs


def load_data_1hot(fasta_file, scores_file, extend_len=None, mean_norm=False, whiten=False, permute=True, sort=False):
    ''' load_data_1hot

    Args
      fasta_file:  Input FASTA file.
      scores_file: Input scores file.

    Returns
      train_seqs:    Matrix with sequence vector rows.
      train_scores:  Matrix with score vector rows.
    '''
    # load sequences
    seq_vecs = hash_sequences_1hot(fasta_file, extend_len)

    # load scores
    seq_scores = hash_scores(scores_file)

    # align and construct input matrix
    train_seqs, train_scores = align_seqs_scores_1hot(seq_vecs, seq_scores, sort)

    # whiten scores
    if whiten:
        train_scores = preprocessing.scale(train_scores)
    elif mean_norm:
        train_scores -= np.mean(train_scores, axis=0)

    # randomly permute
    if permute:
        order = npr.permutation(train_seqs.shape[0])
        train_seqs = train_seqs[order]
        train_scores = train_scores[order]

    return train_seqs, train_scores


################################################################################
# load_sequences
#
# Input
#  fasta_file:  Input FASTA file.
#
# Output
#  train_seqs:    Matrix with sequence vector rows.
#  train_scores:  Matrix with score vector rows.
################################################################################
def load_sequences(fasta_file, permute=False):
    # load sequences
    seq_vecs = hash_sequences_1hot(fasta_file)

    # stacks
    train_seqs = np.array(seq_vecs.values())

    # randomly permute the data
    if permute:
        order = npr.permutation(train_seqs.shape[0])
        train_seqs = train_seqs[order]

    return train_seqs


################################################################################
# one_hot_get
#
# Input
#  seq_vec:
#  pos:
#
# Output
#  nt
################################################################################
def one_hot_get(seq_vec, pos):
    seq_len = len(seq_vec)/4

    a0 = 0
    c0 = seq_len
    g0 = 2*seq_len
    t0 = 3*seq_len

    if seq_vec[a0+pos] == 1:
        nt = 'A'
    elif seq_vec[c0+pos] == 1:
        nt = 'C'
    elif seq_vec[g0+pos] == 1:
        nt = 'G'
    elif seq_vec[t0+pos] == 1:
        nt = 'T'
    else:
        nt = 'N'

    return nt


################################################################################
# one_hot_set
#
# Assuming the sequence is given as 4x1xLENGTH
# Input
#  seq_vec:
#  pos:
#  nt
#
# Output
################################################################################
def one_hot_set(seq_vec, pos, nt):
    # zero all
    for ni in range(4):
        seq_vec[ni,0,pos] = 0

    # set the nt
    if nt == 'A':
        seq_vec[0,0,pos] = 1
    elif nt == 'C':
        seq_vec[1,0,pos] = 1
    elif nt == 'G':
        seq_vec[2,0,pos] = 1
    elif nt == 'T':
        seq_vec[3,0,pos] = 1
    else:
        for ni in range(4):
            seq_vec[ni,0,pos] = 0.25


################################################################################
# one_hot_set_1d
#
# Input
#  seq_vec:
#  pos:
#  nt
#
# Output
################################################################################
def one_hot_set_1d(seq_vec, pos, nt):
    seq_len = len(seq_vec)/4

    a0 = 0
    c0 = seq_len
    g0 = 2*seq_len
    t0 = 3*seq_len

    # zero all
    seq_vec[a0+pos] = 0
    seq_vec[c0+pos] = 0
    seq_vec[g0+pos] = 0
    seq_vec[t0+pos] = 0

    # set the nt
    if nt == 'A':
        seq_vec[a0+pos] = 1
    elif nt == 'C':
        seq_vec[c0+pos] = 1
    elif nt == 'G':
        seq_vec[g0+pos] = 1
    elif nt == 'T':
        seq_vec[t0+pos] = 1
    else:
        seq_vec[a0+pos] = 0.25
        seq_vec[c0+pos] = 0.25
        seq_vec[g0+pos] = 0.25
        seq_vec[t0+pos] = 0.25


def vecs2dna(seq_vecs):
    ''' vecs2dna

    Input:
        seq_vecs:
    Output:
        seqs
    '''

    seqs = []
    for i in range(seq_vecs.shape[0]):
        seq_list = ['']*seq_vecs.shape[1]
        for j in range(seq_vecs.shape[1]):
            if seq_vecs[i,j,0] == 1:
                seq_list[j] = 'A'
            elif seq_vecs[i,j,1] == 1:
                seq_list[j] = 'C'
            elif seq_vecs[i,j,2] == 1:
                seq_list[j] = 'G'
            elif seq_vecs[i,j,3] == 1:
                seq_list[j] = 'T'
            elif seq_vecs[i,j,:].sum() == 1:
                seq_list[j] = 'N'
            else:
                print('Malformed position vector: ', seq_vecs[i,:,j], 'for sequence %d position %d' % (i,j), file=sys.stderr)
        seqs.append(''.join(seq_list))
    return seqs
