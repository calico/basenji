#!/usr/bin/env python
# Copyright 2022 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
from __future__ import print_function
from optparse import OptionParser
import os
import random
import gc

import h5py
import json
import numpy as np
import pandas as pd
import pysam
import tensorflow as tf

from basenji import dna_io
from basenji import seqnn
from basenji import stream

'''
borzoi_trip.py

Predict insertions from TRIP assay.
'''

def eprint(*args, **kwargs):
  print(*args, file=sys.stderr, **kwargs)

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <model_file> <promoter_file> <insertions_file>'
  parser = OptionParser(usage)
  parser.add_option('-f', dest='fasta',
      default='%s/data/hg38.fa' % os.environ['BASENJIDIR'],
      help='Genome FASTA for sequences [Default: %default]')
  parser.add_option('--site', dest='site',
      default=False, action='store_true',
      help='Return the insertion site without the promoter [Default: %default]')
  parser.add_option('--reporter', dest='reporter',
      default=False, action='store_true',
      help='Insert the flanking piggyback reporter with the promoter [Default: %default]')
  parser.add_option('--reporter_bare', dest='reporter_bare',
      default=False, action='store_true',
      help='Insert the flanking piggyback reporter with the promoter (no terminal repeats) [Default: %default]')
  parser.add_option('-o', dest='out_dir',
      default='trip',
      help='Output directory [Default: %default]')
  parser.add_option('--rc', dest='rc',
      default=True, action='store_true',
      help='Average forward and reverse complement predictions [Default: %default]')
  parser.add_option('--shifts', dest='shifts',
      default='0', type='str',
      help='Ensemble prediction shifts [Default: %default]')
  parser.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')
  (options, args) = parser.parse_args()

  if len(args) != 4:
    parser.error('Must provide parameters, model, TRIP promoter sequences, and TRIP insertion sites')
  else:
    params_file = args[0]
    model_file = args[1]
    promoters_file = args[2]
    insertions_file = args[3]

  if not os.path.isdir(options.out_dir):
    os.mkdir(options.out_dir)

  options.shifts = [int(shift) for shift in options.shifts.split(',')]

  #################################################################
  # read parameters and targets

  # read model parameters
  with open(params_file) as params_open:
    params = json.load(params_open)
  params_model = params['model']
  params_train = params['train']

  if options.targets_file is None:
    parser.error('Must provide targets table to properly handle strands.')
  else:
    targets_df = pd.read_csv(options.targets_file, sep='\t', index_col=0)

  #################################################################
  # setup model
  
  target_slice = np.array(targets_df.index.values, dtype='int32')
  strand_pair = np.array(targets_df.strand_pair.values, dtype='int32')
  
  # create local index of strand_pair (relative to sliced targets)
  target_slice_dict = {ix : i for i, ix in enumerate(target_slice.tolist())}
  slice_pair = np.array([target_slice_dict[ix] for ix in strand_pair.tolist()], dtype='int32')
  
  seqnn_model = seqnn.SeqNN(params_model)
  seqnn_model.restore(model_file)
  seqnn_model.build_slice(targets_df.index)
  seqnn_model.strand_pair.append(slice_pair)
  seqnn_model.build_ensemble(options.rc, options.shifts)
  
  ################################################################
  # promoters

  # read promoter info
  promoters_df = pd.read_excel(promoters_file)

  # genome fasta
  fasta_open = pysam.Fastafile(options.fasta)
  
  # define piggyback reporter sequence
  reporter_left = 'TTAACCCTAGAAAGATAGTCTGCGTAAAATTGACGCATGCATTCTTGAAATATTGCTCTCTCTTTCTAAATAGCGCGAATCCGTCGCTGTGCATTTAGGACATCTCAGTCGCCGCTTGGAGCTCCCGTGAGGCGTGCTTGTCAATGCGGTAAGTGTCACTGATTTTGAACTATAACGACCGCGTGAGTCAAAATGACGCATGATTATCTTTTACGTGACTTTTAAGATTTAACTCATACGATAATTATATTGTTATTTCATGTTCTACTTACGTGATAACTTATTATATATATATTTTCTTGTTATAGATATCAACTAGAATGCTAGCATGGGCCCATCTCGAGGATCCACCGGTCTAGAAAGCTTAGGCCTCCAAGG'
  
  reporter_right = 'ATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCCTGACCTACGGCGTGCAGTGCTTCAGCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAGGTGAAGTTCGAGGGCGACACCCTGGTGAACCGCATCGAGCTGAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAGGTGAACTTCAAGATCCGCCACAACATCGAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCACCCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAGTAAGAATTCGCGGCCGCATACGATTTAGGTGACACTGCAGATCATATGACAATTGTGGCCGGCCCTTGTGACTGGGAAAACCCTGGCGTAAATAAAATACGAAATGACTAGTTAAAAGTTTTGTTACTTTATAGAAGAAATTTTGAGTTTTTGTTTTTTTTTAATAAATAAATAAACATAAATAAATTGTTTGTTGAATTTATTATTAGTATGTAAGTGTAAATATAATAAAACTTAATATCTATTCAAATTAATAAATAAACCTCGATATACAGACCGATAAAACACATGCGTCAATTTTACGCATGATTATCTTTAACGTACGTCACAATATGATTATCTTTCTAGGGTTAA'
  
  reporter_right_bare = 'ATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCCTGACCTACGGCGTGCAGTGCTTCAGCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAGGTGAAGTTCGAGGGCGACACCCTGGTGAACCGCATCGAGCTGAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAGGTGAACTTCAAGATCCGCCACAACATCGAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCACCCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAGTAAGAATTCGCGGCCGCATACGATTTAGGTGACACTGCAGATCATATGACAATTGTGGCCGGCCCTTGTGACTGGGAAAACCCTGGCGTAAATAAAATACGAAATGACTAGTTTAATGTTTGTTTTCTTATA'

  # read promoter sequence
  promoter_seq1 = {}
  for pi in range(promoters_df.shape[0]):
    promoter_chr, promoter_range = promoters_df.iloc[pi].Region.split(':')
    promoter_start, promoter_end = promoter_range.split('-')
    promoter_start, promoter_end = int(promoter_start), int(promoter_end)

    promoter_dna = fasta_open.fetch(promoter_chr, promoter_start, promoter_end)
    promoter_1hot = dna_io.dna_1hot(promoter_dna)
    if promoters_df.iloc[pi].Strand == '-':
      promoter_1hot = dna_io.hot1_rc(promoter_1hot)
    
    # optionally insert full piggyback reporter
    if options.reporter :
      if pi == 0 :
        print("Using full reporter construct, " + reporter_left[:5] + "...", flush=True)
      
      reporter_left_1hot = dna_io.dna_1hot(reporter_left)
      reporter_right_1hot = dna_io.dna_1hot(reporter_right)
      
      promoter_1hot = np.concatenate([reporter_left_1hot, promoter_1hot, reporter_right_1hot], axis=0)
    elif options.reporter_bare :
      if pi == 0 :
        print("Using bare reporter construct, " + reporter_right_bare[:5] + "...", flush=True)
      
      reporter_right_bare_1hot = dna_io.dna_1hot(reporter_right_bare)
      
      promoter_1hot = np.concatenate([promoter_1hot, reporter_right_bare_1hot], axis=0)

    promoter_seq1[promoters_df.iloc[pi].Gene] = promoter_1hot

  ################################################################
  # insertions

  # read insertion info
  insertions_df = pd.read_csv(insertions_file, sep='\t')

  # construct sequence generator
  def insertion_seqs():
    for ii in range(insertions_df.shape[0]):
      chrm = insertions_df.iloc[ii].seqname
      pos = insertions_df.iloc[ii].position
      strand = insertions_df.iloc[ii].strand

      if options.site:
        flank_len = params_model['seq_length']
        flank_start = pos - flank_len//2
        flank_end = flank_start + flank_len

        # left flank
        if flank_start < 0:
          flank_dna = ''.join(random.choices('ACGT', k=-flank_start))
          flank_start = 0
        else:
          flank_dna = ''

        # fetch DNA
        flank_dna += fasta_open.fetch(chrm, flank_start, flank_end)

        # right flank
        if len(flank_dna) < flank_len:
          over_len = flank_len - len(flank_dna)
          flank_dna += ''.join(random.choices('ACGT', k=over_len))

        # 1 hot 
        insertion_1hot = dna_io.dna_1hot(flank_dna)

      else:
        promoter = insertions_df.iloc[ii].promoter
        promoter_1hot = promoter_seq1[promoter]

        # reverse complement
        if strand == '-':
          promoter_1hot = dna_io.hot1_rc(promoter_1hot)

        # get flanking sequence
        flank_len = params_model['seq_length'] - promoter_1hot.shape[0]
        flank_start = pos - flank_len//2
        flank_end = flank_start + flank_len

        # left flank
        if flank_start < 0:
          flank_dna_left = ''.join(random.choices('ACGT', k=-flank_start))
          flank_start = 0
        else:
          flank_dna_left = ''
        flank_dna_left += fasta_open.fetch(chrm, flank_start, pos)
        flank_1hot_left = dna_io.dna_1hot(flank_dna_left)

        flank_dna_right = fasta_open.fetch(chrm, pos, flank_end)
        if len(flank_dna_right) < flank_end-pos:
          over_len = flank_end - pos - len(flank_dna_right)
          flank_dna_right += ''.join(random.choices('ACGT', k=over_len))
        flank_1hot_right = dna_io.dna_1hot(flank_dna_right)

        # combine insertion sequence
        insertion_1hot = np.concatenate([flank_1hot_left, promoter_1hot, flank_1hot_right], axis=0)
        # insertion_1hot = np.expand_dims(insertion_1hot, axis=0)

      # orient promoters forward
      if strand == '-':
        insertion_1hot = dna_io.hot1_rc(insertion_1hot)

      assert(insertion_1hot.shape[0] == params_model['seq_length'])

      yield insertion_1hot

  # initialize prediction stream
  pred_stream = stream.PredStreamGen(seqnn_model, insertion_seqs(), 1)

  ################################################################
  # predict

  # initialize h5
  preds_h5 = h5py.File('%s/preds.h5' % options.out_dir, 'w')
  preds_h5.create_dataset('preds', dtype='float16', shape=(len(insertions_df), seqnn_model.target_lengths[0], len(targets_df)))
  
  # predictions index
  pi = 0
  
  # collect garbage after some amount of iterations
  collect_every = 256
  
  # predict for all sequences
  for pi in range(insertions_df.shape[0]) :
    
    # get predictions
    preds = pred_stream[pi]
    
    preds_h5['preds'][pi, ...] = preds[:, :].astype('float16')
    
    # collect garbage after a number of predictions
    if pi % collect_every == 0 :
        gc.collect()
  
  # save h5
  preds_h5.close()

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
