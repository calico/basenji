#!/usr/bin/env python
# Copyright 2022 Calico LLC

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

from optparse import OptionParser, OptionGroup
import glob
import h5py
import json
import os
import pdb
import subprocess
import sys

import numpy as np
import pandas as pd

from basenji import rnann

"""
saluki_predict_genbank.py

Make predictions for RNAs described in a GenBank file using an ensemble of models.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <models_dir> <fasta>'
  parser = OptionParser(usage)
  parser.add_option('-d', dest='data_head',
      default=None, type='int',
      help='Index for dataset/head [Default: %default]')  
  parser.add_option('-o', dest='out_dir',
      default='grads_out',
      help='Output directory for ISM [Default: %default]')
  parser.add_option('--shifts', dest='shifts',
      default='0',
      help='Ensemble prediction shifts [Default: %default]')
  parser.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')
  (options, args) = parser.parse_args()

  if len(args) != 2:
    parser.error('Must provide model directory and FASTA.')
  else:
    models_dir = args[0]
    fasta_file = args[1]

  os.makedirs(options.out_dir, exist_ok=True)

  #######################################################
  # prep work

  model_str = 'model_best.h5'
  if options.data_head is not None:
    model_str = 'model%d_best.h5' % options.data_head

  num_folds = len(glob.glob('%s/f*_c0/train/%s' % (models_dir,model_str)))
  num_crosses = len(glob.glob('%s/f0_c*/train/%s' % (models_dir,model_str)))
  print('Folds %d, Crosses %d' % (num_folds, num_crosses))

  # read targets
  # if options.targets_file is None:
  #   options.targets_file = '%s/f0c0/data0/targets.txt' % models_dir
  # targets_df = pd.read_csv(options.targets_file, index_col=0, sep='\t')

  # read model parameters
  params_file = '%s/params.json' % models_dir
  with open(params_file) as params_open:
    params = json.load(params_open)
  params_model = params['model']
  params_train = params['train']

  # initialize model
  seqnn_model = rnann.RnaNN(params_model)

  # read sequences from genbank
  seqs_1hot = parse_fasta(fasta_file, params_model['seq_length'])
  num_seqs = seqs_1hot.shape[0]

  #######################################################
  # predict

  scores = []

  for fi in range(num_folds):
    for ci in range(num_crosses):
      print('fold %d, cross %d' % (fi,ci))
      model_file = '%s/f%d_c%d/train/%s' % (models_dir, fi, ci, model_str)

      # prepare model
      seqnn_model.restore(model_file, options.data_head)
      seqnn_model.build_ensemble(options.shifts)

      # predict
      scores_fc = []
      for si in range(num_seqs):
        scores_fc.append(seqnn_model.gradients(seqs_1hot[si]))
      scores.append(scores_fc)

  scores = np.array(scores)
  print(scores.shape)
  scores = np.transpose(scores, [1,2,3,0])

  with h5py.File('%s/scores.h5' % options.out_dir, 'w') as scores_h5:
    scores_h5.create_dataset('seqs', data=seqs_1hot, compression='gzip')
    scores_h5.create_dataset('grads', data=scores, compression='gzip')


def parse_fasta(fasta_file, seq_len):
  seqs_1hot = []

  seq_dna = ''
  for line in open(fasta_file):
    if line[0] == '>':
      if len(seq_dna) > 0:
        # save full sequence
        seq_1hot = dna_1hot(seq_dna, seq_len)
        seqs_1hot.append(seq_1hot)

      # reset      
      seq_dna = ''

    else:
      # concat dna
      seq_dna += line.strip()

  if len(seq_dna) > 0:
    # save full sequence
    seq_1hot = dna_1hot(seq_dna, seq_len)
    seqs_1hot.append(seq_1hot)

  return np.array(seqs_1hot)


def dna_1hot(seq_dna, seq_len):
  seq_dna_len = len(seq_dna)
  seq_1hot = np.zeros((seq_dna_len, 6), dtype='bool')

  # TEMP
  seq_1hot[94,5] = 1

  cds_start = None
  for i in range(seq_dna_len):
    nt = seq_dna[i]

    if nt in 'ACGT':
      if cds_start is None:
        cds_start = i

      if (i - cds_start) % 3 == 0:
        seq_1hot[i, 4] = 1

    nt = nt.upper()

    if nt == 'A':
      seq_1hot[i, 0] = 1
    elif nt == 'C':
      seq_1hot[i, 1] = 1
    elif nt == 'G':
      seq_1hot[i, 2] = 1
    elif nt == 'T':
      seq_1hot[i, 3] = 1

  if seq_dna_len < seq_len:
    ext_len = seq_len - seq_dna_len
    ext_1hot = np.zeros((ext_len, 6), dtype='bool')
    seq_1hot = np.concatenate([seq_1hot,ext_1hot], axis=0)
  elif seq_dna_len > seq_len:
    # slice downstream region?
    seq_1hot = seq_1hot[-seq_len:]

  return seq_1hot

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
