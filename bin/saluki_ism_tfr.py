#!/usr/bin/env python
# Copyright 2017 Calico LLC
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
import gc
import json
import os
import pdb

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf

from basenji import dataset
from basenji import dna_io
from basenji import rnann
from basenji import stream

if tf.__version__[0] == '1':
  tf.compat.v1.enable_eager_execution()

"""
saluki_ism_tfr.py

Compute in silico mutagenesis of sequences in tfrecords.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <model_file> <data_dir>'
  parser = OptionParser(usage)
  parser.add_option('-c', dest='coding_stop',
    default=False, action='store_true',
    help='Zero out coding track for stop codon mutations [Default: %default]')
  parser.add_option('-l', dest='mut_len',
      default=None, type='int',
      help='Length of 3\' sequence to mutate [Default: %default]')
  parser.add_option('-o', dest='out_dir',
      default='ism',
      help='Output directory for ISM [Default: %default]')
  # parser.add_option('--shifts', dest='shifts',
  #     default='0',
  #     help='Ensemble prediction shifts [Default: %default]')
  # parser.add_option('-t', dest='targets_file',
  #     default=None, type='str',
  #     help='File specifying target indexes and labels in table format')
  parser.add_option('--split', dest='split_label',
      default='test',
      help='Dataset split label for eg TFR pattern [Default: %default]')
  (options, args) = parser.parse_args()

  if len(args) != 3:
    parser.error('Must provide parameters, model, and test data HDF5')
  else:
    params_file = args[0]
    model_file = args[1]
    data_dir = args[2]

  if not os.path.isdir(options.out_dir):
    os.mkdir(options.out_dir)

  # parse shifts to integers
  # options.shifts = [int(shift) for shift in options.shifts.split(',')]

  #######################################################
  # model

  # read targets
  # if options.targets_file is None:
  #   options.targets_file = '%s/targets.txt' % data_dir
  # targets_df = pd.read_csv(options.targets_file, index_col=0, sep='\t')

  # read data stats
  data_stats_file = '%s/statistics.json' % data_dir
  with open(data_stats_file) as data_stats_open:
    data_stats = json.load(data_stats_open)
  num_targets = data_stats['num_targets']

  # read genes
  genes_df = pd.read_csv('%s/genes.tsv' % data_dir, sep='\t', index_col=0)
  if options.split_label == '*':
    gene_ids = genes_df.Gene
  else:
    gene_ids = genes_df[genes_df.Split==options.split_label].Gene
  gene_ids = np.array(gene_ids, dtype='S')

  # read model parameters
  with open(params_file) as params_open:
    params = json.load(params_open)
  params_model = params['model']
  params_train = params['train']
  params_model['seq_length'] = data_stats['length_t']

  if options.mut_len is None:
    options.mut_len = data_stats['length_t']
  elif options.mut_len > data_stats['length_t']:
    parser.error('Specified mutation length %d is greater than sequence length %d' \
      % (options.mut_len, data_stats['length_t']))

  # initialize model
  seqnn_model = rnann.RnaNN(params_model)
  seqnn_model.restore(model_file)

  
  #######################################################
  # ISM sequences

  # construct dataset
  eval_data = dataset.RnaDataset(data_dir,
    split_label=options.split_label,
    batch_size=1,
    mode='eval')
  num_seqs = eval_data.num_seqs

  # make sequence generator
  seqs_gen = satmut_gen(eval_data, options.mut_len, options.coding_stop)

  #################################################################
  # setup output

  scores_h5_file = '%s/scores.h5' % options.out_dir
  if os.path.isfile(scores_h5_file):
    os.remove(scores_h5_file)
  scores_h5 = h5py.File(scores_h5_file, 'w')
  scores_h5.create_dataset('genes', data=gene_ids)
  scores_h5.create_dataset('seqs', dtype='bool',
      shape=(num_seqs, options.mut_len, 4))
  scores_h5.create_dataset('coding', dtype='bool',
      shape=(num_seqs, options.mut_len))
  scores_h5.create_dataset('splice', dtype='bool',
      shape=(num_seqs, options.mut_len))
  scores_h5.create_dataset('ref', dtype='float16',
      shape=(num_seqs, num_targets))
  scores_h5.create_dataset('ism', dtype='float16',
      shape=(num_seqs, options.mut_len, 4, num_targets))

  # store mutagenesis coordinates?


  #################################################################
  # predict scores, write output

  # initialize predictions stream
  batch_size = 2*params_train['batch_size']
  preds_stream = stream.PredStreamGen(seqnn_model, seqs_gen,
    batch_size, stream_seqs=8*batch_size)

  # sequence index
  si = 0

  # predictions index
  pi = 0

  for seq_1hotc, _ in eval_data.dataset:
    # convert to single numpy 1hot
    seq_1hotc = seq_1hotc.numpy().astype('bool')[0]

    # hack compute actual length
    seq_len = 1 + np.max(np.where(seq_1hotc.sum(axis=-1))[0])

    print('Predicting %d, %d nt' % (si,seq_len), flush=True)

    # write reference sequence  
    seq_mut_len = min(seq_len, options.mut_len)
    seq_1hotc_mut = seq_1hotc[seq_len-seq_mut_len:seq_len]
    seq_1hot_mut = seq_1hotc_mut[:,:4]
    scores_h5['seqs'][si,-seq_mut_len:,:] = seq_1hot_mut
    scores_h5['coding'][si,-seq_mut_len:] = seq_1hotc_mut[:,4]
    scores_h5['splice'][si,-seq_mut_len:] = seq_1hotc_mut[:,5]

    # initialize scores
    seq_scores = np.zeros((seq_mut_len, 4, num_targets), dtype='float32')

    # collect reference prediction
    preds_mut0 = preds_stream[pi]
    pi += 1

    # for each mutated position
    for mi in range(seq_mut_len):
      # if position as nucleotide
      if seq_1hot_mut[mi].max() < 1:
        # reference score
        seq_scores[mi,:,:] = preds_mut0
      else:
        # for each nucleotide
        for ni in range(4):
          if seq_1hot_mut[mi,ni]:
            # reference score
            seq_scores[mi,ni,:] = preds_mut0
          else:
            # collect and set mutation score
            seq_scores[mi,ni,:] = preds_stream[pi]
            pi += 1

    # normalize
    seq_scores -= seq_scores.mean(axis=1, keepdims=True)

    # write to HDF5
    scores_h5['ref'][si] = preds_mut0.astype('float16')
    scores_h5['ism'][si,-seq_mut_len:,:,:] = seq_scores.astype('float16')

    # increment sequence
    si += 1

    # clean memory
    gc.collect()
    
  # close output HDF5
  scores_h5.close()


def find_codon_index(seq_1hotc, ii):
  if seq_1hotc[ii,4] == 1:
    ci = ii
  elif ii-1 >= 0 and seq_1hotc[ii-1,4] == 1:
    ci = ii - 1
  elif ii-2 >= 0 and seq_1hotc[ii-2,4] == 1:
    ci = ii - 2
  else:
    ci = None
  return ci


def satmut_gen(eval_data, mut_len, coding_stop=False):
  """Construct generator for 1 hot encoded saturation
     mutagenesis DNA sequences."""

  # taa1 = dna_io.dna_1hot('TAA')
  # tag1 = dna_io.dna_1hot('TAG')
  # tga1 = dna_io.dna_1hot('TGA')

  for seq_1hotc, _ in eval_data.dataset:
    seq_1hotc = seq_1hotc.numpy()[0]
    yield seq_1hotc

    # hack compute actual length
    seq_len = 1 + np.max(np.where(seq_1hotc.sum(axis=-1))[0])

    # set mutation boundaries
    mut_end = seq_len
    mut_start = max(0, mut_end - mut_len)

    # for mutation positions
    for mi in range(mut_start, mut_end):
      # if position as nucleotide
      if seq_1hotc[mi].max() == 1:
        # for each nucleotide
        for ni in range(4):
          # if non-reference
          if seq_1hotc[mi,ni] == 0:
            # copy and modify
            seq_mut_1hotc = np.copy(seq_1hotc)
            seq_mut_1hotc[mi,:4] = 0
            seq_mut_1hotc[mi,ni] = 1

            if coding_stop:
              ci = find_codon_index(seq_mut_1hotc, mi)
              if ci is not None:
                mut_codon_1hot = seq_mut_1hotc[ci:ci+3,:4]
                mut_codon = dna_io.hot1_dna(mut_codon_1hot)
                if mut_codon in ['TAA','TAG','TGA']:
                  seq_mut_1hotc[ci:,4] = 0

            yield seq_mut_1hotc


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
