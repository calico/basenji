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
import json
import os
import pdb

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf

from basenji import dataset
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
  parser.add_option('-l', dest='mut_len',
      default=100, type='int',
      help='Length of 3\' sequence to mutate [Default: %default]')
  parser.add_option('-o', dest='out_dir',
      default='ism_out',
      help='Output directory for ISM [Default: %default]')
  # parser.add_option('--shifts', dest='shifts',
  #     default='0',
  #     help='Ensemble prediction shifts [Default: %default]')
  # parser.add_option('-t', dest='targets_file',
  #     default=None, type='str',
  #     help='File specifying target indexes and labels in table format')
  parser.add_option('--split', dest='split_label',
      default='fold0',
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
  num_seqs = data_stats['%s_seqs' % options.split_label]
  num_targets = data_stats['num_targets']

  # read model parameters
  with open(params_file) as params_open:
    params = json.load(params_open)
  params_model = params['model']
  params_train = params['train']
  params_model['seq_length'] = data_stats['length_full']

  # initialize model
  seqnn_model = rnann.RnaNN(params_model)
  seqnn_model.restore(model_file)

  
  #######################################################
  # ISM sequences

  # construct dataset
  eval_data = dataset.RnaDataset(data_dir,
    split_labels=[options.split_label],
    batch_size=1,
    mode='eval')

  # make sequence generator
  seqs_gen = satmut_gen(eval_data, options.mut_len)


  #################################################################
  # setup output

  scores_h5_file = '%s/scores.h5' % options.out_dir
  if os.path.isfile(scores_h5_file):
    os.remove(scores_h5_file)
  scores_h5 = h5py.File(scores_h5_file, 'w')
  scores_h5.create_dataset('seqs', dtype='bool',
      shape=(num_seqs, options.mut_len, 4))
  scores_h5.create_dataset('ism', dtype='float16',
      shape=(num_seqs, options.mut_len, 4, num_targets))

  # store mutagenesis coordinates?

  #################################################################
  # predict scores, write output

  # initialize predictions stream
  preds_stream = stream.PredStreamGen(seqnn_model, seqs_gen, params_train['batch_size'])

  # sequence index
  si = 0

  # predictions index
  pi = 0

  for seq_1hot, _ in eval_data.dataset:
    print('Predicting %d' % si, flush=True)

    # convert to single numpy 1hot
    seq_1hot = seq_1hot[0,:,:4].numpy().astype('bool')

    # hack compute actual length
    seq_len = np.max(np.where(seq_1hot.sum(axis=-1))[0])

    # write reference sequence  
    seq_mut_len = min(seq_len, options.mut_len)
    seq_1hot_mut = seq_1hot[seq_len-seq_mut_len:seq_len]
    scores_h5['seqs'][si,-seq_mut_len:,:] = seq_1hot_mut

    # initialize scores
    seq_scores = np.zeros((seq_mut_len, 4, num_targets), dtype='float32')

    # collect reference prediction
    preds_mut0 = preds_stream[pi]
    pi += 1

    # for each mutated position
    for mi in range(seq_mut_len):
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
    scores_h5['ism'][si,-seq_mut_len:,:,:] = seq_scores.astype('float16')

    # increment sequence
    si += 1
    
  # close output HDF5
  scores_h5.close()


def satmut_gen(eval_data, mut_len):
  """Construct generator for 1 hot encoded saturation
     mutagenesis DNA sequences."""

  for seq_1hot, _ in eval_data.dataset:
    seq_1hot = seq_1hot.numpy()[0]
    yield seq_1hot

    # hack compute actual length
    seq_len = np.max(np.where(seq_1hot.sum(axis=-1))[0])

    # set mutation boundaries
    mut_end = seq_len
    mut_start = max(0, mut_end - mut_len)

    # for mutation positions
    for mi in range(mut_start, mut_end):
      # for each nucleotide
      for ni in range(4):
        # if non-reference
        if seq_1hot[mi,ni] == 0:
          # copy and modify
          seq_mut_1hot = np.copy(seq_1hot)
          seq_mut_1hot[mi,:] = 0
          seq_mut_1hot[mi,ni] = 1
          yield seq_mut_1hot


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
