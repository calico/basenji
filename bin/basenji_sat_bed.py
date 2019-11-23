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
import pickle
from queue import Queue
import random
import sys
from threading import Thread

import h5py
import numpy as np
import pandas as pd
import pysam
import tensorflow as tf

if tf.__version__[0] == '1':
  tf.compat.v1.enable_eager_execution()

from basenji import bed
from basenji import dna_io
from basenji import seqnn
from basenji import stream

'''
basenji_sat_bed.py

Perform an in silico saturation mutagenesis of sequences in a BED file.
'''

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <model_file> <bed_file>'
  parser = OptionParser(usage)
  parser.add_option('-f', dest='genome_fasta',
      default=None,
      help='Genome FASTA for sequences [Default: %default]')
  parser.add_option('-l', dest='mut_len',
      default=200, type='int',
      help='Length of center sequence to mutate [Default: %default]')
  parser.add_option('-o', dest='out_dir',
      default='sat_mut', help='Output directory [Default: %default]')
  parser.add_option('--plots', dest='plots',
      default=False, action='store_true',
      help='Make heatmap plots [Default: %default]')
  parser.add_option('-p', dest='processes',
      default=None, type='int',
      help='Number of processes, passed by multi script')
  parser.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Ensemble forward and reverse complement predictions [Default: %default]')
  parser.add_option('--shifts', dest='shifts',
      default='0',
      help='Ensemble prediction shifts [Default: %default]')
  parser.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')
  (options, args) = parser.parse_args()

  if len(args) == 3:
    # single worker
    params_file = args[0]
    model_file = args[1]
    bed_file = args[2]

  elif len(args) == 5:
    # multi worker
    options_pkl_file = args[0]
    params_file = args[1]
    model_file = args[2]
    bed_file = args[3]
    worker_index = int(args[4])

    # load options
    options_pkl = open(options_pkl_file, 'rb')
    options = pickle.load(options_pkl)
    options_pkl.close()

    # update output directory
    options.out_dir = '%s/job%d' % (options.out_dir, worker_index)

  else:
    parser.error('Must provide parameter and model files and BED file')

  if not os.path.isdir(options.out_dir):
    os.mkdir(options.out_dir)

  options.shifts = [int(shift) for shift in options.shifts.split(',')]

  #################################################################
  # read parameters and construct model

  # read model parameters
  with open(params_file) as params_open:
    params = json.load(params_open)
  params_model = params['model']
  params_train = params['train']

  # initialize model
  seqnn_model = seqnn.SeqNN(params_model)
  seqnn_model.restore(model_file)
  seqnn_model.build_ensemble(options.rc, options.shifts)

  #################################################################
  # collet target information

  num_targets = seqnn_model.num_targets()

  if options.targets_file is None:
    target_ids = ['t%d' % ti for ti in range(num_targets)]
    target_labels = ['']*len(target_ids)
    target_indexes = np.arange(num_targets)

  else:
    targets_df = pd.read_table(options.targets_file, index_col=0)
    target_ids = targets_df.identifier
    target_labels = targets_df.description
    target_indexes = np.array(targets_df.index)

  num_targets = len(target_ids)

  #################################################################
  # sequence dataset

  # read sequences from BED
  seqs_dna, seqs_coords = bed.make_bed_seqs(
    bed_file, options.genome_fasta, params_model['seq_length'], stranded=True)

  # filter for worker SNPs
  if options.processes is not None:
    worker_bounds = np.linspace(0, len(seqs_dna), options.processes+1, dtype='int')
    seqs_dna = seqs_dna[worker_bounds[worker_index]:worker_bounds[worker_index+1]]
    seqs_coords = seqs_coords[worker_bounds[worker_index]:worker_bounds[worker_index+1]]

  num_seqs = len(seqs_dna)

  # determine mutation region limits
  seq_mid = params_model['seq_length'] // 2
  mut_start = seq_mid - options.mut_len // 2
  mut_end = mut_start + options.mut_len

  # make sequence generator
  seqs_gen = satmut_gen(seqs_dna, mut_start, mut_end)

  #################################################################
  # setup output

  scores_h5_file = '%s/scores.h5' % options.out_dir
  if os.path.isfile(scores_h5_file):
    os.remove(scores_h5_file)
  scores_h5 = h5py.File('%s/scores.h5' % options.out_dir)
  scores_h5.create_dataset('scores', dtype='float16',
      shape=(num_seqs, options.mut_len, 4, num_targets))
  scores_h5.create_dataset('seqs', dtype='bool',
      shape=(num_seqs, options.mut_len, 4))

  # store mutagenesis sequence coordinates
  seqs_chr, seqs_start, _, seqs_strand = zip(*seqs_coords)
  seqs_chr = np.array(seqs_chr, dtype='S')
  seqs_start = np.array(seqs_start) + mut_start
  seqs_end = seqs_start + options.mut_len
  seqs_strand = np.array(seqs_strand, dtype='S')
  scores_h5.create_dataset('chrom', data=seqs_chr)
  scores_h5.create_dataset('start', data=seqs_start)
  scores_h5.create_dataset('end', data=seqs_end)
  scores_h5.create_dataset('strand', data=seqs_strand)

  preds_per_seq = 1 + 3*options.mut_len

  score_threads = []
  score_queue = Queue()
  for i in range(1):
    sw = ScoreWorker(score_queue, scores_h5)
    sw.start()
    score_threads.append(sw)

  #################################################################
  # predict scores, write output

  # initialize predictions stream
  preds_stream = stream.PredStream(seqnn_model, seqs_gen, params['train']['batch_size'])

  # predictions index
  pi = 0

  for si in range(num_seqs):
    print('Predicting %d' % si, flush=True)

    # collect sequence predictions
    seq_preds = []
    for spi in range(preds_per_seq):
      preds_subset = preds_stream[pi][...,target_indexes]
      seq_preds.append(preds_subset)
      pi += 1

    # wait for previous to finish
    score_queue.join()

    # queue sequence for scoring
    score_queue.put((seqs_dna[si], seq_preds, si))

    # queue sequence for plotting
    if options.plots:
      plot_queue.put((seqs_dna[si], seq_preds, si))

  # finish queue
  print('Waiting for threads to finish.', flush=True)
  score_queue.join()

  # close output HDF5
  scores_h5.close()


def satmut_gen(seqs_dna, mut_start, mut_end):
  """Construct generator for 1 hot encoded saturation
     mutagenesis DNA sequences."""

  for seq_dna in seqs_dna:
    # 1 hot code DNA
    seq_1hot = dna_io.dna_1hot(seq_dna, n_random=True)
    yield seq_1hot

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


class PlotWorker(Thread):
  """Compute summary statistics and write to HDF."""
  def __init__(self, plot_queue, out_dir):
    Thread.__init__(self)
    self.queue = plot_queue
    self.daemon = True
    self.out_dir = out_dir

  def run(self):
    while True:
      # unload predictions
      seq_dna, seq_preds, si = self.queue.get()
      print('Plotting %d' % si, flush=True)

      # communicate finished task
      self.queue.task_done()


class ScoreWorker(Thread):
  """Compute summary statistics and write to HDF."""
  def __init__(self, score_queue, scores_h5):
    Thread.__init__(self)
    self.queue = score_queue
    self.daemon = True
    self.scores_h5 = scores_h5

  def run(self):
    while True:
      try:
        # unload predictions
        seq_dna, seq_preds, si = self.queue.get()
        print('Writing %d' % si, flush=True)

        # seq_preds is (1 + 3*mut_len) x (target_len) x (num_targets)
        seq_preds = np.array(seq_preds)
        num_preds = seq_preds.shape[0]
        num_targets = seq_preds.shape[-1]

        # reverse engineer mutagenesis position parameters
        mut_len = (num_preds - 1) // 3
        mut_mid = len(seq_dna) // 2
        mut_start = mut_mid - mut_len//2
        mut_end = mut_start + mut_len

        # one hot code mutagenized DNA
        seq_dna_mut = seq_dna[mut_start:mut_end]
        seq_1hot_mut = dna_io.dna_1hot(seq_dna_mut)

        # initialize scores
        seq_scores = np.zeros((mut_len, 4, num_targets), dtype='float32')

        # sum across length
        seq_preds_sum = seq_preds.sum(axis=1, dtype='float32')

        # predictions index (starting at first mutagenesis)
        pi = 1

        # for each mutated position
        for mi in range(mut_len):
          # for each nucleotide
          for ni in range(4):
            if seq_1hot_mut[mi,ni]:
              # reference score
              seq_scores[mi,ni,:] = seq_preds_sum[0,:]
            else:
              # mutation score
              seq_scores[mi,ni,:] = seq_preds_sum[pi,:]
              pi += 1

        # normalize positions
        seq_scores -= seq_scores.mean(axis=1, keepdims=True)

        # write to HDF5
        self.scores_h5['scores'][si,:,:,:] = seq_scores.astype('float16')
        self.scores_h5['seqs'][si,:,:] = seq_1hot_mut

      except:
        # communicate error
        print('ERROR: Sequence %d failed' % si, file=sys.stderr, flush=True)

      # communicate finished task
      self.queue.task_done()


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
