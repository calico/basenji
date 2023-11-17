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
import pickle
import random
import sys

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
from borzoi_satg_gene import unaugment_grads

'''
sonnet_satg_bed.py

Perform an in silico saturation mutagenesis of sequences in a BED file.
'''

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <model> <bed_file>'
  parser = OptionParser(usage)
  parser.add_option('-c', dest='slice_center',
      default=False, action='store_true',
      help='Slice center position(s) for gradient [Default: %default]')
  parser.add_option('-d', dest='mut_down',
      default=0, type='int',
      help='Nucleotides downstream of center sequence to mutate [Default: %default]')
  parser.add_option('-f', dest='genome_fasta',
      default='%s/data/hg38.fa' % os.environ['BASENJIDIR'],
      help='Genome FASTA for sequences [Default: %default]')
  parser.add_option('-l', dest='mut_len',
      default=0, type='int',
      help='Length of center sequence to mutate [Default: %default]')
  parser.add_option('-o', dest='out_dir',
      default='sat_mut', help='Output directory [Default: %default]')
  parser.add_option('-p', dest='processes',
      default=None, type='int',
      help='Number of processes, passed by multi script')
  parser.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Ensemble forward and reverse complement predictions [Default: %default]')
  parser.add_option('--shifts', dest='shifts',
      default='0', type='str',
      help='Ensemble prediction shifts [Default: %default]')
  parser.add_option('--species', dest='species',
      default='human')
  parser.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')
  parser.add_option('-u', dest='mut_up',
      default=0, type='int',
      help='Nucleotides upstream of center sequence to mutate [Default: %default]')
  (options, args) = parser.parse_args()

  if len(args) == 2:
    # single worker
    model_file = args[0]
    bed_file = args[1]

  elif len(args) == 3:
    # master script
    options_pkl_file = args[0]
    model_file = args[1]
    bed_file = args[2]

    # load options
    options_pkl = open(options_pkl_file, 'rb')
    options = pickle.load(options_pkl)
    options_pkl.close()

  elif len(args) == 4:
    # multi worker
    options_pkl_file = args[0]
    model_file = args[1]
    bed_file = args[2]
    worker_index = int(args[3])

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

  if options.mut_up > 0 or options.mut_down > 0:
    options.mut_len = options.mut_up + options.mut_down
  else:
    assert(options.mut_len > 0)
    options.mut_up = options.mut_len // 2
    options.mut_down = options.mut_len - options.mut_up

  options.shifts = [int(shift) for shift in options.shifts.split(',')]

  #################################################################
  # setup model

  seqnn_model = tf.saved_model.load(model_file).model

  # query num model targets 
  seq_length = seqnn_model.predict_on_batch.input_signature[0].shape[1]
  null_1hot = np.zeros((1,seq_length,4))
  null_preds = seqnn_model.predict_on_batch(null_1hot)
  null_preds = null_preds[options.species].numpy()
  _, preds_length, num_targets = null_preds.shape

  # read targets
  if options.targets_file is None:
    targets_mask = np.ones(num_targets, dtype='float32')
  else:
    targets_df = pd.read_table(options.targets_file, index_col=0)
    targets_mask = np.zeros(num_targets, dtype='float32')
    targets_mask[targets_df.index] = 1.0

  #################################################################
  # sequence dataset

  # read sequences from BED
  seqs_dna, seqs_coords = bed.make_bed_seqs(
    bed_file, options.genome_fasta, seq_length, stranded=True)

  # filter for worker SNPs
  if options.processes is not None:
    worker_bounds = np.linspace(0, len(seqs_dna), options.processes+1, dtype='int')
    seqs_dna = seqs_dna[worker_bounds[worker_index]:worker_bounds[worker_index+1]]
    seqs_coords = seqs_coords[worker_bounds[worker_index]:worker_bounds[worker_index+1]]

  num_seqs = len(seqs_dna)

  # determine mutation region limits
  seq_mid = seq_length // 2
  mut_start = seq_mid - options.mut_up
  mut_end = mut_start + options.mut_len

  #################################################################
  # setup output

  scores_h5_file = '%s/scores.h5' % options.out_dir
  if os.path.isfile(scores_h5_file):
    os.remove(scores_h5_file)
  scores_h5 = h5py.File(scores_h5_file, 'w')
  scores_h5.create_dataset('seqs', dtype='bool',
      shape=(num_seqs, options.mut_len, 4))
  scores_h5.create_dataset('grads', dtype='float16',
      shape=(num_seqs, options.mut_len, 4))

  # store mutagenesis sequence coordinates
  scores_chr = []
  scores_start = []
  scores_end = []
  scores_strand = []
  for seq_chr, seq_start, seq_end, seq_strand in seqs_coords:
    scores_chr.append(seq_chr)
    scores_strand.append(seq_strand)
    if seq_strand == '+':
      score_start = seq_start + mut_start
      score_end = score_start + options.mut_len
    else:
      score_end = seq_end - mut_start
      score_start = score_end - options.mut_len
    scores_start.append(score_start)
    scores_end.append(score_end)

  scores_h5.create_dataset('chr', data=np.array(scores_chr, dtype='S'))
  scores_h5.create_dataset('start', data=np.array(scores_start))
  scores_h5.create_dataset('end', data=np.array(scores_end))
  scores_h5.create_dataset('strand', data=np.array(scores_strand, dtype='S'))

  preds_per_seq = 1 + 3*options.mut_len

  #################################################################
  # predict scores, write output

  # find center
  if options.slice_center:
    center_start = preds_length // 2 - 1
    if preds_length % 2 == 0:
      center_end = center_start + 2
    else:
      center_end = center_start + 1
  else:
    center_start = 0
    center_end = preds_length    

  si = 0
  for seq_dna in seqs_dna:
    print('Predicting %d' % si, flush=True)
    seq_1hot = dna_io.dna_1hot(seq_dna)

    grad_ens = []
    for shift in options.shifts:
      seq_1hot_aug = dna_io.hot1_augment(seq_1hot, shift=shift)
      seq_1hot_tf = tf.convert_to_tensor(seq_1hot_aug, dtype=tf.float32)[tf.newaxis]
      grad_aug = input_gradients(seqnn_model, seq_1hot_tf, targets_mask,
          center_start, center_end, options.species).numpy()
      grad_aug = unaugment_grads(grad_aug, fwdrc=True, shift=shift)
      grad_ens.append(grad_aug)
      
      if options.rc:
        seq_1hot_aug = dna_io.hot1_rc(seq_1hot_aug)
        seq_1hot_tf = tf.convert_to_tensor(seq_1hot_aug, dtype=tf.float32)[tf.newaxis]
        grad_aug = input_gradients(seqnn_model, seq_1hot_tf, targets_mask,
          center_start, center_end, options.species).numpy()
        grad_aug = unaugment_grads(grad_aug, fwdrc=False, shift=shift)
        grad_ens.append(grad_aug)

    grad = np.array(grad_ens).mean(axis=0)

    # write to HDF5
    scores_h5['seqs'][si] = seq_1hot[mut_start:mut_end]
    scores_h5['grads'][si] = grad[mut_start:mut_end].astype('float16')

    si += 1

  # close output HDF5
  scores_h5.close()
    

@tf.function
def input_gradients(seqnn_model, seq_1hot_tf, targets_mask, pos_start, pos_end, species):
  # watch prediction
  targets_mass = tf.reduce_sum(targets_mask)
  with tf.GradientTape() as tape:
    tape.watch(seq_1hot_tf)
    pred_raw = seqnn_model.predict_on_batch(seq_1hot_tf)[species]

    # mean across targets
    raw2_pred = tf.reduce_sum(targets_mask[tf.newaxis]*pred_raw / targets_mass, axis=-1)

    # slice center and take mean
    grad_pred = tf.reduce_sum(raw2_pred[:,pos_start:pos_end], axis=-1)

  # compute gradient
  grad = tape.gradient(grad_pred, seq_1hot_tf)
  grad = tf.squeeze(grad, axis=0)

  # zero mean each position
  grad = grad - tf.reduce_mean(grad, axis=-1, keepdims=True)

  return grad


def unaugment_grads(grads, fwdrc=False, shift=0):
  """ Undo sequence augmentation."""
  # reverse complement
  if not fwdrc:
    # reverse
    grads = grads[::-1, :]

    # swap A and T
    grads[:, [0, 3]] = grads[:, [3, 0]]

    # swap C and G
    grads[:, [1, 2]] = grads[:, [2, 1]]

  # undo shift
  if shift < 0:
    # shift sequence right
    grads[-shift:, :] = grads[:shift, :]

    # fill in left unknowns
    grads[:-shift, :] = 0

  elif shift > 0:
    # shift sequence left
    grads[:-shift, :] = grads[shift:, :]

    # fill in right unknowns
    grads[-shift:, :] = 0

  return grads

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
