#!/usr/bin/env python
# Copyright 2020 Calico LLC
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
import pdb
import pickle
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd
import pysam
import tensorflow as tf
if tf.__version__[0] == '1':
  tf.compat.v1.enable_eager_execution()

from basenji import dna_io
from basenji import seqnn
from basenji import vcf as bvcf
from basenji_sad import initialize_output_h5, write_pct, write_snp

'''
sonnet_sad.py

Compute SNP Activity Difference (SAD) scores for SNPs in a VCF file,
using a saved Sonnet model.
'''

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <model> <vcf_file>'
  parser = OptionParser(usage)
  parser.add_option('-c', dest='slice_center',
      default=None, type='int',
      help='Slice center positions [Default: %default]')
  parser.add_option('-f', dest='genome_fasta',
      default='%s/data/hg38.fa' % os.environ['BASENJIDIR'],
      help='Genome FASTA for sequences [Default: %default]')
  parser.add_option('-n', dest='norm_file',
      default=None,
      help='Normalize SAD scores')
  parser.add_option('-o',dest='out_dir',
      default='sad',
      help='Output directory for tables and plots [Default: %default]')
  parser.add_option('-p', dest='processes',
      default=None, type='int',
      help='Number of processes, passed by multi script')
  parser.add_option('--pseudo', dest='log_pseudo',
      default=1, type='float',
      help='Log2 pseudocount [Default: %default]')
  parser.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Average forward and reverse complement predictions [Default: %default]')
  parser.add_option('--shifts', dest='shifts',
      default='0', type='str',
      help='Ensemble prediction shifts [Default: %default]')
  parser.add_option('--species', dest='species',
      default='human')
  parser.add_option('--stats', dest='sad_stats',
      default='SAD',
      help='Comma-separated list of stats to save. [Default: %default]')
  parser.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')
  (options, args) = parser.parse_args()

  if len(args) == 2:
    # single worker
    model_file = args[0]
    vcf_file = args[1]

  elif len(args) == 3:
    # multi separate
    options_pkl_file = args[0]
    model_file = args[1]
    vcf_file = args[2]

    # save out dir
    out_dir = options.out_dir

    # load options
    options_pkl = open(options_pkl_file, 'rb')
    options = pickle.load(options_pkl)
    options_pkl.close()

    # update output directory
    options.out_dir = out_dir

  elif len(args) == 4:
    # multi worker
    options_pkl_file = args[0]
    model_file = args[1]
    vcf_file = args[2]
    worker_index = int(args[3])

    # load options
    options_pkl = open(options_pkl_file, 'rb')
    options = pickle.load(options_pkl)
    options_pkl.close()

    # update output directory
    options.out_dir = '%s/job%d' % (options.out_dir, worker_index)

  else:
    parser.error('Must provide model and VCF file')

  if not os.path.isdir(options.out_dir):
    os.mkdir(options.out_dir)

  options.shifts = [int(shift) for shift in options.shifts.split(',')]
  options.sad_stats = options.sad_stats.split(',')


  #################################################################
  # read parameters and targets

  if options.targets_file is None:
    target_slice = None
  else:
    targets_df = pd.read_csv(options.targets_file, sep='\t', index_col=0)
    target_ids = targets_df.identifier
    target_labels = targets_df.description
    target_slice = targets_df.index

  #################################################################
  # setup model

  seqnn_model = tf.saved_model.load(model_file).model

  # query num model targets 
  seq_length = seqnn_model.predict_on_batch.input_signature[0].shape[1]
  null_1hot = np.zeros((1,seq_length,4))
  null_preds = seqnn_model.predict_on_batch(null_1hot)
  null_preds = null_preds[options.species].numpy()
  _, targets_length, num_targets = null_preds.shape

  if options.targets_file is None:
    target_ids = ['t%d' % ti for ti in range(num_targets)]
    target_labels = ['']*len(target_ids)

  #################################################################
  # load SNPs

  # filter for worker SNPs
  if options.processes is not None:
    # determine boundaries
    num_snps = bvcf.vcf_count(vcf_file)
    worker_bounds = np.linspace(0, num_snps, options.processes+1, dtype='int')

    # read SNPs form VCF
    snps = bvcf.vcf_snps(vcf_file, start_i=worker_bounds[worker_index],
      end_i=worker_bounds[worker_index+1])

  else:
    # read SNPs form VCF
    snps = bvcf.vcf_snps(vcf_file)

  num_snps = len(snps)

  # open genome FASTA
  genome_open = pysam.Fastafile(options.genome_fasta)

  # create SNP sequence generator
  def snp_gen():
    for snp in snps:
      # get SNP sequences
      snp_1hot_list = bvcf.snp_seq1(snp, seq_length, genome_open)
      for snp_1hot in snp_1hot_list:
        yield snp_1hot

  #################################################################
  # setup output

  sad_out = initialize_output_h5(options.out_dir, options.sad_stats,
                                 snps, target_ids, target_labels, targets_length)

  #################################################################
  # predict SNP scores, write output

  # initialize predictions stream
  preds_stream = PredStreamGen(seqnn_model, snp_gen(),
    rc=options.rc, shifts=options.shifts,
    slice_center=options.slice_center, species=options.species)

  # predictions index
  pi = 0

  for si in range(num_snps):
    # get predictions
    ref_preds = preds_stream[pi]
    pi += 1
    alt_preds = preds_stream[pi]
    pi += 1

    # process SNP
    write_snp(ref_preds, alt_preds, sad_out, si,
              options.sad_stats, options.log_pseudo)

  # close genome
  genome_open.close()

  ###################################################
  # compute SAD distributions across variants

  write_pct(sad_out, options.sad_stats)
  sad_out.close()


class PredStreamGen:
  """ Interface to acquire predictions via a buffered stream mechanism
        rather than getting them all at once and using excessive memory.
        Accepts generator and constructs stream batches from it. """
  def __init__(self, model, seqs_gen, batch_size=4, stream_size=32,
               rc=False, shifts=[0], slice_center=None, 
               species='human', verbose=False):
    self.model = model
    self.seqs_gen = seqs_gen
    self.batch_size = batch_size
    self.stream_size = stream_size
    self.rc = rc
    self.shifts = shifts
    self.ensembled = len(self.shifts) + int(self.rc)*len(self.shifts)
    self.slice_center = slice_center
    self.species = species
    self.verbose = verbose

    self.stream_start = 0
    self.stream_end = 0


  def __getitem__(self, i):
    # acquire predictions, if needed
    if i >= self.stream_end:
      # update start
      self.stream_start = self.stream_end

      if self.verbose:
        print('Predicting from %d' % self.stream_start, flush=True)

      # get next sequences
      seqs_1hot = self.next_seqs()

      # predict stream
      stream_preds = []
      si = 0
      while si < seqs_1hot.shape[0]:
        spreds = self.model.predict_on_batch(seqs_1hot[si:si+self.batch_size])
        spreds = spreds[self.species].numpy()
        stream_preds.append(spreds)
        si += self.batch_size
      stream_preds = np.concatenate(stream_preds, axis=0)
      
      # slice center
      if self.slice_center is not None:
        _, seq_len, _ = stream_preds.shape
        mid_pos = seq_len // 2
        slice_start = mid_pos - self.slice_center//2
        slice_end = slice_start + self.slice_center
        stream_preds = stream_preds[:,slice_start:slice_end,:]

      # average ensemble
      ens_seqs, seq_len, num_targets = stream_preds.shape
      num_seqs = ens_seqs // self.ensembled
      stream_preds = np.reshape(stream_preds,
          (num_seqs, self.ensembled, seq_len, num_targets))
      self.stream_preds = stream_preds.mean(axis=1)

      # update end
      self.stream_end = self.stream_start + self.stream_preds.shape[0]

    return self.stream_preds[i - self.stream_start]

  def next_seqs(self):
    """ Construct array of sequences for this stream chunk. """

    # extract next sequences from generator
    seqs_1hot = []
    stream_end = self.stream_start+self.stream_size
    for si in range(self.stream_start, stream_end):
      try:
        seqs_1hot.append(self.seqs_gen.__next__())
      except StopIteration:
        continue

    # initialize ensemble
    seqs_1hot_ens = []

    # add rc/shifts
    for seq_1hot in seqs_1hot:
      for shift in self.shifts:
        seq_1hot_aug = dna_io.hot1_augment(seq_1hot, shift=shift)
        seqs_1hot_ens.append(seq_1hot_aug)
        if self.rc:
          seq_1hot_aug = dna_io.hot1_rc(seq_1hot_aug)
          seqs_1hot_ens.append(seq_1hot_aug)

    seqs_1hot_ens = np.array(seqs_1hot_ens, dtype='float32')
    return seqs_1hot_ens

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
