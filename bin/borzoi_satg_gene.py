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

import gc
import json
import os
import pdb
import pickle
from queue import Queue
import random
import sys
from threading import Thread
import time

import h5py
import numpy as np
import pandas as pd
import pysam
import tensorflow as tf

from basenji import dna_io
from basenji import gene as bgene
from basenji import seqnn
from borzoi_sed import targets_prep_strand

'''
borzoi_satg_gene.py

Perform a gradient saliency analysis for genes specified in a GTF file.
'''

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params> <model> <gene_gtf>'
  parser = OptionParser(usage)
  parser.add_option('-f', dest='genome_fasta',
      default='%s/data/hg38.fa' % os.environ['BASENJIDIR'],
      help='Genome FASTA for sequences [Default: %default]')
  parser.add_option('-o', dest='out_dir',
      default='satg_out', help='Output directory [Default: %default]')
  parser.add_option('-p', dest='processes',
      default=None, type='int',
      help='Number of processes, passed by multi script')
  parser.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Ensemble forward and reverse complement predictions [Default: %default]')
  parser.add_option('--shifts', dest='shifts',
      default='0', type='str',
      help='Ensemble prediction shifts [Default: %default]')
  parser.add_option('--span', dest='span',
      default=False, action='store_true',
      help='Aggregate entire gene span [Default: %default]')
  parser.add_option('--sum', dest='sum_targets',
      default=False, action='store_true',
      help='Sum targets for single output [Default: %default]')
  parser.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')
  (options, args) = parser.parse_args()

  if len(args) == 3:
    # single worker
    params_file = args[0]
    model_file = args[1]
    genes_gtf_file = args[2]

  elif len(args) == 4:
    # master script
    options_pkl_file = args[0]
    params_file = args[1]
    model_file = args[2]
    genes_gtf_file = args[3]

    # load options
    options_pkl = open(options_pkl_file, 'rb')
    options = pickle.load(options_pkl)
    options_pkl.close()

  elif len(args) == 5:
    # multi worker
    options_pkl_file = args[0]
    params_file = args[1]
    model_file = args[2]
    genes_gtf_file = args[3]
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
  # read parameters and targets

  # read model parameters
  with open(params_file) as params_open:
    params = json.load(params_open)
  params_model = params['model']
  params_train = params['train']
  seq_len = params_model['seq_length']

  if options.targets_file is None:
    parser.error('Must provide targets table to properly handle strands.')
  else:
    targets_df = pd.read_csv(options.targets_file, sep='\t', index_col=0)

  # prep strand
  orig_new_index = dict(zip(targets_df.index, np.arange(targets_df.shape[0])))
  targets_strand_pair = np.array([orig_new_index[ti] for ti in targets_df.strand_pair])
  targets_strand_df = targets_prep_strand(targets_df)
  num_targets = targets_strand_df.shape[0]
  if options.sum_targets:
    num_targets = 1

  # params strand_pair unnecessary because I'm not building ensemble in graph

  #################################################################
  # setup model

  seqnn_model = seqnn.SeqNN(params_model)
  seqnn_model.restore(model_file)
  seqnn_model.build_slice(targets_df.index, options.sum_targets)

  model_stride = seqnn_model.model_strides[0]
  model_crop = seqnn_model.target_crops[0]
  target_length = seqnn_model.target_lengths[0]

  #################################################################
  # read genes

  # parse GTF
  transcriptome = bgene.Transcriptome(genes_gtf_file)

  # order valid genes
  genome_open = pysam.Fastafile(options.genome_fasta)
  gene_list = sorted(transcriptome.genes.keys())
  num_genes = len(gene_list)

  # filter for worker genes
  if options.processes is not None:
    # determine boundaries
    worker_bounds = np.linspace(0, num_genes, options.processes+1, dtype='int')
    worker_start = worker_bounds[worker_index]
    worker_end = worker_bounds[worker_index+1]
    gene_list = [gene_list[gi] for gi in range(worker_start, worker_end)]
    num_genes = len(gene_list)

  #################################################################
  # setup output

  min_start = -model_stride*model_crop

  # choose gene sequences
  genes_chr = []
  genes_start = []
  genes_end = []
  genes_strand = []
  for gene_id in gene_list:
    gene = transcriptome.genes[gene_id]
    genes_chr.append(gene.chrom)
    genes_strand.append(gene.strand)

    gene_midpoint = gene.midpoint()
    gene_start = max(min_start, gene_midpoint - seq_len//2)
    gene_end = gene_start + seq_len
    genes_start.append(gene_start)
    genes_end.append(gene_end)

  # initialize HDF5
  scores_h5_file = '%s/scores.h5' % options.out_dir
  if os.path.isfile(scores_h5_file):
    os.remove(scores_h5_file)
  scores_h5 = h5py.File(scores_h5_file, 'w')
  scores_h5.create_dataset('seqs', dtype='bool',
      shape=(num_genes, seq_len, 4))
  scores_h5.create_dataset('grads', dtype='float16',
      shape=(num_genes, seq_len, 4, num_targets))
  scores_h5.create_dataset('gene', data=np.array(gene_list, dtype='S'))
  scores_h5.create_dataset('chr', data=np.array(genes_chr, dtype='S'))
  scores_h5.create_dataset('start', data=np.array(genes_start))
  scores_h5.create_dataset('end', data=np.array(genes_end))
  scores_h5.create_dataset('strand', data=np.array(genes_strand, dtype='S'))

  #################################################################
  # predict scores, write output 

  for gi, gene_id in enumerate(gene_list):
    print('Predicting %d, %s' % (gi, gene_id), flush=True)
    gene = transcriptome.genes[gene_id]
    
    # make sequence
    seq_1hot = make_seq_1hot(genome_open, genes_chr[gi], genes_start[gi], genes_end[gi], seq_len)

    # determine output sequence start
    seq_out_start = genes_start[gi] + model_stride*model_crop
    seq_out_len = model_stride*target_length

    # determine output positions
    gene_slice = gene.output_slice(seq_out_start, seq_out_len, model_stride, options.span)
    if options.rc:
      gene_slice_rc = target_length - gene_slice - 1

    if len(gene_slice) == 0:
      print('WARNING: %s no gene positions found.' % gene_id)
      grads = np.zeros((seq_len, 4, num_targets), dtype='float16')

    else:
      grads_ens = []
      for shift in options.shifts:
        seq_1hot_aug = dna_io.hot1_augment(seq_1hot, shift=shift)
        grads_aug = seqnn_model.gradients(seq_1hot_aug, pos_slice=gene_slice)
        grads_aug = unaugment_grads(grads_aug, fwdrc=True, shift=shift)
        grads_ens.append(grads_aug)

        if options.rc:
          seq_1hot_aug = dna_io.hot1_rc(seq_1hot_aug)
          grads_aug = seqnn_model.gradients(seq_1hot_aug, pos_slice=gene_slice_rc)
          grads_aug = unaugment_grads(grads_aug, fwdrc=False, shift=shift)
          grads_aug = grads_aug[...,targets_strand_pair]
          grads_ens.append(grads_aug)

      # ensemble mean
      grads = np.array(grads_ens).mean(axis=0)

      # slice relevant strand targets
      if genes_strand[gi] == '+':
        gene_strand_mask = (targets_df.strand != '-')
      else:
        gene_strand_mask = (targets_df.strand != '+')
      grads = grads[...,gene_strand_mask]

    # write to HDF5
    scores_h5['seqs'][gi] = seq_1hot
    scores_h5['grads'][gi] = grads

    gc.collect()

  # close files
  genome_open.close()
  scores_h5.close()    


def unaugment_grads(grads, fwdrc=False, shift=0):
  """ Undo sequence augmentation."""
  # reverse complement
  if not fwdrc:
    # reverse
    grads = grads[::-1, :, :]

    # swap A and T
    grads[:, [0, 3], :] = grads[:, [3, 0], :]

    # swap C and G
    grads[:, [1, 2], :] = grads[:, [2, 1], :]

  # undo shift
  if shift < 0:
    # shift sequence right
    grads[-shift:, :, :] = grads[:shift, :, :]

    # fill in left unknowns
    grads[:-shift, :, :] = 0

  elif shift > 0:
    # shift sequence left
    grads[:-shift, :, :] = grads[shift:, :, :]

    # fill in right unknowns
    grads[-shift:, :, :] = 0

  return grads


def make_seq_1hot(genome_open, chrm, start, end, seq_len):
  if start < 0:
    seq_dna = 'N'*(-start) + genome_open.fetch(chrm, 0, end)
  else:
    seq_dna = genome_open.fetch(chrm, start, end)
    
  # extend to full length
  if len(seq_dna) < seq_len:
    seq_dna += 'N'*(seq_len-len(seq_dna))

  seq_1hot = dna_io.dna_1hot(seq_dna)
  return seq_1hot


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
