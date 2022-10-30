#!/usr/bin/env python
# Copyright 2021 Calico LLC
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
import subprocess
import time

import glob
import h5py
from intervaltree import IntervalTree
from natsort import natsorted
import numpy as np
import pandas as pd
import pyBigWig
import tensorflow as tf
from tqdm import tqdm

from basenji import dataset
from basenji import seqnn
from borzoi_sed import targets_prep_strand

"""
basenji_bigwig_tfr.py

Produce BigWig tracks for model predictions and targets from TFRecords.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <model_file> <data_dir>'
  parser = OptionParser(usage)
  parser.add_option('-g', dest='genome_file',
      default=None,
      help='Chromosome length information [Default: %default]')
  parser.add_option('--head', dest='head_i',
      default=0, type='int',
      help='Parameters head to test [Default: %default]')
  parser.add_option('-o', dest='out_dir',
      default='bedg_out',
      help='Output directory for test statistics [Default: %default]')
  parser.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Average the fwd and rc predictions [Default: %default]')
  parser.add_option('--shifts', dest='shifts',
      default='0',
      help='Ensemble prediction shifts [Default: %default]')
  parser.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')
  parser.add_option('--split', dest='split_label',
      default='test',
      help='Dataset split label for eg TFR pattern [Default: %default]')
  parser.add_option('--tfr', dest='tfr_pattern',
      default=None,
      help='TFR pattern string appended to data_dir/tfrecords for subsetting [Default: %default]')
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
  options.shifts = [int(shift) for shift in options.shifts.split(',')]

  #######################################################
  # inputs

  # read data parameters
  with open('%s/statistics.json'%data_dir) as data_open:
    data_stats = json.load(data_open)
  pool_width = data_stats['pool_width']

  # read targets
  if options.targets_file is None:
    options.targets_file = '%s/targets.txt' % data_dir
  targets_df = pd.read_csv(options.targets_file, index_col=0, sep='\t')
  num_targets = targets_df.shape[0]
  target_slice = targets_df.index

  # read model parameters
  with open(params_file) as params_open:
    params = json.load(params_open)
  params_model = params['model']
  params_train = params['train']

  # set strand pairs
  if 'strand_pair' in targets_df.columns:
    # re-index if not the full targets set
    orig_new_index = dict(zip(targets_df.index, np.arange(targets_df.shape[0])))
    targets_strand_pair = np.array([orig_new_index[ti] for ti in targets_df.strand_pair])
    params_model['strand_pair'] = [targets_strand_pair]

  # construct eval data
  eval_data = dataset.SeqDataset(data_dir,
    split_label=options.split_label,
    batch_size=1,
    mode='eval',
    tfr_pattern=options.tfr_pattern)

  # sequences
  seqs_bed_file = '%s/sequences.bed' % data_dir
  seqs_df = pd.read_csv(seqs_bed_file, sep='\t', names=['chr', 'start', 'end', 'split'])
  seqs_df = seqs_df[seqs_df.split == options.split_label]

  # initialize model
  seqnn_model = seqnn.SeqNN(params_model)
  seqnn_model.restore(model_file, options.head_i)
  seqnn_model.build_slice(target_slice)
  seqnn_model.build_ensemble(options.rc, options.shifts)

  #######################################################
  # prepare bigwigs

  # read chromosome sizes
  chrom_sizes = []
  for line in open(options.genome_file):
    a = line.split()
    chrom_sizes.append((a[0], int(a[1])))

  preds_bigwig = []
  targs_bigwig = []

  for ti in range(num_targets):
    # open bigwigs
    preds_out = pyBigWig.open('%s/preds_t%d.bw' % (options.out_dir, ti), 'w')
    targs_out = pyBigWig.open('%s/targs_t%d.bw' % (options.out_dir, ti), 'w')

    # add chromosome lengths header
    preds_out.addHeader(chrom_sizes)
    targs_out.addHeader(chrom_sizes)

    # append
    preds_bigwig.append(preds_out)
    targs_bigwig.append(targs_out)


  #######################################################
  # aggregate predictions/targets by contig

  t0 = time.time()
  print('Aggregating predictions/targets by contig...', flush=True)

  # read sequence positions
  seqs_df = pd.read_csv('%s/sequences.bed'%data_dir, sep='\t',
    names=['chr','start','end','split'])
  seqs_df = seqs_df[seqs_df.split == options.split_label]

  # merge sequences into nonoverlapping contigs
  seqs_bed_file = '%s/sequences.bed' % data_dir
  ctgs_bed_file = '%s/contigs.bed' % options.out_dir
  ctgs_cmd = 'grep %s %s' % (options.split_label, seqs_bed_file)
  ctgs_cmd += ' | bedtools sort -i -'
  ctgs_cmd += ' | bedtools merge -i - > %s' % ctgs_bed_file
  subprocess.call(ctgs_cmd, shell=True)
  ctgs_df = pd.read_csv(ctgs_bed_file, sep='\t', names=['chr', 'start', 'end'])
  ctgs_df['len'] = ctgs_df.end - ctgs_df.start

  # map chromosomes and intervals to contig indexes
  # initialize contig data structures
  chr_chr_idx = {}
  ctgs_preds = []
  ctgs_targs = []
  ctgs_ct = []
  for ci, ctg in ctgs_df.iterrows():
    # add chr
    if ctg.chr not in chr_chr_idx:  
      chr_chr_idx[ctg.chr] = IntervalTree()

    # map index
    chr_chr_idx[ctg.chr][ctg.start:ctg.end] = ci

    # initialize arrays
    ctgs_preds.append(np.zeros((ctg.len, num_targets), dtype='float16'))
    ctgs_targs.append(np.zeros((ctg.len, num_targets), dtype='float16'))
    ctgs_ct.append(np.zeros(ctg.len, dtype='uint8'))

  # add intervals
  si = 0
  for x, y in tqdm(eval_data.dataset):
    print(si, flush=True)

    # predictions/targets
    y = y.numpy()[0]
    y = y[:,target_slice]
    yh = seqnn_model.predict(x)[0]

    # map sequence
    seq_chr = seqs_df.iloc[si].chr
    seq_start = seqs_df.iloc[si].start
    seq_end = seqs_df.iloc[si].end

    # find contig index
    ctg_ovls = list(chr_chr_idx[seq_chr][seq_start:seq_start+1])
    assert(len(ctg_ovls) == 1)
    ci = ctg_ovls[0].data
    ctg_start = ctgs_df.iloc[ci].start

    ctg_seq_start = seq_start - ctg_start
    ctg_seq_end = seq_end - ctg_start

    # repeat to nt resolution
    y_nt = np.repeat(y, pool_width, axis=0)
    yh_nt = np.repeat(yh, pool_width, axis=0)

    # updates values
    ctgs_preds[ci][ctg_seq_start:ctg_seq_end] += yh_nt
    ctgs_targs[ci][ctg_seq_start:ctg_seq_end] += y_nt
    ctgs_ct[ci][ctg_seq_start:ctg_seq_end] += 1

    si += 1

  print('DONE IN %ds' % (time.time()-t0))

  #######################################################
  # write bigwigs

  t0 = time.time()
  print('Writing BigWigs...', flush=True)

  # sort the contigs
  ctgs_df.sort_values(['chr','start'], inplace=True)

  chrom_sizes = []
  for line in open(options.genome_file):
    a = line.split()
    chrom_sizes.append((a[0], int(a[1])))
  chrom_sizes = sorted(chrom_sizes)

  for ti in range(num_targets):
    print('Targets %d' % ti, flush=True)

    # open bigwigs
    preds_out = pyBigWig.open('%s/preds_t%d.bw' % (options.out_dir, ti), 'w')
    targs_out = pyBigWig.open('%s/targs_t%d.bw' % (options.out_dir, ti), 'w')

    # add chromosome lengths header
    preds_out.addHeader(chrom_sizes)
    targs_out.addHeader(chrom_sizes)

    for ci, ctg in ctgs_df.iterrows():
      # normalize values by count
      ctgs_upreds = ctgs_preds[ci][:,ti] / ctgs_ct[ci]
      ctgs_utargs = ctgs_targs[ci][:,ti] / ctgs_ct[ci]

      # write into BigWig
      preds_out.addEntries(ctg.chr, ctg.start, values=ctgs_upreds, span=1, step=1)
      targs_out.addEntries(ctg.chr, ctg.start, values=ctgs_utargs, span=1, step=1)

    preds_out.close()
    targs_out.close()

  print('DONE IN %ds' % (time.time()-t0))


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
