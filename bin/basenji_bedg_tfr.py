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
import subprocess
import time

import glob
import h5py
from intervaltree import IntervalTree
from natsort import natsorted
import numpy as np
import pandas as pd
import tensorflow as tf

from basenji import bed
from basenji import dataset
from basenji import seqnn

if tf.__version__[0] == '1':
  tf.compat.v1.enable_eager_execution()

"""
basenji_bedg_tfr.py

Produce BedGraph tracks for model predictions and targets from TFRecords.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <model_file> <data_dir>'
  parser = OptionParser(usage)
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

  # read targets
  if options.targets_file is None:
    options.targets_file = '%s/targets.txt' % data_dir
  targets_df = pd.read_csv(options.targets_file, index_col=0, sep='\t')
  target_slice = targets_df.index

  # read model parameters
  with open(params_file) as params_open:
    params = json.load(params_open)
  params_model = params['model']
  params_train = params['train']

  # set strand pairs
  if 'strand_pair' in targets_df.columns:
    params_model['strand_pair'] = [np.array(targets_df.strand_pair)]

  # construct eval data
  eval_data = dataset.SeqDataset(data_dir,
    split_label=options.split_label,
    batch_size=params_train['batch_size'],
    mode='eval',
    tfr_pattern=options.tfr_pattern)

  # initialize model
  seqnn_model = seqnn.SeqNN(params_model)
  seqnn_model.restore(model_file, options.head_i)
  seqnn_model.build_slice(target_slice)
  seqnn_model.build_ensemble(options.rc, options.shifts)

  #######################################################
  # predict
  '''
  # compute predictions
  test_preds = seqnn_model.predict(eval_data)

  # read targets
  test_targets = eval_data.numpy(return_inputs=False, target_slice=target_slice)

  # write bedgraph
  bed.write_bedgraph(test_preds, test_targets, data_dir,
    options.out_dir, options.split_label)
  '''
  #######################################################
  # collapse

  # merge sequences into nonoverlapping contigs
  seqs_bed_file = '%s/sequences.bed' % data_dir
  ctgs_bed_file = '%s/contigs.bed' % options.out_dir
  ctgs_cmd = 'grep %s %s' % (options.split_label, seqs_bed_file)
  ctgs_cmd += ' | bedtools sort -i -'
  ctgs_cmd += ' | bedtools merge -i - > %s' % ctgs_bed_file
  subprocess.call(ctgs_cmd, shell=True)
  ctgs_df = pd.read_csv(ctgs_bed_file, sep='\t', names=['chr', 'start', 'end'])
  ctgs_df['len'] = ctgs_df.end - ctgs_df.start

  for bedg_file in glob.glob('%s/*.bedgraph' % options.out_dir):

    # map chromosomes and intervals to contig indexes
    # initialize contig data structures
    chr_chr_idx = {}
    ctgs_val = []
    ctgs_ct = []
    for ci, ctg in ctgs_df.iterrows():
      # add chr
      if ctg.chr not in chr_chr_idx:  
        chr_chr_idx[ctg.chr] = IntervalTree()

      # map index
      chr_chr_idx[ctg.chr][ctg.start:ctg.end] = ci

      # initialize arrays
      ctgs_val.append(np.zeros(ctg.len, dtype='float32'))
      ctgs_ct.append(np.zeros(ctg.len, dtype='uint16'))

    # add intervals
    for line in open(bedg_file):
      a = line.rstrip().split('\t')
      ichrm = a[0]
      istart = int(a[1])
      iend = int(a[2])
      ivalue = float(a[3])

      # find contig index
      ctg_ovls = list(chr_chr_idx[ichrm][istart:iend])
      assert(len(ctg_ovls) == 1)
      ci = ctg_ovls[0].data

      # update arrays
      cstart = ctgs_df.iloc[ci].start
      astart = istart - cstart
      aend = iend - cstart
      ctgs_val[ci][astart:aend] += ivalue
      ctgs_ct[ci][astart:aend] += 1

    # collapse and write intervals
    bedgc_file = bedg_file.replace('.bedgraph','c.bedgraph')
    bedgc_out = open(bedgc_file, 'w')

    for ci, ctg in ctgs_df.iterrows():
      # initialize active interval
      active_start = 0
      active_end = 0
      active_value = None
      active_count = None

      cpos = ctg.start
      for cv, cc in zip(ctgs_val[ci], ctgs_ct[ci]):
        if cv == active_value and cc == active_count:
          # extend current interval
          active_end += 1
        else:
          if active_value is not None:
            # write interval
            active_mean = active_value / active_count
            cols = [ctg.chr, str(active_start), str(active_end), '%.5f'%active_mean]
            print('\t'.join(cols), file=bedgc_out)

          # start new interval          
          active_start = cpos
          active_end = cpos+1
          active_value = cv
          active_count = cc

        cpos += 1

    # write final interval
    active_mean = active_value / active_count
    cols = [ctg.chr, str(active_start), str(active_end), '%.5f'%active_mean]
    print('\t'.join(cols), file=bedgc_out)

    bedgc_out.close()


''' contig intervaltree
    for line in open(ctgs_bed_file):
      a = line.rstrip().split('\t')
      cchrm = a[0]
      cstart = int(a[1])
      cend = int(a[2])
      
      print('%s:%d-%d' % (cchrm, cstart, cend), flush=True)
      interval_values = IntervalTree()
      interval_counts = IntervalTree()

      # add intervals
      t0 = time.time()
      print(' Adding intervals...', flush=True)
      for line in open(bedg_file):
        a = line.rstrip().split('\t')
        ichrm = a[0]
        istart = int(a[1])
        iend = int(a[2])
        if ichrm == cchrm and cstart <= istart and iend <= cend:          
          ivalue = float(a[3])
          interval_values[istart:iend] = ivalue
          interval_counts[istart:iend] = 1
      print('DONE in %ds' % (time.time()-t0))

      # split overlaps
      t0 = time.time()
      print(' Splitting overlaps...', flush=True)
      interval_values.split_overlaps()
      interval_counts.split_overlaps()
      print('DONE in %ds' % (time.time()-t0))

      # merge and sum
      t0 = time.time()
      print(' Summing...', flush=True)
      interval_values.merge_equals(lambda a, b: a+b)
      interval_counts.merge_equals(lambda a, b: a+b)
      print('DONE in %ds' % (time.time()-t0))

      # write
      t0 = time.time()
      print(' Writing...', flush=True)
      bedgc_file = bedg_file.replace('.bedgraph','c.bedgraph')
      bedgc_out = open(bedgc_file, 'w')

      for iv, ic in zip(interval_values, interval_counts):
        interval_mean = iv.data / ic.data
        cols = [cchrm, str(iv.begin), str(iv.end), '%.5f'%interval_mean]
        print('\t'.join(cols), file=bedgc_out)

      bedgc_out.close()
      print('DONE in %ds' % (time.time()-t0))
'''

"""
def merge_intervals():
  active_intervals = []
  active_chrm = None
  active_line = 0
  for line in open(bedgs_file):
    a = line.split()
    chrm = a[0]
    start = int(a[1])
    end = int(a[2])
    value = float(a[3])

    if chrm != active_chrm:
      for ai in active_intervals:
        ai.write(bedgc_out)

    else:
      # write closed intervals / maintain active
      active_line = start
      maintain_intervals = []
      for ai in active_intervals:
        if ai.end <= active_line:
          ai.write(bedgc_out)
        else:
          maintain_intervals.append(ai)
      
      # update active intervals with new
      update_intervals = []
      for ai in maintain_intervals:
        if start < ai.end:
          # break

      active_intervals = update_intervals


class Interval:
  def __init__(self, chrm, start, end, value):
    self.chrm = chrm
    self.start = start
    self.end = end
    self.value = value
  
  def write(self, bedg_out):
    cols = [self.chrm, str(self.start), str(self.end), '%.5f'%self.value]
    print('\t'.join(cols), file=bedg_out)
"""

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
