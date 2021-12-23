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

import json
import os
import sys
import time
import subprocess

from intervaltree import IntervalTree
import numpy as np
import pandas as pd
import pysam

from basenji import dna_io

################################################################################
# bed.py
#
# Methods to work with BED files.
################################################################################

def make_bed_seqs(bed_file, fasta_file, seq_len, stranded=False):
  """Return BED regions as sequences and regions as a list of coordinate
  tuples, extended to a specified length."""
  """Extract and extend BED sequences to seq_len."""
  fasta_open = pysam.Fastafile(fasta_file)

  seqs_dna = []
  seqs_coords = []

  for line in open(bed_file):
    a = line.split()
    chrm = a[0]
    start = int(float(a[1]))
    end = int(float(a[2]))
    if len(a) >= 6:
      strand = a[5]
    else:
      strand = '+'

    # determine sequence limits
    mid = (start + end) // 2
    seq_start = mid - seq_len//2
    seq_end = seq_start + seq_len

    # save
    if stranded:
      seqs_coords.append((chrm,seq_start,seq_end,strand))
    else:
      seqs_coords.append((chrm,seq_start,seq_end))

    # initialize sequence
    seq_dna = ''

    # add N's for left over reach
    if seq_start < 0:
      print('Adding %d Ns to %s:%d-%s' % \
          (-seq_start,chrm,start,end), file=sys.stderr)
      seq_dna = 'N'*(-seq_start)
      seq_start = 0

    # get dna
    seq_dna += fasta_open.fetch(chrm, seq_start, seq_end).upper()

    # add N's for right over reach
    if len(seq_dna) < seq_len:
      print('Adding %d Ns to %s:%d-%s' % \
          (seq_len-len(seq_dna),chrm,start,end), file=sys.stderr)
      seq_dna += 'N'*(seq_len-len(seq_dna))

    # reverse complement
    if stranded and strand == '-':
      seq_dna = dna_io.dna_rc(seq_dna)

    # append
    seqs_dna.append(seq_dna)

  fasta_open.close()

  return seqs_dna, seqs_coords


def read_bed_coords(bed_file, seq_len):
  """Return BED regions as a list of coordinate
  tuples, extended to a specified length."""
  seqs_coords = []

  for line in open(bed_file):
    a = line.split()
    chrm = a[0]
    start = int(float(a[1]))
    end = int(float(a[2]))

    # determine sequence limits
    mid = (start + end) // 2
    seq_start = mid - seq_len//2
    seq_end = seq_start + seq_len

    # save
    seqs_coords.append((chrm,seq_start,seq_end))

  return seqs_coords


def write_bedgraph(test_preds, test_targets, data_dir, out_dir, split_label, bedgraph_indexes=None):
  """Write BED graph files for predictions and targets."""
  
  # get shapes
  num_seqs, target_length, num_targets = test_targets.shape

  # set bedgraph indexes
  if bedgraph_indexes is None:
    bedgraph_indexes = np.arange(num_targets)

  # read data parameters
  with open('%s/statistics.json'%data_dir) as data_open:
    data_stats = json.load(data_open)
    pool_width = data_stats['pool_width']
    # crop_bp = data_stats['crop_bp']

  # read sequence positions
  seqs_df = pd.read_csv('%s/sequences.bed'%data_dir, sep='\t',
    names=['chr','start','end','split'])
  seqs_df = seqs_df[seqs_df.split == split_label]

  # initialize output directory
  os.makedirs('%s/bedgraph' % out_dir, exist_ok=True)

  for ti in bedgraph_indexes:
    print('Writing %d bedgraph...' % ti, end='')
    t0 = time.time()

    # slice preds/targets
    test_preds_ti = test_preds[:,:,ti]
    test_targets_ti = test_targets[:,:,ti]

    # initialize raw predictions/targets
    preds_out = open('%s/bedgraph/preds_t%d.bedgraph' % (out_dir, ti), 'w')
    targets_out = open('%s/bedgraph/targets_t%d.bedgraph' % (out_dir, ti), 'w')

    # write raw predictions/targets
    for si in range(num_seqs):
      seq_chr = seqs_df.iloc[si].chr

      # ignore crop for new datasets
      bin_start = seqs_df.iloc[si].start # + crop_bp
      for bi in range(target_length):
        bin_end = bin_start + pool_width
        cols = [seq_chr, str(bin_start), str(bin_end), str(test_preds_ti[si,bi])]
        print('\t'.join(cols), file=preds_out)
        cols = [seq_chr, str(bin_start), str(bin_end), str(test_targets_ti[si,bi])]
        print('\t'.join(cols), file=targets_out)
        bin_start = bin_end

    preds_out.close()
    targets_out.close()

    print('DONE in %ds' % (time.time()-t0))


def write_bedgraph_v1(test_preds, test_targets, data_dir, out_dir, split_label, bedgraph_indexes=None):
  """Write BED graph files for predictions and targets."""
  
  # get shapes
  num_seqs, target_length, num_targets = test_targets.shape

  # set bedgraph indexes
  if bedgraph_indexes is None:
    bedgraph_indexes = np.arange(num_targets)

  # read data parameters
  with open('%s/statistics.json'%data_dir) as data_open:
    data_stats = json.load(data_open)
    pool_width = data_stats['pool_width']
    crop_bp = data_stats['crop_bp']

  # read sequence positions
  seqs_df = pd.read_csv('%s/sequences.bed'%data_dir, sep='\t',
    names=['chr','start','end','split'])
  seqs_df = seqs_df[seqs_df.split == split_label]

  # initialize output directory
  os.makedirs('%s/bedgraph' % out_dir, exist_ok=True)

  for ti in bedgraph_indexes:
    print('Writing %d bedgraph...' % ti, end='')
    t0 = time.time()

    # slice preds/targets
    test_preds_ti = test_preds[:,:,ti]
    test_targets_ti = test_targets[:,:,ti]

    # initialize predictions/targets
    preds_out = open('%s/bedgraph/preds_t%d.bedgraph' % (out_dir, ti), 'w')
    targets_out = open('%s/bedgraph/targets_t%d.bedgraph' % (out_dir, ti), 'w')

    # save written
    intervals_written = {}

    # write predictions/targets
    for si in range(num_seqs):
      seq_chr = seqs_df.iloc[si].chr
      if seq_chr not in intervals_written:
        intervals_written[seq_chr] = IntervalTree()

      bin_start = seqs_df.iloc[si].start + crop_bp
      for bi in range(target_length):
        bin_end = bin_start + pool_width
        if intervals_written[seq_chr][bin_start:bin_end]:
          pass
        else:
          intervals_written[seq_chr][bin_start:bin_end] = True
          cols = [seq_chr, str(bin_start), str(bin_end), str(test_preds_ti[si,bi])]
          print('\t'.join(cols), file=preds_out)
          cols = [seq_chr, str(bin_start), str(bin_end), str(test_targets_ti[si,bi])]
          print('\t'.join(cols), file=targets_out)
        bin_start = bin_end

    preds_out.close()
    targets_out.close()

    print('DONE in %ds' % (time.time()-t0))