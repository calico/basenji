#!/usr/bin/env python
# Copyright 2017 Calico LLC

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

from optparse import OptionParser

import os

import h5py
import numpy as np
import intervaltree

from basenji_data import ModelSeq

"""
basenji_data_read.py

Read sequence values from coverage files.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <genome_cov_file> <seqs_bed_file> <seqs_cov_file>'
  parser = OptionParser(usage)
  parser.add_option('-b', dest='blacklist_bed',
      help='Set blacklist nucleotides to a baseline value.')
  parser.add_option('-w',dest='pool_width',
      default=1, type='int',
      help='Average pooling width [Default: %default]')
  (options, args) = parser.parse_args()

  if len(args) != 3:
    parser.error('')
  else:
    genome_cov_file = args[0]
    seqs_bed_file = args[1]
    seqs_cov_file = args[2]

  # read model sequences
  model_seqs = []
  for line in open(seqs_bed_file):
    a = line.split()
    model_seqs.append(ModelSeq(a[0],int(a[1]),int(a[2])))

  # read blacklist regions
  black_chr_trees = read_blacklist(options.blacklist_bed)

  # compute dimensions
  num_seqs = len(model_seqs)
  seq_len_nt = model_seqs[0].end - model_seqs[0].start
  seq_len_pool = seq_len_nt // options.pool_width

  # initialize sequences coverage file
  seqs_cov_open = h5py.File(seqs_cov_file, 'w')
  seqs_cov_open.create_dataset('seqs_cov', shape=(num_seqs, seq_len_pool), dtype='float16')

  # open genome coverage file
  genome_cov_open = h5py.File(genome_cov_file, 'r')

  # for each model sequence
  for si in range(num_seqs):
    mseq = model_seqs[si]

    # read coverage
    if mseq.chr in genome_cov_open:
      seq_cov_nt = genome_cov_open[mseq.chr][mseq.start:mseq.end]
    else:
      print("WARNING: %s doesn't see %s:%d-%d. Setting to all zeros." %
            (cov_file, mseq.chr, mseq.start, mseq.end))
      seq_cov_nt = np.zeros(mseq.end - mseq.start, dtype='float16')

    # determine baseline coverage
    baseline_cov = np.percentile(seq_cov_nt, 10)

    # set blacklist to baseline
    if mseq.chr in black_chr_trees:
      for black_interval in black_chr_trees[mseq.chr][mseq.start:mseq.end]:
        # adjust for sequence indexes
        black_seq_start = black_interval.begin - mseq.start
        black_seq_end = black_interval.end - mseq.start
        seq_cov_nt[black_seq_start:black_seq_end] = baseline_cov

    # set NaN's to baseline
    nan_mask = np.isnan(seq_cov_nt)
    seq_cov_nt[nan_mask] = baseline_cov

    # sum pool
    seq_cov = seq_cov_nt.reshape(seq_len_pool, options.pool_width)
    seq_cov = seq_cov.sum(axis=1, dtype='float32')

    # write
    seqs_cov_open['seqs_cov'][si,:] = seq_cov

  # close genome coverage file
  genome_cov_open.close()

  # close sequences coverage file
  seqs_cov_open.close()


def read_blacklist(blacklist_bed, black_buffer=5):
  """Construct interval trees of blacklist
     regions for each chromosome."""
  black_chr_trees = {}

  if os.path.isfile(blacklist_bed):
    for line in open(blacklist_bed):
      a = line.split()
      chrm = a[0]
      start = max(0, int(a[1]) - black_buffer)
      end = int(a[2]) + black_buffer

      if chrm not in black_chr_trees:
        black_chr_trees[chrm] = intervaltree.IntervalTree()

      black_chr_trees[chrm][start:end] = True

  return black_chr_trees

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
