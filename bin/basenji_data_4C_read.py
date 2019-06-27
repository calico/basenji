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
import sys

import h5py
import numpy as np
import pyBigWig
import intervaltree

from basenji_data import ModelSeq

# hic imports
import astropy.convolution as astroconv
import cooler
from cooltools.lib.numutils import observed_over_expected, adaptive_coarsegrain, interpolate_bad_singletons, set_diag, interp_nan
import pandas as pd


"""
basenji_data_4C_read.py

Read virtual 4C profiles from coolers.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <genome_hic_file> <seqs_bed_file> <seqs_4C_file>'
  parser = OptionParser(usage)
  parser.add_option('-b', dest='blacklist_bed',
      help='Set blacklist nucleotides to a baseline value.')
  parser.add_option('-c', dest='clip',
      default=None, type='float',
      help='Clip values post-summary to a maximum [Default: %default]')
  parser.add_option('-s', dest='scale',
      default=1., type='float',
      help='Scale values by [Default: %default]')
  parser.add_option('--soft', dest='soft_clip',
      default=False, action='store_true',
      help='Soft clip values, applying sqrt to the execess above the threshold [Default: %default]')
  parser.add_option('-u', dest='sum_stat',
      default='sum',
      help='Summary statistic to compute in windows [Default: %default]')
  parser.add_option('-w',dest='pool_width',
      default=1, type='int',
      help='Average pooling width [Default: %default]')
  parser.add_option('--as_obsexp',dest='as_obsexp',
      default=False,action="store_true",
      help='save targets as obsexp profiles')

  (options, args) = parser.parse_args()

  if len(args) != 3:
    parser.error('')
  else:
    genome_hic_file = args[0]
    seqs_bed_file = args[1]
    seqs_4C_file = args[2]

  print('saving TFRs as obsexp:',options.as_obsexp)

  # read model sequences
  model_seqs = []
  for line in open(seqs_bed_file):
    a = line.split()
    model_seqs.append(ModelSeq(a[0],int(a[1]),int(a[2]),None))

  # read blacklist regions
  black_chr_trees = read_blacklist(options.blacklist_bed)

  # compute dimensions
  num_seqs = len(model_seqs)
  seq_len_nt = model_seqs[0].end - model_seqs[0].start
  seq_len_pool = seq_len_nt // options.pool_width

  # initialize sequences coverage file
  seqs_4C_open = h5py.File(seqs_4C_file, 'w')
  seqs_4C_open.create_dataset('seqs_cov', shape=(num_seqs, seq_len_pool), dtype='float16')

  # open genome coverage file
  genome_hic_cool = cooler.Cooler(genome_hic_file)

  # check for "chr" prefix
  chr_pre = 'chr1' in genome_hic_cool.chromnames

  # assert that resolution matches
  assert(options.pool_width == genome_hic_cool.info['bin-size'])

  # for each model sequence
  for si in range(num_seqs):
    mseq = model_seqs[si]

    try:
      # pull hic values
      if chr_pre:
        mseq_str = '%s:%d-%d' % (mseq.chr, mseq.start, mseq.end)
      else:
        mseq_str = '%s:%d-%d' % (mseq.chr[3:], mseq.start, mseq.end)
      #print('mseq_str:', mseq_str)
      
      seq_hic_raw = genome_hic_cool.matrix(balance=True).fetch(mseq_str)
      seq_hic_nan = np.isnan(seq_hic_raw)
      if np.sum(  seq_hic_nan[len(seq_hic_nan)//2-1:len(seq_hic_nan)//2+1, len(seq_hic_nan)//2-2:len(seq_hic_nan)//2+2 ]) > 4:
        print("WARNING: %s lots of zeros, check that umap_midpoint is correct %s. " % (genome_hic_file, mseq_str))

      # set blacklist to NaNs
      if mseq.chr in black_chr_trees:
        for black_interval in black_chr_trees[mseq.chr][mseq.start:mseq.end]:
          # adjust for sequence indexes
          black_seq_start = (black_interval.begin - mseq.start)// options.pool_width
          black_seq_end =   int(  np.ceil( (black_interval.end - mseq.start)/ options.pool_width ) )
          seq_hic_raw[:,black_seq_start:black_seq_end] = np.nan
          seq_hic_raw[black_seq_start:black_seq_end,:] = np.nan
        seq_hic_nan = np.isnan(seq_hic_raw)

      seq_hic_smoothed =  adaptive_coarsegrain(seq_hic_raw, genome_hic_cool.matrix(balance=False).fetch(mseq_str),  cutoff=.5, max_levels=8)
      
      #todo: pass an option to add a certain pseudocount value, or the minimum nonzero value
      #seq_hic_min = np.min(seq_hic_raw[seq_hic_raw > 0])
      #seq_hic_raw += seq_hic_min


      if options.as_obsexp == True:
        # interpolate single missing bins
        seq_hic_interpolated =  interpolate_bad_singletons(seq_hic_smoothed, mask=(~seq_hic_nan),
                                                 fillDiagonal=True, returnMask=False, secondPass=True,verbose=False)
        seq_hic_nan = np.isnan(seq_hic_interpolated)

        # compute observed/expected
        seq_hic_obsexp = observed_over_expected(seq_hic_interpolated, ~seq_hic_nan)[0]
        # todo: allow passing a global expected rather than computing locally

        # log
        seq_hic_obsexp = np.log(seq_hic_obsexp)

        # set nan to 0
        seq_hic_obsexp = np.nan_to_num(seq_hic_obsexp)

        # todo: make clip an option for obs/exp 4C, but not otherwise
        seq_hic_obsexp = np.clip(seq_hic_obsexp,-2,2)

        # take the mean
        seq_4C = np.nanmean( seq_hic_obsexp[len(seq_hic_obsexp)//2-1:len(seq_hic_obsexp)//2+1,:],axis=0)
      
      else:
        # interpolate all missing bins
        seq_hic_interpolated =  interp_nan(seq_hic_smoothed)

        # clip first diagonals and high values
        clipval = np.nanmedian(np.diag(seq_hic_interpolated,2))
        for i in [-1,0,1]: set_diag(seq_hic_interpolated,clipval,i)
        seq_hic_interpolated = np.clip(seq_hic_interpolated, 0, clipval)

        # take the mean, rescale
        seq_4C = 100000*np.nanmean( seq_hic_interpolated[len(seq_hic_interpolated)//2-1:len(seq_hic_interpolated)//2+1,:],axis=0)

    except ValueError:
      print("WARNING: %s doesn't see %s. Setting to all zeros." % (genome_hic_file, mseq_str))
      seq_4C = np.zeros((seq_len_pool,), dtype='float16')
    
    # write
    seqs_4C_open['seqs_cov'][si,:] = seq_4C.astype('float16')
    
  # close sequences coverage file
  seqs_4C_open.close()

def smooth(y, box_pts):
  box = np.ones(box_pts)/box_pts
  y_smooth = astroconv.convolve(y, box, boundary='extend') # also: None, fill, wrap, extend
  return y_smooth



def read_blacklist(blacklist_bed, black_buffer=20):
  """Construct interval trees of blacklist
     regions for each chromosome."""
  black_chr_trees = {}

  if blacklist_bed is not None and os.path.isfile(blacklist_bed):
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
