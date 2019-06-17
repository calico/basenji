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
from cooltools.lib.numutils import observed_over_expected


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
  (options, args) = parser.parse_args()

  if len(args) != 3:
    parser.error('')
  else:
    genome_hic_file = args[0]
    seqs_bed_file = args[1]
    seqs_4C_file = args[2]

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
  seqs_4C_open.create_dataset('seqs_4C', shape=(num_seqs, seq_len_pool), dtype='float16')

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
      
      # interpolate
      seq_hic_raw = interpolateNearest(seq_hic_raw)
      # todo: offer a complete interpolation after the blacklist or after the generation of the 4C profile if obs/exp not used

      # find minimum nonzero value
      seq_hic_min = np.min(seq_hic_raw[seq_hic_raw > 0])
      seq_hic_raw += seq_hic_min
      # todo: pass an option to add a certain pseudocount value, 
      # todo: or pass adaptive_coarsegrain(ar, countar, cutoff=1, max_levels=8) https://github.com/mirnylab/cooltools/blob/master/cooltools/lib/numutils.py

      # set blacklist to NaNs
      if mseq.chr in black_chr_trees:
        for black_interval in black_chr_trees[mseq.chr][mseq.start:mseq.end]:
          # adjust for sequence indexes
          black_seq_start = (black_interval.begin - mseq.start)//pool_width
          black_seq_end =   int(  np.ceil( (black_interval.end - mseq.start)/pool_width ) )
          seq_hic_raw[:,black_seq_start:black_seq_end] = np.nan
          seq_hic_raw[black_seq_start:black_seq_end,:] = np.nan
     
      # todo: pass an option to save raw values to TFR instead of obs/exp
      # if not obsexp, interpolate blacklist & NaNs (aka local baseline for a distant-dependent signal)

      # todo: raise a warning if there are a ton of interpolated nans, and maybe print possibly bad regions to some file?
      # todo: maybe print discrepancy between # HiC nans and the passed blacklist
      
      # compute observed/expected
      seq_hic_nan = np.isnan(seq_hic_raw)
      seq_hic_obsexp = observed_over_expected(seq_hic_raw, ~seq_hic_nan)[0]

      # log
      seq_hic_obsexp = np.log(seq_hic_obsexp)

      # set nan to 0=
      seq_hic_obsexp = np.nan_to_num(seq_hic_obsexp)
 
      # todo: make clip an option for obs/exp 4C, but not otherwise
      seq_hic_obsexp = np.clip(seq_hic_obsexp,-2,2)
     
      seq_hic = seq_hic_obsexp.copy()

    except ValueError:
      print("WARNING: %s doesn't see %s. Setting to all zeros." % (genome_hic_file, mseq_str))
      seq_hic_obsexp = np.zeros((seq_len_pool,seq_len_pool), dtype='float16')


    seq_4C = np.nanmean( seq_hic[len(seq_hic)//2-1:len(seq_hic)//2+1,:],axis=0)

    # write
    seqs_4C_open['seqs_4C'][si,:] = seq_4C.astype('float16')


  # close sequences coverage file
  seqs_4C_open.close()


def interpolateNearest(mat):
  badBins = np.sum(np.isnan(mat),axis=0)==len(mat)
  singletons =(((np.sum(np.isnan(mat),axis=0)==len(mat)) * smooth(np.sum(np.isnan(mat),axis=0)!=len(mat),3  )) )  > 1/3
  locs = np.zeros(np.shape(mat)); locs[singletons,:]=1; locs[:,singletons] = 1
  bb_minus_single = (badBins.astype('int8')-singletons.astype('int8')).astype('bool')
  locs[bb_minus_single,:]=0; locs[:,bb_minus_single] = 0
  locs = np.nonzero(locs)#np.isnan(mat))
  interpvals = np.zeros(np.shape(mat))
  for loc in zip(locs[0], locs[1]):
    i,j = loc
    if loc[0] > loc[1]:
      if loc[0]>0 and loc[1] > 0 and loc[0] < len(mat)-2 and loc[1]< len(mat)-2:
        interpvals[i,j] = np.nanmean(  [mat[i-1,j-1],mat[i+1,j+1]])
  interpvals = interpvals+interpvals.T
  mat2 = np.copy(mat)
  mat2[np.nonzero(interpvals)] = interpvals[np.nonzero(interpvals)]
  return mat2

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
