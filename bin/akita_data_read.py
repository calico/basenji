#!/usr/bin/env python
# Copyright 2019 Calico LLC

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

import pdb
import os
import sys

import h5py
import numpy as np
import pandas as pd
import pyBigWig
import intervaltree

from basenji_data import ModelSeq
from basenji_data_read import read_blacklist

# hic imports
import cooler
from cooltools.lib.numutils import observed_over_expected, adaptive_coarsegrain
from cooltools.lib.numutils import interpolate_bad_singletons, set_diag, interp_nan
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve

"""
akita_data_read.py

Read and pre-process Hi-C/uC data from cooler.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <genome_hic_file> <seqs_bed_file> <seqs_hic_file>'
  parser = OptionParser(usage)
  parser.add_option('-b', dest='blacklist_bed',
      help='Set blacklist nucleotides to a baseline value.')
  parser.add_option('--clip', dest='clip',
      default=None, type='float',
      help='Clip values post-summary to a maximum [Default: %default]')
  parser.add_option('--crop', dest='crop_bp',
      default=0, type='int',
      help='Crop bp off each end [Default: %default]')
  parser.add_option('-d', dest='diagonal_offset',
      default=2, type='int',
      help='Positions on the diagonal to ignore [Default: %default]')
  parser.add_option('-k', dest='kernel_stddev',
      default=0, type='int',
      help='Gaussian kernel stddev to smooth values [Default: %default]')
  # parser.add_option('-s', dest='scale',
  #     default=1., type='float',
  #     help='Scale values by [Default: %default]')
  parser.add_option('-w',dest='pool_width',
      default=1, type='int',
      help='Average pooling width [Default: %default]')
  parser.add_option('--as_obsexp',dest='as_obsexp',
      default=False,action="store_true",
      help='save targets as obsexp profiles')
  parser.add_option('--global_obsexp',dest='global_obsexp',
      default=False,action="store_true",
      help='use global obs/exp')
  parser.add_option('--no_log',dest='no_log',
      default=False,action="store_true",
      help='no not take log for obs/exp')

  (options, args) = parser.parse_args()

  if len(args) != 3:
    parser.error('')
  else:
    genome_hic_file = args[0]
    seqs_bed_file = args[1]
    seqs_hic_file = args[2]

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

  if options.crop_bp == 0:
    seq_len_crop = seq_len_pool
  else:
    crop_start = options.crop_bp // options.pool_width
    crop_end = seq_len_pool - crop_start
    seq_len_crop = seq_len_pool - 2*crop_start

  # compute upper triangular indexes
  triu_tup = np.triu_indices(seq_len_crop, options.diagonal_offset)
  seq_len_nodiag = seq_len_crop - options.diagonal_offset
  seq_len_hic = seq_len_nodiag*(seq_len_nodiag + 1) // 2

  # initialize sequences coverage file
  seqs_hic_open = h5py.File(seqs_hic_file, 'w')
  seqs_hic_open.create_dataset('targets', shape=(num_seqs, seq_len_hic), dtype='float16')

  if options.kernel_stddev > 0:
    # initialize Gaussian kernel
    kernel = Gaussian2DKernel(x_stddev=options.kernel_stddev)
  else:
    kernel = None

  # open genome coverage file
  genome_hic_cool = cooler.Cooler(genome_hic_file)

  if options.global_obsexp:
    try:
      print('loading by-chromosome expected')
      genome_hic_expected = pd.read_csv(genome_hic_file.replace('.cool','.expected'), sep='\t')
    except:
      print('not found: '+genome_hic_file.replace('cool','expected'))
      raise ValueError('invalid expected file')
   
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
      num_filtered_bins = np.sum(np.sum(seq_hic_nan,axis=0) == len(seq_hic_nan))
      if num_filtered_bins > (.5*len(seq_hic_nan)):
        print("WARNING: %s >50%% bins filtered, check:  %s. " % (genome_hic_file, mseq_str))

      # set blacklist to NaNs
      if mseq.chr in black_chr_trees:
        for black_interval in black_chr_trees[mseq.chr][mseq.start:mseq.end]:
          # adjust for sequence indexes
          black_seq_start = (black_interval.begin - mseq.start)// options.pool_width
          black_seq_end =   int(  np.ceil( (black_interval.end - mseq.start)/ options.pool_width ) )
          seq_hic_raw[:,black_seq_start:black_seq_end] = np.nan
          seq_hic_raw[black_seq_start:black_seq_end,:] = np.nan
        seq_hic_nan = np.isnan(seq_hic_raw)

      # clip first diagonals and high values
      clipval = np.nanmedian(np.diag(seq_hic_raw,options.diagonal_offset))
      for i in range(-options.diagonal_offset+1,options.diagonal_offset):
        set_diag(seq_hic_raw, clipval, i)
      seq_hic_raw = np.clip(seq_hic_raw, 0, clipval)
      seq_hic_raw[seq_hic_nan] = np.nan

      # adaptively coarsegrain based on raw counts
      seq_hic_smoothed = adaptive_coarsegrain(
                              seq_hic_raw,
                              genome_hic_cool.matrix(balance=False).fetch(mseq_str),
                              cutoff=2, max_levels=8)
      seq_hic_nan = np.isnan(seq_hic_smoothed)
      #todo: pass an option to add a certain pseudocount value, or the minimum nonzero value

      if options.as_obsexp:
        # compute obs/exp        
        if options.global_obsexp: # compute global obs/exp
          exp_chr = genome_hic_expected.iloc[ genome_hic_expected['chrom'].values ==mseq.chr][0:seq_len_pool]
          if len(exp_chr) ==0: 
              raise ValueError('no expected values found for chr:'+mseq.chr)
          exp_map= np.zeros((seq_len_pool,seq_len_pool))
          for i in range(seq_len_pool):
            set_diag(exp_map,exp_chr['balanced.avg'].values[i],i)
            set_diag(exp_map,exp_chr['balanced.avg'].values[i],-i)
          seq_hic_obsexp = seq_hic_smoothed / exp_map
          for i in range(-options.diagonal_offset+1,options.diagonal_offset): set_diag(seq_hic_obsexp,1.0,i)
          seq_hic_obsexp[seq_hic_nan] = np.nan          

        else: # compute local obs/exp
          seq_hic_obsexp = observed_over_expected(seq_hic_smoothed, ~seq_hic_nan)[0]

        # log
        if options.no_log==False:
          seq_hic_obsexp = np.log(seq_hic_obsexp)
          if options.clip is not None:
            seq_hic_obsexp = np.clip(seq_hic_obsexp, -options.clip, options.clip)
          seq_hic_obsexp = interp_nan(seq_hic_obsexp)
          for i in range(-options.diagonal_offset+1, options.diagonal_offset): set_diag(seq_hic_obsexp, 0,i)
        else:
          if options.clip is not None:
            seq_hic_obsexp = np.clip(seq_hic_obsexp, 0, options.clip)
          seq_hic_obsexp = interp_nan(seq_hic_obsexp)
          for i in range(-options.diagonal_offset+1, options.diagonal_offset): set_diag(seq_hic_obsexp, 1,i)

        # apply kernel
        if kernel is not None:
          seq_hic = convolve(seq_hic_obsexp, kernel)
        else:
          seq_hic = seq_hic_obsexp

      else:
        # interpolate all missing bins
        seq_hic_interpolated = interp_nan(seq_hic_smoothed)

        # rescale, reclip
        seq_hic = 100000*seq_hic_interpolated
        clipval = np.nanmedian(np.diag(seq_hic,options.diagonal_offset))
        for i in range(-options.diagonal_offset+1, options.diagonal_offset):
          set_diag(seq_hic,clipval,i)
        seq_hic = np.clip(seq_hic, 0, clipval)

        #extra smoothing. todo pass kernel specs
        if kernel is not None:
          seq_hic = convolve(seq_hic, kernel)

    except ValueError:
      print("WARNING: %s doesn't see %s. Setting to all zeros." % (genome_hic_file, mseq_str))
      seq_hic = np.zeros((seq_len_pool,seq_len_pool), dtype='float16')

    # crop
    if options.crop_bp > 0:
      seq_hic = seq_hic[crop_start:crop_end,:]
      seq_hic = seq_hic[:,crop_start:crop_end]

    # unroll upper triangular
    seq_hic = seq_hic[triu_tup]

    # write
    seqs_hic_open['targets'][si,:] = seq_hic.astype('float16')

  # close sequences coverage file
  seqs_hic_open.close()

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
