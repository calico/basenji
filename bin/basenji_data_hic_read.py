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

import h5py
import numpy as np

import cooler
from cooltools.lib.numutils import observed_over_expected

from basenji_data import ModelSeq

"""
basenji_data_read.py

Read sequence values from coverage files.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <genome_hic_file> <seqs_bed_file> <seqs_hic_file>'
  parser = OptionParser(usage)
  parser.add_option('-w',dest='pool_width',
      default=1, type='int',
      help='Average pooling width [Default: %default]')
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
    model_seqs.append(ModelSeq(a[0],int(a[1]),int(a[2])))

  # compute dimensions
  num_seqs = len(model_seqs)
  seq_len_nt = model_seqs[0].end - model_seqs[0].start
  seq_len_pool = seq_len_nt // options.pool_width

  # initialize sequences Hi-C file
  seqs_hic_open = h5py.File(seqs_hic_file, 'w')
  seqs_hic_open.create_dataset('seqs_hic', shape=(num_seqs, seq_len_pool, seq_len_pool), dtype='float16')

  # open genome Hi-C file
  genome_hic_cool = cooler.Cooler(genome_hic_file)

  # assert that resolution matches
  assert(options.pool_width, genome_hic_cool.info['bin-size'])

  # for each model sequence
  for si in range(num_seqs):
    mseq = model_seqs[si]

    # read Hi-C
    try:
      # pull values
      mseq_str = '%s:%d-%s' % mseq.chr, mseq.start, mseq.end
      seq_hic_raw = genome_hic_cool.matrix(balance=True).fetch(mseq_str)

      # interpolate
      seq_hic_raw = interpolateNearest(seq_hic_raw)

      # compute observed/expected
      seq_hic_nan = np.isnan(seq_hic_raw)
      seq_hic_obsexp = observed_over_expected(seq_hic_raw+1e-9, ~seq_hic_nan)[0]

    except:
      print("WARNING: %s doesn't see %s:%d-%d. Setting to all zeros." %
            (genome_hic_file, mseq.chr, mseq.start, mseq.end))
      seq_hic_obsexp = np.zeros((seq_len_pool,seq_len_pool), dtype='float16')

    # write
    seqs_hic_open['seqs_hic'][si,:] = seq_hic_obsexp

  # close sequences coverage file
  seqs_hic_open.close()


def interpolateNearest(mat):
  badBins = np.sum(np.isnan(mat),axis=0)==len(mat)
  singletons =(((np.sum(np.isnan(mat),axis=0)==len(mat)) * smooth(np.sum(np.isnan(mat),axis=0)!=len(mat),3  )) )  > 1/3
  locs = np.zeros(np.shape(mat)); locs[singletons,:]=1; locs[:,singletons] = 1
  locs[badBins-singletons,:]=0; locs[:,badBins-singletons] = 0
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

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
