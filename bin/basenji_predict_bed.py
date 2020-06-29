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
import pickle
import sys

import h5py
import numpy as np
import pandas as pd
import pysam
import pyBigWig
import tensorflow as tf

if tf.__version__[0] == '1':
  tf.compat.v1.enable_eager_execution()

from basenji import bed
from basenji import dna_io
from basenji import seqnn
from basenji import stream

'''
basenji_predict_bed.py

Predict sequences from a BED file.
'''

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <model_file> <bed_file>'
  parser = OptionParser(usage)
  parser.add_option('-b', dest='bigwig_indexes',
      default=None, help='Comma-separated list of target indexes to write BigWigs')
  parser.add_option('-e', dest='embed_layer',
      default=None, type='int',
      help='Embed sequences using the specified layer index.')
  parser.add_option('-f', dest='genome_fasta',
      default=None,
      help='Genome FASTA for sequences [Default: %default]')
  parser.add_option('-g', dest='genome_file',
      default=None,
      help='Chromosome length information [Default: %default]')
  parser.add_option('-l', dest='site_length',
      default=None, type='int',
      help='Prediction site length. [Default: params.seq_length]')
  parser.add_option('-o', dest='out_dir',
      default='pred_out',
      help='Output directory [Default: %default]')
  # parser.add_option('--plots', dest='plots',
  #     default=False, action='store_true',
  #     help='Make heatmap plots [Default: %default]')
  parser.add_option('-p', dest='processes',
      default=None, type='int',
      help='Number of processes, passed by multi script')
  parser.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Ensemble forward and reverse complement predictions [Default: %default]')
  parser.add_option('-s', dest='sum',
      default=False, action='store_true',
      help='Sum site predictions [Default: %default]')
  parser.add_option('--shifts', dest='shifts',
      default='0',
      help='Ensemble prediction shifts [Default: %default]')
  parser.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')
  (options, args) = parser.parse_args()

  if len(args) == 3:
    params_file = args[0]
    model_file = args[1]
    bed_file = args[2]

  elif len(args) == 5:
    # multi worker
    options_pkl_file = args[0]
    params_file = args[1]
    model_file = args[2]
    bed_file = args[3]
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

  if options.bigwig_indexes is not None:
    options.bigwig_indexes = [int(bi) for bi in options.bigwig_indexes.split(',')]
  else:
    options.bigwig_indexes = []

  if len(options.bigwig_indexes) > 0:
    bigwig_dir = '%s/bigwig' % options.out_dir
    if not os.path.isdir(bigwig_dir):
      os.mkdir(bigwig_dir)

  #################################################################
  # read parameters and collet target information

  with open(params_file) as params_open:
    params = json.load(params_open)
  params_model = params['model']

  if options.targets_file is None:
    target_slice = None
  else:
    targets_df = pd.read_table(options.targets_file, index_col=0)
    target_slice = targets_df.index

  #################################################################
  # setup model

  # initialize model
  seqnn_model = seqnn.SeqNN(params_model)
  seqnn_model.restore(model_file)
  seqnn_model.build_slice(target_slice)
  seqnn_model.build_ensemble(options.rc, options.shifts)

  if options.embed_layer is not None:
    seqnn_model.build_embed(options.embed_layer)
    _, preds_length, preds_depth  = seqnn_model.embed.output.shape
  else:
    _, preds_length, preds_depth = seqnn_model.model.output.shape
    
  if type(preds_length) == tf.compat.v1.Dimension:
    preds_length = preds_length.value
    preds_depth = preds_depth.value

  preds_window = seqnn_model.model_strides[0]
  seq_crop = seqnn_model.target_crops[0]*preds_window


  #################################################################
  # sequence dataset

  if options.site_length is None:
    options.site_length = preds_window*preds_length
    print('site_length: %d' % options.site_length)

  # construct model sequences
  model_seqs_dna, model_seqs_coords = bed.make_bed_seqs(
    bed_file, options.genome_fasta,
    params_model['seq_length'], stranded=False)

  # construct site coordinates
  site_seqs_coords = bed.read_bed_coords(bed_file, options.site_length)

  # filter for worker SNPs
  if options.processes is not None:
    worker_bounds = np.linspace(0, len(model_seqs_dna), options.processes+1, dtype='int')
    model_seqs_dna = model_seqs_dna[worker_bounds[worker_index]:worker_bounds[worker_index+1]]
    model_seqs_coords = model_seqs_coords[worker_bounds[worker_index]:worker_bounds[worker_index+1]]
    site_seqs_coords = site_seqs_coords[worker_bounds[worker_index]:worker_bounds[worker_index+1]]

  num_seqs = len(model_seqs_dna)


  #################################################################
  # setup output

  assert(preds_length % 2 == 0)
  preds_mid = preds_length // 2

  assert(options.site_length % preds_window == 0)
  site_preds_length = options.site_length // preds_window

  assert(site_preds_length % 2 == 0)
  site_preds_start = preds_mid - site_preds_length//2
  site_preds_end = site_preds_start + site_preds_length

  # initialize HDF5
  out_h5_file = '%s/predict.h5' % options.out_dir
  if os.path.isfile(out_h5_file):
    os.remove(out_h5_file)
  out_h5 = h5py.File(out_h5_file, 'w')

  # create predictions
  if options.sum:
    out_h5.create_dataset('preds', shape=(num_seqs, preds_depth), dtype='float16')
  else:
    out_h5.create_dataset('preds', shape=(num_seqs, site_preds_length, preds_depth), dtype='float16')

  # store site coordinates
  site_seqs_chr, site_seqs_start, site_seqs_end = zip(*site_seqs_coords)
  site_seqs_chr = np.array(site_seqs_chr, dtype='S')
  site_seqs_start = np.array(site_seqs_start)
  site_seqs_end = np.array(site_seqs_end)
  out_h5.create_dataset('chrom', data=site_seqs_chr)
  out_h5.create_dataset('start', data=site_seqs_start)
  out_h5.create_dataset('end', data=site_seqs_end)


  #################################################################
  # predict scores, write output

  # define sequence generator
  def seqs_gen():
    for seq_dna in model_seqs_dna:
      yield dna_io.dna_1hot(seq_dna)

  # predict
  preds_stream = stream.PredStreamGen(seqnn_model, seqs_gen(), params['train']['batch_size'])

  for si in range(num_seqs):
    preds_seq = preds_stream[si]

    # slice site
    preds_site = preds_seq[site_preds_start:site_preds_end,:]

    # write
    if options.sum:
      out_h5['preds'][si] = preds_site.sum(axis=0)
    else:
      out_h5['preds'][si] = preds_site

    # write bigwig
    for ti in options.bigwig_indexes:
      bw_file = '%s/s%d_t%d.bw' % (bigwig_dir, si, ti)
      bigwig_write(preds_seq[:,ti], model_seqs_coords[si], bw_file,
                   options.genome_file, seq_crop)

  # close output HDF5
  out_h5.close()


def bigwig_open(bw_file, genome_file):
  """ Open the bigwig file for writing and write the header. """

  bw_out = pyBigWig.open(bw_file, 'w')

  chrom_sizes = []
  for line in open(genome_file):
    a = line.split()
    chrom_sizes.append((a[0], int(a[1])))

  bw_out.addHeader(chrom_sizes)

  return bw_out


def bigwig_write(signal, seq_coords, bw_file, genome_file, seq_crop=0):
  """ Write a signal track to a BigWig file over the region
         specified by seqs_coords.

    Args
     signal:      Sequences x Length signal array
     seq_coords:  (chr,start,end)
     bw_file:     BigWig filename
     genome_file: Chromosome lengths file
     seq_crop:    Sequence length cropped from each side of the sequence.
    """
  target_length = len(signal)

  # open bigwig
  bw_out = bigwig_open(bw_file, genome_file)

  # initialize entry arrays
  entry_starts = []
  entry_ends = []

  # set entries
  chrm, start, end = seq_coords
  preds_pool = (end - start - 2 * seq_crop) // target_length

  bw_start = start + seq_crop
  for li in range(target_length):
    bw_end = bw_start + preds_pool
    entry_starts.append(bw_start)
    entry_ends.append(bw_end)
    bw_start = bw_end

  # add
  bw_out.addEntries(
          [chrm]*target_length,
          entry_starts,
          ends=entry_ends,
          values=[float(s) for s in signal])

  bw_out.close()


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
