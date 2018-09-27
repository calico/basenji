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

import os
import pickle
import sys

import h5py
import numpy as np
import pandas as pd
import pysam
import pyBigWig
import tensorflow as tf

import basenji.dna_io as dna_io
from basenji import params
from basenji import seqnn
from basenji.stream import PredStream

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
      default='', help='Comma-separated list of target indexes to write BigWigs')
  parser.add_option('-f', dest='genome_fasta',
      default='%s/assembly/ucsc/hg38.fa' % os.environ['HG38'],
      help='Genome FASTA for sequences [Default: %default]')
  parser.add_option('-g', dest='genome_file',
      default='%s/assembly/ucsc/human.hg38.genome' % os.environ['HG38'],
      help='Chromosome length information [Default: %default]')
  parser.add_option('-l', dest='mid_len',
      default=256, type='int',
      help='Length of center sequence to sum predictions for [Default: %default]')
  parser.add_option('-o', dest='out_h5_file',
      default='sat_bed.h5', help='Output HDF5 [Default: %default]')
  parser.add_option('--plots', dest='plots',
      default=False, action='store_true',
      help='Make heatmap plots [Default: %default]')
  parser.add_option('-p', dest='processes',
      default=None, type='int',
      help='Number of processes, passed by multi script')
  parser.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Ensemble forward and reverse complement predictions [Default: %default]')
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

    # update output hdf5
    options.out_h5_file = options.out_h5_file.replace('.h5', '_job%d.h5'%worker_index)

  else:
    parser.error('Must provide parameter and model files and BED file')

  options.shifts = [int(shift) for shift in options.shifts.split(',')]
  options.bigwig_indexes = [int(bi) for bi in options.bigwig_indexes.split(',')]

  if len(options.bigwig_indexes) > 0:
    bigwig_dir = os.path.splitext(options.out_h5_file)[0]
    if not os.path.isdir(bigwig_dir):
      os.mkdir(bigwig_dir)

  #################################################################
  # read parameters and collet target information

  job = params.read_job_params(params_file)

  num_targets = np.sum(job['num_targets'])
  if options.targets_file is None:
    target_subset = None
  else:
    targets_df = pd.read_table(options.targets_file)
    target_subset = targets_df.index
    if len(target_subset) == num_targets:
      target_subset = None
    else:
      num_targets = len(target_subset)


  #################################################################
  # sequence dataset

  # read sequences from BED
  seqs_dna, seqs_coords = bed_seqs(bed_file, options.genome_fasta, job['seq_length'])

  # filter for worker SNPs
  if options.processes is not None:
    worker_bounds = np.linspace(0, len(seqs_dna), options.processes+1, dtype='int')
    seqs_dna = seqs_dna[worker_bounds[worker_index]:worker_bounds[worker_index+1]]
    seqs_coords = seqs_coords[worker_bounds[worker_index]:worker_bounds[worker_index+1]]

  num_seqs = len(seqs_dna)

  # make data ops
  data_ops = seq_data_ops(seqs_dna, job['batch_size'])

  #################################################################
  # setup model

  # build model
  model = seqnn.SeqNN()
  model.build_sad(job, data_ops, ensemble_rc=options.rc,
                  ensemble_shifts=options.shifts,
                  target_subset=target_subset)

  #################################################################
  # setup output

  if os.path.isfile(options.out_h5_file):
    os.remove(options.out_h5_file)
  out_h5 = h5py.File(options.out_h5_file, 'w')
  out_h5.create_dataset('preds', shape=(num_seqs, num_targets), dtype='float16')

  # store sequence coordinates
  seqs_chr, seqs_start, _ = zip(*seqs_coords)
  seqs_chr = np.array(seqs_chr, dtype='S')
  seqs_start = np.array(seqs_start)
  seqs_end = seqs_start + job['seq_length']
  out_h5.create_dataset('chrom', data=seqs_chr)
  out_h5.create_dataset('start', data=seqs_start)
  out_h5.create_dataset('end', data=seqs_end)

  if model.preds_length % 2 == 0:
    # sum center two
    mid_start = model.preds_length//2 - 1
    mid_end = mid_start + 2
  else:
    # take center one
    mid_start = model.preds_length//2
    mid_end = mid_start + 1


  #################################################################
  # predict scores, write output

  # initialize saver
  saver = tf.train.Saver()

  with tf.Session() as sess:
    # coordinator
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(coord=coord)

    # load variables into session
    saver.restore(sess, model_file)

    # initialize predictions stream
    preds_stream = PredStream(sess, model, 64)

    for si in range(num_seqs):
      print('Predicting %d' % si, flush=True)

      # predict
      preds_full = preds_stream[si]

      # slice middle and summarize
      preds = preds_full[mid_start:mid_end,:].sum(axis=0)

      # write
      out_h5['preds'][si] = preds

      # write bigwig
      for ti in options.bigwig_indexes:
        bw_file = '%s/s%d_t%d.bw' % (bigwig_dir, si, ti)
        bigwig_write(preds_full[:,ti], seqs_coords[si], bw_file,
                     options.genome_file, model.hp.batch_buffer)

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


def bigwig_write(signal, seq_coords, bw_file, genome_file, seq_buffer=0):
  """ Write a signal track to a BigWig file over the region
         specified by seqs_coords.

    Args
     signal:      Sequences x Length signal array
     seq_coords:  (chr,start,end)
     bw_file:     BigWig filename
     genome_file: Chromosome lengths file
     seq_buffer:  Length skipped on each side of the region.
    """
  target_length = len(signal)

  # open bigwig
  bw_out = bigwig_open(bw_file, genome_file)

  # initialize entry arrays
  entry_starts = []
  entry_ends = []

  # set entries
  chrm, start, end = seq_coords
  preds_pool = (end - start - 2 * seq_buffer) // target_length

  bw_start = start + seq_buffer
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

def bed_seqs(bed_file, fasta_file, seq_len):
  """Extract and extend BED sequences to seq_len."""
  fasta_open = pysam.Fastafile(fasta_file)

  seqs_dna = []
  seqs_coords = []

  for line in open(bed_file):
    a = line.split()
    chrm = a[0]
    start = int(a[1])
    end = int(a[2])

    # determine sequence limits
    mid = (start + end) // 2
    seq_start = mid - seq_len//2
    seq_end = seq_start + seq_len

    # save
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

    # append
    seqs_dna.append(seq_dna)

  fasta_open.close()

  return seqs_dna, seqs_coords


def seq_data_ops(seqs_dna, batch_size):
  """Construct 1 hot encoded DNA sequences for tf.data."""

  # make sequence generator
  def seqs_gen():
    for seq_dna in seqs_dna:
      # 1 hot code DNA
      seq_1hot = dna_io.dna_1hot(seq_dna)
      yield {'sequence':seq_1hot}

  # auxiliary info
  seq_len = len(seqs_dna[0])
  seqs_types = {'sequence': tf.float32}
  seqs_shapes = {'sequence': tf.TensorShape([tf.Dimension(seq_len),
                                            tf.Dimension(4)])}

  # create dataset
  dataset = tf.data.Dataset().from_generator(seqs_gen,
                                             output_types=seqs_types,
                                             output_shapes=seqs_shapes)
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(2*batch_size)

  # make iterator ops
  iterator = dataset.make_one_shot_iterator()
  data_ops = iterator.get_next()

  return data_ops


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
