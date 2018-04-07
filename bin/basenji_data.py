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
from __future__ import print_function

from optparse import OptionParser
from collections import OrderedDict
import math
import os
import random
import subprocess
import sys
import tempfile
import time

import h5py
import joblib
import numpy as np
import pyBigWig
import pysam

import slurm

import basenji

'''
basenji_data.py

Tile the genome and project the full functional profile to latent space
using a given model. Save the result in HDF5 for Basenji learning.

To Do:
 -If unmappable regions are cutting my data, I could squeeze a little more out
   by allowing them to sit at the edges of sequences where I'm not making
   predictions anyway.
'''

################################################################################
def main():
  usage = 'usage: %prog [options] <fasta_file> <sample_wigs_file>'
  parser = OptionParser(usage)
  parser.add_option('-b', dest='limit_bed',
      help='Limit to segments that overlap regions in a BED file')
  parser.add_option('-c', dest='clip',
      default=None, type='float',
      help='Clip target values to have minimum [Default: %default]')
  parser.add_option('-d', dest='sample_pct',
      default=1.0, type='float',
      help='Down-sample the segments')
  parser.add_option('-g', dest='gaps_file',
      help='Genome assembly gaps BED [Default: %default]')
  parser.add_option('-l', dest='seq_length',
      default=131072, type='int',
      help='Sequence length [Default: %default]')
  parser.add_option('--unmap_t', dest='unmap_t',
      default=0.3, type='float',
      help='Remove sequences with more than this unmappable bin % [Default: %default]')
  parser.add_option('-o', dest='out_dir',
      default='data_out',
      help='Output directory [Default: %default]')
  parser.add_option('-p', dest='processes',
      default=1, type='int',
      help='Number parallel processes [Default: %default]')
  parser.add_option('-s', dest='stride_train',
      type='float', default=1.,
      help='Stride to advance segments [Default: seq_length]')
  parser.add_option('-t', dest='test_pct_or_chr',
      type='str', default=0.05,
      help='Proportion of the data for testing [Default: %default]')
  parser.add_option('-u', dest='unmap_bed',
      help='Unmappable segments to set to NA')
  parser.add_option('-w', dest='pool_width',
      type='int', default=128,
      help='Sum pool width [Default: %default]')
  parser.add_option('-v', dest='valid_pct_or_chr',
      type='str', default=0.05,
      help='Proportion of the data for validation [Default: %default]')
  parser.add_option('-z', dest='compression',
      help='h5py compression [Default: %default]')
  (options, args) = parser.parse_args()

  if len(args) != 2:
    parser.error('Must provide FASTA and sample coverage labels and paths.)
  else:
    fasta_file = args[0]
    sample_wigs_file = args[1]

  random.seed(1)

  ################################################################
  # assess bigwigs
  ################################################################
  # get wig files and labels
  target_wigs = OrderedDict()
  target_strands = []
  target_labels = []
  for line in open(sample_wigs_file, encoding='UTF-8'):
    a = line.rstrip().split('\t')

    if a[0] in target_wigs:
      print('WARNING: duplicate target id %s' % a[0], file=sys.stderr)

    target_wigs[a[0]] = a[1]
    if len(a) > 2:
      target_strands.append(a[2])
    else:
      target_strands.append('*')
    if len(a) > 3:
      target_labels.append(a[3])
    else:
      target_labels.append('')

  ################################################################
  # prepare genomic segments
  ################################################################
  chrom_segments = basenji.genome.load_chromosomes(fasta_file)

  # remove gaps
  if options.gaps_file:
    chrom_segments = basenji.genome.split_contigs(chrom_segments,
                                                  options.gaps_file)

  # ditch the chromosomes
  segments = []
  for chrom in chrom_segments:
    segments += [(chrom, seg_start, seg_end)
                 for seg_start, seg_end in chrom_segments[chrom]]

  # standardize order
  segments.sort()

  # filter for large enough
  segments = [cse for cse in segments if cse[2] - cse[1] >= options.seq_length]

  # down-sample
  if options.sample_pct < 1.0:
    segments = random.sample(segments, int(options.sample_pct * len(segments)))

  # limit to a BED file
  if options.limit_bed is not None:
    segments = limit_segments(segments, options.limit_bed)

  if not os.path.isdir(options.cluster_dir):
    os.mkdir(options.cluster_dir)

  # print segments to BED file
  seg_bed_file = '%s/segments.bed' % options.cluster_dir
  seg_bed_out = open(seg_bed_file, 'w')
  for chrom, seg_start, seg_end in segments:
    print('%s\t%d\t%d' % (chrom, seg_start, seg_end), file=seg_bed_out)
  seg_bed_out.close()

  ################################################################
  # bigwig read and process
  ################################################################
  print(
      'Reading and pre-processing bigwigs for %d segments' % len(segments),
      flush=True)

  targets_real = []

  # generate numpy arrays on cluster
  jobs = []
  for target_label in target_wigs.keys():
    wig_file = target_wigs[target_label]
    npy_file = '%s/%s' % (options.cluster_dir, target_label)
    if not os.path.isfile(npy_file) and not os.path.isfile('%s.npy' % npy_file):
      print(npy_file)

      if os.path.splitext(wig_file)[1] == '.h5':
        script = 'seqs_hdf5.py'
      else:
        script = 'bigwig_hdf5.py'

      cmd = 'echo $HOSTNAME; %s -l %d -s %d -w %d %s %s %s' % (
          script, options.seq_length, options.stride, options.pool_width,
          wig_file, seg_bed_file, npy_file)
      name = 'hdf5_%s' % target_label
      outf = '%s/%s.out' % (options.cluster_dir, target_label)
      errf = '%s/%s.err' % (options.cluster_dir, target_label)
      j = slurm.Job(
          cmd, name, outf, errf, queue='standard,tbdisk', mem=15000, time='12:0:0')
      jobs.append(j)

  slurm.multi_run(jobs)

  # load into targets_real
  for target_label in target_wigs.keys():
    npy_file = '%s/%s.npy' % (options.cluster_dir, target_label)
    wig_targets = np.load(npy_file)
    targets_real.append(wig_targets)

  # transpose from TxSxL to SxLxT
  targets_real = np.transpose(np.array(targets_real), axes=(1, 2, 0))

  print('%d target sequences' % targets_real.shape[0])

  ################################################################
  # one hot code sequences
  ################################################################
  seqs_1hot, seqs_segments = segments_1hot(fasta_file, segments,
                                           options.seq_length, options.stride)
  print('%d sequences one hot coded' % seqs_1hot.shape[0])

  ################################################################
  # correct for unmappable regions
  ################################################################
  if options.unmap_bed is not None:
    seqs_na = annotate_na(seqs_segments, options.unmap_bed, options.seq_length,
                          options.pool_width)

    # determine mappable sequences and update test indexes
    map_indexes = []

    for i in range(seqs_na.shape[0]):
      # mappable
      if seqs_na[i, :].mean(dtype='float64') < options.na_t:
        map_indexes.append(i)

      # unmappable
      else:
        # forget it
        pass

    # update data structures
    targets_real = targets_real[map_indexes]

    seqs_1hot = seqs_1hot[map_indexes]
    seqs_segments = [seqs_segments[mi] for mi in map_indexes]
    seqs_na = seqs_na[map_indexes]

  ################################################################
  # write to train, valid, test HDF5
  ################################################################

  # choose test indexes
  if options.test_pct_or_chr.startswith('chr'):
    test_indexes = [
        si for si in range(len(seqs_segments))
        if seqs_segments[si][0] == options.test_pct_or_chr
    ]
  else:
    test_pct = float(options.test_pct_or_chr)
    test_indexes = [
        twi for twi in range(len(seqs_segments)) if random.random() < test_pct
    ]

  # choose valid indexes
  if options.valid_pct_or_chr.startswith('chr'):
    # valid_indexes = np.array([seq_seg[0] == options.valid_pct_or_chr for seq_seg in seqs_segments])
    valid_indexes = [
        si for si in range(len(seqs_segments))
        if seqs_segments[si][0] == options.valid_pct_or_chr
    ]
  else:
    valid_pct = float(options.valid_pct_or_chr)
    valid_n = int(valid_pct * len(seqs_segments))
    nontest_indexes = set(range(len(seqs_segments))) - set(test_indexes)
    valid_indexes = random.sample(nontest_indexes, valid_n)

  # remainder is training
  train_indexes = list(
      set(range(len(seqs_segments))) - set(valid_indexes) - set(test_indexes))

  # training may require shuffling
  random.shuffle(sorted(train_indexes))
  random.shuffle(sorted(valid_indexes))
  random.shuffle(sorted(test_indexes))

  # write to HDF5
  hdf5_out = h5py.File(hdf5_file, 'w')

  # store pooling
  hdf5_out.create_dataset('pool_width', data=options.pool_width, dtype='int')

  # store targets
  target_ids = np.array(list(target_wigs.keys()), dtype='S')
  hdf5_out.create_dataset('target_ids', data=target_ids)

  target_labels = np.array(target_labels, dtype='S')
  hdf5_out.create_dataset('target_labels', data=target_labels)

  target_strands = np.array(target_strands, dtype='S')
  hdf5_out.create_dataset('target_strands', data=target_strands)

  # HDF5 train
  hdf5_out.create_dataset(
      'train_in',
      data=seqs_1hot[train_indexes],
      dtype='bool',
      compression=options.compression)
  hdf5_out.create_dataset(
      'train_out',
      data=targets_real[train_indexes],
      dtype='float16',
      compression=options.compression)
  hdf5_out.create_dataset(
      'train_na',
      data=seqs_na[train_indexes],
      dtype='bool',
      compression=options.compression)

  # HDF5 valid
  hdf5_out.create_dataset(
      'valid_in',
      data=seqs_1hot[valid_indexes],
      dtype='bool',
      compression=options.compression)
  hdf5_out.create_dataset(
      'valid_out',
      data=targets_real[valid_indexes],
      dtype='float16',
      compression=options.compression)
  hdf5_out.create_dataset(
      'valid_na',
      data=seqs_na[valid_indexes],
      dtype='bool',
      compression=options.compression)

  # HDF5 test
  hdf5_out.create_dataset(
      'test_in',
      data=seqs_1hot[test_indexes],
      dtype='bool',
      compression=options.compression)
  hdf5_out.create_dataset(
      'test_out',
      data=targets_real[test_indexes],
      dtype='float16',
      compression=options.compression)
  hdf5_out.create_dataset(
      'test_na',
      data=seqs_na[test_indexes],
      dtype='bool',
      compression=options.compression)

  hdf5_out.close()

  # output BED file
  if options.out_bed_file:
    out_bed_out = open(options.out_bed_file, 'w')
    for si in train_indexes:
      print('%s\t%d\t%d\ttrain' % seqs_segments[si], file=out_bed_out)
    for si in valid_indexes:
      print('%s\t%d\t%d\tvalid' % seqs_segments[si], file=out_bed_out)
    for si in test_indexes:
      print('%s\t%d\t%d\ttest' % seqs_segments[si], file=out_bed_out)
    out_bed_out.close()


################################################################################
def annotate_na(seqs_segments, unmap_bed, seq_length, pool_width):
  """ Intersect the sequence segments with unmappable regions
         and annoate the segments as NaN to possible be ignored.

    Args:
      seqs_segments: list of (chrom,start,end) sequence segments
      unmap_bed: unmappable regions BED file
      seq_length: sequence length
      pool_width: pooled bin width

    Returns:
      seqs_na: NxL binary NA indicators
    """

  # print sequence segments to file
  segs_temp = tempfile.NamedTemporaryFile()
  segs_bed = segs_temp.name
  segs_out = open(segs_bed, 'w')
  for (chrom, start, end) in seqs_segments:
    print('%s\t%d\t%d' % (chrom, start, end), file=segs_out)
  segs_out.close()

  # hash segments to indexes
  segment_indexes = {}
  for i in range(len(seqs_segments)):
    segment_indexes[seqs_segments[i]] = i

  # initialize NA array
  seqs_na = np.zeros(
      (len(seqs_segments), seq_length // pool_width), dtype='bool')

  # intersect with unmappable regions
  p = subprocess.Popen(
      'bedtools intersect -wo -a %s -b %s' % (segs_bed, unmap_bed),
      shell=True,
      stdout=subprocess.PIPE)
  for line in p.stdout:
    line = line.decode('utf-8')
    a = line.split()

    seg_chrom = a[0]
    seg_start = int(a[1])
    seg_end = int(a[2])
    seg_tup = (seg_chrom, seg_start, seg_end)

    unmap_start = int(a[4])
    unmap_end = int(a[5])

    seg_unmap_start_i = math.floor((unmap_start - seg_start) / pool_width)
    seg_unmap_end_i = math.ceil((unmap_end - seg_start) / pool_width)

    # skip minor overlaps to the first
    first_start = seg_start + seg_unmap_start_i * pool_width
    first_end = first_start + pool_width
    first_overlap = first_end - unmap_start
    if first_overlap < 0.25 * pool_width:
      seg_unmap_start_i += 1

    # skip minor overlaps to the last
    last_start = seg_start + (seg_unmap_end_i - 1) * pool_width
    last_overlap = unmap_end - last_start
    if last_overlap < 0.25 * pool_width:
      seg_unmap_end_i -= 1

    seqs_na[segment_indexes[seg_tup], seg_unmap_start_i:seg_unmap_end_i] = True

  return seqs_na


################################################################################
def batch_end(segments, bstart, batch_max):
  """ Determine the batch end that will keep the
          batch length under the given max. """

  bi = bstart
  blength = 0

  while bi < len(segments) and blength < batch_max:
    chrom, seg_start, seg_end = segments[bi]
    blength += seg_end - seg_start
    bi += 1

  bend = bi
  if bstart >= bend or bend > len(segments):
    print("I've made a terrible mistake. On batching segments", file=sys.stderr)
    exit(1)

  return bend


################################################################################
def limit_segments(segments, filter_bed):
  """ Limit to segments overlapping the given BED.

    Args
     segments: list of (chrom,start,end) genomic segments
     filter_bed: BED file to filter by

    Returns:
     fsegments: list of (chrom,start,end) genomic segments
    """

  # print segments to BED
  seg_fd, seg_bed_file = tempfile.mkstemp()
  seg_bed_out = open(seg_bed_file, 'w')
  for chrom, seg_start, seg_end in segments:
    print('%s\t%d\t%d' % (chrom, seg_start, seg_end), file=seg_bed_out)
  seg_bed_out.close()

  # intersect w/ filter_bed
  fsegments = []
  p = subprocess.Popen(
      'bedtools intersect -u -a %s -b %s' % (seg_bed_file, filter_bed),
      shell=True,
      stdout=subprocess.PIPE)
  for line in p.stdout:
    a = line.decode('utf-8').split()
    chrom = a[0]
    seg_start = int(a[1])
    seg_end = int(a[2])
    fsegments.append((chrom, seg_start, seg_end))

  p.communicate()

  os.close(seg_fd)
  os.remove(seg_bed_file)

  return fsegments


################################################################################
def segments_1hot(fasta_file, segments, seq_length, stride):
  """ Read and 1-hot code sequences in their segment batches.

    Args
     fasta_file: FASTA genome
     segments: list of (chrom,start,end) genomic segments to read
     seq_length: sequence length to break them into
     stride: distance to advance each sequence

    Returns:
     seqs_1hot: You know.
     seqs_segments: list of (chrom,start,end) sequence segments
    """

  # open fasta
  fasta = pysam.Fastafile(fasta_file)

  # initialize 1-hot coding list
  seqs_1hot = []

  # segment corresponding to each sequence
  seqs_segments = []

  for chrom, seg_start, seg_end in segments:
    # read sequence
    seg_seq = fasta.fetch(chrom, seg_start, seg_end)

    # break up into batchable sequences (as above in bigwig_batch)
    bstart = 0
    bend = bstart + seq_length
    while bend < len(seg_seq):
      # append
      seqs_1hot.append(basenji.dna_io.dna_1hot(seg_seq[bstart:bend]))

      seqs_segments.append((chrom, seg_start + bstart, seg_start + bend))

      # update
      bstart += stride
      bend += stride

  return np.array(seqs_1hot), seqs_segments


################################################################################
if __name__ == '__main__':
  main()
