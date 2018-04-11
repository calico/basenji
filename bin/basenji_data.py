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
  parser.add_option('-o', dest='out_dir',
      default='data_out',
      help='Output directory [Default: %default]')
  parser.add_option('-p', dest='processes',
      default=1, type='int',
      help='Number parallel processes [Default: %default]')
  parser.add_option('--stride_train', dest='stride_train',
      type='float', default=1.,
      help='Stride to advance train sequences [Default: seq_length]')
  parser.add_option('--stride_test', dest='stride_test',
      type='float', default=1.,
      help='Stride to advance valid and test sequences [Default: seq_length]')
  parser.add_option('-t', dest='test_pct_or_chr',
      type='str', default=0.05,
      help='Proportion of the data for testing [Default: %default]')
  parser.add_option('-u', dest='unmap_bed',
      help='Unmappable segments to set to NA')
  parser.add_option('--unmap_t', dest='unmap_t',
      default=0.3, type='float',
      help='Remove sequences with more than this unmappable bin % [Default: %default]')
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
    parser.error('Must provide FASTA and sample coverage labels and paths.')
  else:
    fasta_file = args[0]
    sample_wigs_file = args[1]

  random.seed(1)

  if not os.path.isdir(options.out_dir):
    os.mkdir(options.out_dir)

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
  # define genomic contigs
  ################################################################
  chrom_contigs = basenji.genome.load_chromosomes(fasta_file)

  # remove gaps
  if options.gaps_file:
    chrom_contigs = basenji.genome.split_contigs(chrom_contigs,
                                                 options.gaps_file)

  # ditch the chromosomes for contigs
  contigs = []
  for chrom in chrom_contigs:
    contigs += [Contig(chrom, ctg_start, ctg_end)
                 for ctg_start, ctg_end in chrom_contigs[chrom]]

  # limit to a BED file
  if options.limit_bed is not None:
    contigs = limit_contigs(contigs, options.limit_bed)

  # filter for large enough
  contigs = [ctg for ctg in contigs if ctg.end - ctg.start >= options.seq_length]

  # down-sample
  if options.sample_pct < 1.0:
    contigs = random.sample(contigs, int(options.sample_pct= * len(contigs)))

  # print contigs to BED file
  ctg_bed_file = '%s/contigs.bed' % options.out_dir
  write_seqs_bed(ctg_bed_file, contigs)


  ################################################################
  # divide between train/valid/test
  ################################################################
  contig_sets = divide_contigs(contigs, options.test_pct_or_chr, options.valid_pct_or_chr)
  train_contigs, valid_contigs, test_contigs = contig_sets

  ################################################################
  # define model sequences
  ################################################################

  # stride sequences across contig
  train_mseqs = contig_sequences(train_contigs, options.seq_length, options.stride_train)
  valid_mseqs = contig_sequences(valid_contigs, options.seq_length, options.stride_test)
  test_mseqs = contig_sequences(test_contigs, options.seq_length, options.stride_test)


  # TODO: shuffle?
  # TODO: merge back together with a list of labels?


  if options.unmap_bed is not None:
    # annotate unmappable positions
    train_unmap = annotate_unmap(train_mseqs, options.unmap_bed,
                                 options.seq_length, options.pool_width)
    valid_unmap = annotate_unmap(valid_mseqs, options.unmap_bed,
                                 options.seq_length, options.pool_width)
    test_unmap = annotate_unmap(test_mseqs, options.unmap_bed,
                                 options.seq_length, options.pool_width)

    # filter unmappable
    train_map_mask = (train_unmap.mean(axis=1, dtype='float64') < options.unmap_t)
    train_mseqs = [train_mseqs[i] for i in range(len(train_mseqs)) if train_map_mask[i]]
    train_unmap = train_unmap[train_map_mask,:]

    valid_map_mask = (valid_unmap.mean(axis=1, dtype='float64') < options.unmap_t)
    valid_mseqs = [valid_mseqs[i] for i in range(len(valid_mseqs)) if valid_map_mask[i]]
    valid_unmap = valid_unmap[valid_map_mask,:]

    test_map_mask = (test_unmap.mean(axis=1, dtype='float64') < options.unmap_t)
    test_mseqs = [test_mseqs[i] for i in range(len(test_mseqs)) if test_map_mask[i]]
    test_unmap = test_unmap[test_map_mask,:]

  # write sequences to BED
  mseqs_bed_file = '%s/sequences.bed' % options.out_dir
  mseq_labels = ['train']*len(train_mseqs) + ['valid']*len(valid_mseqs) + ['test']*len(test_mseqs)
  write_seqs_bed(mseqs_bed_file, train_mseqs+valid_mseqs+test_mseqs, mseq_labels)

  ################################################################
  # read sequence coverage values
  ################################################################

  seqs_cov_dir = '%s/seqs_cov' % options.out_dir
  if not os.path.isdir(seqs_cov_dir):
    os.mkdir(seqs_cov_dir)

  # generate numpy arrays on cluster
  jobs = []
  for target_label in target_wigs.keys():
    genome_cov_file = target_wigs[target_label]
    seqs_cov_file = '%s/%s.h5' % (seqs_cov_dir, target_label)
    if not os.path.isfile(seq_cov_file):
      cmd = 'echo $HOSTNAME; basenji_data_read.py %s %s %s' %
          (genome_cov_file, seqs_bed_file, seqs_cov_file)

      j = slurm.Job(cmd,
          name='cov_%s' % target_label,
          out_file='%s/%s.out' % (seqs_cov_dir, target_label),
          err_file='%s/%s.err' % (seqs_cov_dir, target_label),
          queue='standard,tbdisk', mem=15000, time='12:0:0')
      jobs.append(j)

  slurm.multi_run(jobs)

  ################################################################
  # write TF Records
  ################################################################








  # load into targets_real
  targets_real = []

  for target_label in target_wigs.keys():
    npy_file = '%s/%s.npy' % (options.out_dir, target_label)
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
def annotate_unmap(mseqs, unmap_bed, seq_length, pool_width):
  """ Intersect the sequence segments with unmappable regions
         and annoate the segments as NaN to possible be ignored.

    Args:
      mseqs: list of ModelSeq's
      unmap_bed: unmappable regions BED file
      seq_length: sequence length
      pool_width: pooled bin width

    Returns:
      seqs_unmap: NxL binary NA indicators
    """

  # print sequence segments to file
  seqs_temp = tempfile.NamedTemporaryFile()
  seqs_bed = seqs_temp.name
  seqs_out = open(seqs_bed, 'w')
  for mseq in mseqs:
    print('%s\t%d\t%d' % (mseq.chr, mseq.start, mseq.end), file=seqs_out)
  seqs_out.close()

  # hash segments to indexes
  chr_start_indexes = {}
  for i in range(len(mseqs)):
    chr_start_indexes[(mseqs[i].chr,mseqs[i].start)] = i

  # initialize unmappable array
  pool_seq_length = seq_length // pool_width
  seqs_unmap = np.zeros((len(mseqs), pool_seq_length), dtype='bool')

  # intersect with unmappable regions
  p = subprocess.Popen(
      'bedtools intersect -wo -a %s -b %s' % (seqs_bed, unmap_bed),
      shell=True,
      stdout=subprocess.PIPE)
  for line in p.stdout:
    line = line.decode('utf-8')
    a = line.split()

    seq_chrom = a[0]
    seq_start = int(a[1])
    seq_key = (seq_chrom, seq_start)

    unmap_start = int(a[4])
    unmap_end = int(a[5])

    pool_seq_unmap_start = math.floor((unmap_start - seq_start) / pool_width)
    pool_seq_unmap_end = math.ceil((unmap_end - seq_start) / pool_width)

    # skip minor overlaps to the first
    first_start = seq_start + pool_seq_unmap_start * pool_width
    first_end = first_start + pool_width
    first_overlap = first_end - unmap_start
    if first_overlap < 0.2 * pool_width:
      pool_seq_unmap_start += 1

    # skip minor overlaps to the last
    last_start = seq_start + (pool_seq_unmap_end - 1) * pool_width
    last_overlap = unmap_end - last_start
    if last_overlap < 0.2 * pool_width:
      pool_seq_unmap_end -= 1

    seqs_unmap[chr_start_indexes[seq_key], pool_seq_unmap_start:pool_seq_unmap_end] = True

  return seqs_unmap


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
def contig_sequences(contigs, seq_length, stride):
  ''' Break up a list of Contig's into a list of ModelSeq's. '''
  mseqs = []

  for chrom, ctg_start, ctg_end in contigs:
    seq_start = ctg_start
    seq_end = seq_start + seq_length

    while seq_end < ctg_end:
      # record sequence
      mseqs.append(ModelSeq(chrom, seq_start, seq_end))

      # update
      bstart += stride
      bend += stride

  return mseqs


################################################################################
def divide_contigs(contigs, test_pct, valid_pct, first_train=100):
  """Divide list of contigs intro train/valid/test lists,
     aiming for the specified nucleotide percentages."""

  # sort contigs descending by length
  length_contigs = [(ctg.end-ctg.start,ctg) for ctg in contigs]
  length_contigs.sort(reverse=True)

  # compute total nucleotides
  total_nt = sum([lc[0] for lc in length_contigs])

  # compute aimed train/valid/test nucleotides
  test_nt_aim = test_pct * total_nt
  valid_nt_aim = valid_pct * total_nt
  train_nt_aim = total_nt - valid_nt_aim - test_nt_aim

  # initialize current train/valid/test nucleotides
  train_nt = 0
  valid_nt = 0
  test_nt = 0

  # initialie train/valid/test contig lists
  train_contigs = []
  valid_contigs = []
  test_contigs = []

  # add the longest contigs to the training set
  ci = 0
  while ci < first_train and ci < len(length_contigs):
    ctg_len, ctg = length_contigs[ci]

    train_contigs.append(ctg)
    train_nt += ctg_len
    ci += 1

  # sample the remainder
  while ci < len(length_contigs):
    ctg_len, ctg = length_contigs[ci]

    # compute new train/valid/test sample %'s
    remaining_nt = total_nt - train_nt - valid_nt - test_nt
    test_pct_aim = max(0, (test_nt_aim - test_nt) / remaining_nt)
    valid_pct_aim = max(0, (valid_nt_aim - valid_nt) / remaining_nt)
    train_pct_aim = 1.0 - valid_pct_aim - test_pct_aim

    # sample train/valid/test
    ri = np.random.choice(range(3), 1, [train_pct_aim, valid_pct_aim, test_pct_aim])
    if ri == 0:
      train_contigs.append(ctg)
      train_nt += ctg_len
    elif ri == 1:
      valid_contigs.append(ctg)
      valid_nt += ctg_len
    elif ri == 2:
      test_contigs.append(ctg)
      test_nt += ctg_len
    else:
      print('TVT random number beyond 0,1,2', file=sys.stderr)
      exit(1)

    ci += 1

  print('Contigs divided into')
  print(' Train: %9d (%.4f)' % (train_nt, train_nt/total_nt))
  print(' Valid: %9d (%.4f)' % (valid_nt, valid_nt/valid_nt))
  print(' Test:  %9d (%.4f)' % (test_nt, test_nt/total_nt))

  return train_contigs, valid_contigs, test_contigs


################################################################################
def limit_contigs(contigs, filter_bed):
  """ Limit to contigs overlapping the given BED.

    Args
     contigs: list of Contigs
     filter_bed: BED file to filter by

    Returns:
     fcontigs: list of Contigs
    """

  # print ctgments to BED
  ctg_fd, ctg_bed_file = tempfile.mkstemp()
  ctg_bed_out = open(ctg_bed_file, 'w')
  for ctg in contigs:
    print('%s\t%d\t%d' % (ctg.chrom, ctg.start, ctg.end), file=ctg_bed_out)
  ctg_bed_out.close()

  # intersect w/ filter_bed
  fcontigs = []
  p = subprocess.Popen(
      'bedtools intersect -u -a %s -b %s' % (ctg_bed_file, filter_bed),
      shell=True,
      stdout=subprocess.PIPE)
  for line in p.stdout:
    a = line.decode('utf-8').split()
    chrom = a[0]
    ctg_start = int(a[1])
    ctg_end = int(a[2])
    fcontigs.append(Contig(chrom, ctg_start, ctg_end))

  p.communicate()

  os.close(ctg_fd)
  os.remove(ctg_bed_file)

  return fcontigs


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
def write_seqs_bed(bed_file, seqs, labels=None):
  '''Write sequences to BED file.'''
  bed_out = open(bed_file, 'w')
  for i in range(len(seqs)):
    line = '%s\t%d\t%d' % (seqs[i].chr, seqs[i].start, seqs[i].end)
    if labels is not None:
      line += '\t%s' % labels[i]
    print(line, file=bed_out)
  bed_out.close()

################################################################################
Contig = collections.namedtuple('chr', 'start', 'end')
ModelSeq = collections.namedtuple('chr', 'start', 'end')


################################################################################
if __name__ == '__main__':
  main()
