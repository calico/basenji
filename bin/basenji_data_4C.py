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
import collections
import heapq
import math
import pdb
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time

import h5py
import numpy as np
import pandas as pd

from basenji import genome
from basenji import util

try:
  import slurm
except ModuleNotFoundError:
  pass

'''
basenji_data.py

Compute model sequences from the genome, extracting DNA coverage values.
'''

################################################################################
def main():
  usage = 'usage: %prog [options] <fasta_file> <targets_file>'
  parser = OptionParser(usage)
  parser.add_option('-b', dest='blacklist_bed',
      help='Set blacklist nucleotides to a baseline value.')
  parser.add_option('--break', dest='break_t',
      default=786432, type='int',
      help='Break in half contigs above length [Default: %default]')
  # parser.add_option('-c', dest='clip',
  #     default=None, type='float',
  #     help='Clip target values to have minimum [Default: %default]')
  parser.add_option('-d', dest='sample_pct',
      default=1.0, type='float',
      help='Down-sample the segments')
  parser.add_option('-g', dest='gaps_file',
      help='Genome assembly gaps BED [Default: %default]')
  parser.add_option('-l', dest='seq_length',
      default=131072, type='int',
      help='Sequence length [Default: %default]')
  parser.add_option('--limit', dest='limit_bed',
      help='Limit to segments that overlap regions in a BED file')
  parser.add_option('--local', dest='run_local',
      default=False, action='store_true',
      help='Run jobs locally as opposed to on SLURM [Default: %default]')
  parser.add_option('-o', dest='out_dir',
      default='data_out',
      help='Output directory [Default: %default]')
  parser.add_option('-p', dest='processes',
      default=None, type='int',
      help='Number parallel processes [Default: %default]')
  parser.add_option('-r', dest='seqs_per_tfr',
      default=256, type='int',
      help='Sequences per TFRecord file [Default: %default]')
  parser.add_option('--restart', dest='restart',
      default=False, action='store_true',
      help='Skip already read HDF5 coverage values. [Default: %default]')
  parser.add_option('--seed', dest='seed',
      default=44, type='int',
      help='Random seed [Default: %default]')
  parser.add_option('--stride_train', dest='stride_train',
      default=1., type='float',
      help='Stride to advance train sequences [Default: seq_length]')
  parser.add_option('--stride_test', dest='stride_test',
      default=1., type='float',
      help='Stride to advance valid and test sequences [Default: seq_length]')
  parser.add_option('--soft', dest='soft_clip',
      default=False, action='store_true',
      help='Soft clip values, applying sqrt to the execess above the threshold [Default: %default]')
  parser.add_option('-t', dest='test_pct_or_chr',
      default=0.05, type='str',
      help='Proportion of the data for testing [Default: %default]')
  parser.add_option('-u', dest='umap_bed',
      help='Unmappable regions in BED format')
  parser.add_option('--umap_midpoints', dest='umap_midpoints',
      help='Regions with midpoints to exclude in BED format. Used for 4C.')
  parser.add_option('--umap_t', dest='umap_t',
      default=0.3, type='float',
      help='Remove sequences with more than this unmappable bin % [Default: %default]')
  parser.add_option('--umap_set', dest='umap_set',
      default=None, type='float',
      help='Set unmappable regions to this percentile in the sequences\' distribution of values')
  parser.add_option('-w', dest='pool_width',
      default=128, type='int',
      help='Sum pool width [Default: %default]')
  parser.add_option('-v', dest='valid_pct_or_chr',
      default=0.05, type='str',
      help='Proportion of the data for validation [Default: %default]')
  parser.add_option('--snap', dest='snap',
      default=None, type='int',
      help='snap stride to multiple for binned targets in bp, if not None seq_length must be a multiple of snap')
  (options, args) = parser.parse_args()

  if len(args) != 2:
    parser.error('Must provide FASTA and sample coverage labels and paths.')
  else:
    fasta_file = args[0]
    targets_file = args[1]

  random.seed(options.seed)
  np.random.seed(options.seed)

  # transform proportion strides to base pairs
  if options.stride_train <= 1:
    print('stride_train %.f'%options.stride_train, end='')
    options.stride_train = options.stride_train*options.seq_length
    print(' converted to %f' % options.stride_train)
  options.stride_train = int(np.round(options.stride_train))
  if options.stride_test <= 1:
    print('stride_test %.f'%options.stride_test, end='')
    options.stride_test = options.stride_test*options.seq_length
    print(' converted to %f' % options.stride_test)
  options.stride_test = int(np.round(options.stride_test))

  if options.snap != None:
    if np.mod(options.seq_length, options.snap) !=0: 
      raise ValueError('seq_length must be a multiple of snap')
    if np.mod(options.stride_test, options.snap) !=0 or  np.mod(options.stride_train, options.snap) !=0: 
      raise ValueError('stride lengths must be a multiple of snap')

  if not os.path.isdir(options.out_dir):
    os.mkdir(options.out_dir)

  ################################################################
  # define genomic contigs
  ################################################################
  chrom_contigs = genome.load_chromosomes(fasta_file)

  # remove gaps
  if options.gaps_file:
    chrom_contigs = genome.split_contigs(chrom_contigs,
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

  # break up large contigs
  if options.break_t is not None:
    contigs = break_large_contigs(contigs, options.break_t)

  # print contigs to BED file
  ctg_bed_file = '%s/contigs.bed' % options.out_dir
  write_seqs_bed(ctg_bed_file, contigs)


  ################################################################
  # divide between train/valid/test
  ################################################################
  try:
    # convert to float pct
    valid_pct = float(options.valid_pct_or_chr)
    test_pct = float(options.test_pct_or_chr)
    assert(0 <= valid_pct <= 1)
    assert(0 <= test_pct <= 1)

    # divide by pct
    contig_sets = divide_contigs_pct(contigs, test_pct, valid_pct)

  except (ValueError, AssertionError):
    # divide by chr
    valid_chr = options.valid_pct_or_chr
    test_chr = options.test_pct_or_chr
    contig_sets = divide_contigs_chr(contigs, test_chr, valid_chr)

  train_contigs, valid_contigs, test_contigs = contig_sets

  # rejoin broken contigs within set
  train_contigs = rejoin_large_contigs(train_contigs)
  valid_contigs = rejoin_large_contigs(valid_contigs)
  test_contigs = rejoin_large_contigs(test_contigs)

  ################################################################
  # define model sequences
  ################################################################
  # stride sequences across contig
  train_mseqs = contig_sequences(train_contigs, options.seq_length, options.stride_train, options.snap, label='train')
  valid_mseqs = contig_sequences(valid_contigs, options.seq_length, options.stride_test,  options.snap, label='valid')
  test_mseqs  = contig_sequences(test_contigs,  options.seq_length, options.stride_test,  options.snap, label='test')

  # shuffle
  random.shuffle(train_mseqs)
  random.shuffle(valid_mseqs)
  random.shuffle(test_mseqs)

  # down-sample
  if options.sample_pct < 1.0:
    train_mseqs = random.sample(train_mseqs, int(options.sample_pct*len(train_mseqs)))
    valid_mseqs = random.sample(valid_mseqs, int(options.sample_pct*len(valid_mseqs)))
    test_mseqs = random.sample(test_mseqs, int(options.sample_pct*len(test_mseqs)))

  # merge
  mseqs = train_mseqs + valid_mseqs + test_mseqs


  ################################################################
  # mappability
  ################################################################
  if (options.umap_bed is not None) or (options.umap_midpoints is not None):
    if shutil.which('bedtools') is None:
      print('Install Bedtools to annotate unmappable sites', file=sys.stderr)
      exit(1)

  if options.umap_bed is not None:
    # annotate unmappable positions
    mseqs_unmap = annotate_unmap(mseqs, options.umap_bed,
                                 options.seq_length, options.pool_width)

    # filter unmappable
    mseqs_map_mask = (mseqs_unmap.mean(axis=1, dtype='float64') < options.umap_t)
    mseqs = [mseqs[i] for i in range(len(mseqs)) if mseqs_map_mask[i]]
    mseqs_unmap = mseqs_unmap[mseqs_map_mask,:]

    # write to file
    unmap_npy = '%s/mseqs_unmap.npy' % options.out_dir
    np.save(unmap_npy, mseqs_unmap)

  if options.umap_midpoints is not None:
    # annotate unmappable midpoints for 4C
    mseqs_unmap = annotate_unmap(mseqs, options.umap_midpoints,
                                 options.seq_length, options.pool_width)

    # filter unmappable
    seqmid =  mseqs_unmap.shape[1]//2  #int( options.seq_length / options.pool_width /2)
    mseqs_map_mask = (np.sum(mseqs_unmap[:,seqmid-1:seqmid+1],axis=1) == 0)

    mseqs = [mseqs[i] for i in range(len(mseqs)) if mseqs_map_mask[i]]
    mseqs_unmap = mseqs_unmap[mseqs_map_mask,:]

    # write to file
    unmap_npy = '%s/mseqs_unmap_midpoints.npy' % options.out_dir
    np.save(unmap_npy, mseqs_unmap)

  # write sequences to BED
  seqs_bed_file = '%s/sequences.bed' % options.out_dir
  write_seqs_bed(seqs_bed_file, mseqs, True)


  ################################################################
  # read sequence coverage values
  ################################################################
  # read target datasets
  targets_df = pd.read_table(targets_file, index_col=0)

  seqs_cov_dir = '%s/seqs_cov' % options.out_dir
  if not os.path.isdir(seqs_cov_dir):
    os.mkdir(seqs_cov_dir)

  read_jobs = []

  for ti in range(targets_df.shape[0]):
    genome_cov_file = targets_df['file'].iloc[ti]
    seqs_cov_stem = '%s/%d' % (seqs_cov_dir, ti)
    seqs_cov_file = '%s.h5' % seqs_cov_stem

    clip_ti = None
    if 'clip' in targets_df.columns:
      clip_ti = targets_df['clip'].iloc[ti]

    scale_ti = 1
    if 'scale' in targets_df.columns:
      scale_ti = targets_df['scale'].iloc[ti]

    if options.restart and os.path.isfile(seqs_cov_file):
      print('Skipping existing %s' % seqs_cov_file, file=sys.stderr)
    else:
      cmd = 'basenji_data_4C_read.py'
      cmd += ' -w %d' % options.pool_width
      cmd += ' -u %s' % targets_df['sum_stat'].iloc[ti]
      if clip_ti is not None:
        cmd += ' -c %f' % clip_ti
      if options.soft_clip:
        cmd += ' --soft'
      cmd += ' -s %f' % scale_ti
      if options.blacklist_bed:
        cmd += ' -b %s' % options.blacklist_bed
      cmd += ' %s' % genome_cov_file
      cmd += ' %s' % seqs_bed_file
      cmd += ' %s' % seqs_cov_file

      if options.run_local:
        #cmd += ' &> %s.err' % seqs_cov_stem ##comment this out to work in ubuntu
        read_jobs.append(cmd)
      else:
        j = slurm.Job(cmd,
            name='read_t%d' % ti,
            out_file='%s.out' % seqs_cov_stem,
            err_file='%s.err' % seqs_cov_stem,
            queue='standard', mem=15000, time='12:0:0')
        read_jobs.append(j)

  if options.run_local:
    util.exec_par(read_jobs, options.processes, verbose=True)
  else:
    slurm.multi_run(read_jobs, options.processes, verbose=True,
                    launch_sleep=1, update_sleep=5)

  ################################################################
  # write TF Records
  ################################################################
  # copy targets file
  shutil.copy(targets_file, '%s/targets.txt' % options.out_dir)

  # initialize TF Records dir
  tfr_dir = '%s/tfrecords' % options.out_dir
  if not os.path.isdir(tfr_dir):
    os.mkdir(tfr_dir)

  write_jobs = []

  for tvt_set in ['train', 'valid', 'test']:
    tvt_set_indexes = [i for i in range(len(mseqs)) if mseqs[i].label == tvt_set]
    tvt_set_start = tvt_set_indexes[0]
    tvt_set_end = tvt_set_indexes[-1] + 1

    tfr_i = 0
    tfr_start = tvt_set_start
    tfr_end = min(tfr_start+options.seqs_per_tfr, tvt_set_end)

    while tfr_start <= tvt_set_end:
      tfr_stem = '%s/%s-%d' % (tfr_dir, tvt_set, tfr_i)

      cmd = 'basenji_data_write.py'
      cmd += ' -s %d' % tfr_start
      cmd += ' -e %d' % tfr_end
      if options.umap_bed is not None:
        cmd += ' -u %s' % unmap_npy
      if options.umap_set is not None:
        cmd += ' --umap_set %f' % options.umap_set

      cmd += ' %s' % fasta_file
      cmd += ' %s' % seqs_bed_file
      cmd += ' %s' % seqs_cov_dir
      cmd += ' %s.tfr' % tfr_stem

      if options.run_local:
        cmd += ' &> %s.err' % tfr_stem
        write_jobs.append(cmd)
      else:
        j = slurm.Job(cmd,
              name='write_%s-%d' % (tvt_set, tfr_i),
              out_file='%s.out' % tfr_stem,
              err_file='%s.err' % tfr_stem,
              queue='standard', mem=15000, time='12:0:0')
        write_jobs.append(j)

      # update
      tfr_i += 1
      tfr_start += options.seqs_per_tfr
      tfr_end = min(tfr_start+options.seqs_per_tfr, tvt_set_end)

  if options.run_local:
    util.exec_par(write_jobs, options.processes, verbose=True)
  else:
    slurm.multi_run(write_jobs, options.processes, verbose=True,
                    launch_sleep=1, update_sleep=5)


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
  seqs_bed_file = seqs_temp.name
  write_seqs_bed(seqs_bed_file, mseqs)

  # hash segments to indexes
  chr_start_indexes = {}
  for i in range(len(mseqs)):
    chr_start_indexes[(mseqs[i].chr, mseqs[i].start)] = i

  # initialize unmappable array
  pool_seq_length = seq_length // pool_width
  seqs_unmap = np.zeros((len(mseqs), pool_seq_length), dtype='bool')

  # intersect with unmappable regions
  p = subprocess.Popen(
      'bedtools intersect -wo -a %s -b %s' % (seqs_bed_file, unmap_bed),
      shell=True, stdout=subprocess.PIPE)
  for line in p.stdout:
    line = line.decode('utf-8')
    a = line.split()

    seq_chrom = a[0]
    seq_start = int(a[1])
    seq_end = int(a[2])
    seq_key = (seq_chrom, seq_start)

    unmap_start = int(a[4])
    unmap_end = int(a[5])

    overlap_start = max(seq_start, unmap_start)
    overlap_end = min(seq_end, unmap_end)

    pool_seq_unmap_start = math.floor((overlap_start - seq_start) / pool_width)
    pool_seq_unmap_end = math.ceil((overlap_end - seq_start) / pool_width)

    # skip minor overlaps to the first
    first_start = seq_start + pool_seq_unmap_start * pool_width
    first_end = first_start + pool_width
    first_overlap = first_end - overlap_start
    if first_overlap < 0.2 * pool_width:
      pool_seq_unmap_start += 1

    # skip minor overlaps to the last
    last_start = seq_start + (pool_seq_unmap_end - 1) * pool_width
    last_overlap = overlap_end - last_start
    if last_overlap < 0.2 * pool_width:
      pool_seq_unmap_end -= 1

    seqs_unmap[chr_start_indexes[seq_key], pool_seq_unmap_start:pool_seq_unmap_end] = True
    assert(seqs_unmap[chr_start_indexes[seq_key], pool_seq_unmap_start:pool_seq_unmap_end].sum() == pool_seq_unmap_end-pool_seq_unmap_start)

  return seqs_unmap

################################################################################
def break_large_contigs(contigs, break_t, verbose=False):
  """Break large contigs in half until all contigs are under
     the size threshold."""

  # initialize a heapq of contigs and lengths
  contig_heapq = []
  for ctg in contigs:
    ctg_len = ctg.end - ctg.start
    heapq.heappush(contig_heapq, (-ctg_len, ctg))

  ctg_len = break_t + 1
  while ctg_len > break_t:

    # pop largest contig
    ctg_nlen, ctg = heapq.heappop(contig_heapq)
    ctg_len = -ctg_nlen

    # if too large
    if ctg_len > break_t:
      if verbose:
        print('Breaking %s:%d-%d (%d nt)' % (ctg.chr,ctg.start,ctg.end,ctg_len))

      # break in two
      ctg_mid = ctg.start + ctg_len//2

      try:
        ctg_left = Contig(ctg.genome, ctg.chr, ctg.start, ctg_mid)
        ctg_right = Contig(ctg.genome, ctg.chr, ctg_mid, ctg.end)
      except AttributeError:
        ctg_left = Contig(ctg.chr, ctg.start, ctg_mid)
        ctg_right = Contig(ctg.chr, ctg_mid, ctg.end)

      # add left
      ctg_left_len = ctg_left.end - ctg_left.start
      heapq.heappush(contig_heapq, (-ctg_left_len, ctg_left))

      # add right
      ctg_right_len = ctg_right.end - ctg_right.start
      heapq.heappush(contig_heapq, (-ctg_right_len, ctg_right))

  # return to list
  contigs = [len_ctg[1] for len_ctg in contig_heapq]

  return contigs


################################################################################
def contig_sequences(contigs, seq_length, stride, snap, label=None):
  ''' Break up a list of Contig's into a list of ModelSeq's. '''
  mseqs = []
  for ctg in contigs:
    if snap==None:
      seq_start = ctg.start
      seq_end = seq_start + seq_length
    else:
      seq_start =  int( np.ceil(ctg.start/snap)*snap)
      seq_end   =  int( ((seq_start + seq_length)//snap) *snap)
    while seq_end < ctg.end:
      # record sequence
      mseqs.append(ModelSeq(ctg.chr, seq_start, seq_end, label))

      # update
      seq_start += stride
      seq_end += stride
      
  return mseqs


################################################################################
def divide_contigs_pct(contigs, test_pct, valid_pct, pct_abstain=0.2):
  """Divide list of contigs into train/valid/test lists,
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

  # initialize train/valid/test contig lists
  train_contigs = []
  valid_contigs = []
  test_contigs = []

  # process contigs
  for ctg_len, ctg in length_contigs:

    # compute gap between current and aim
    test_nt_gap = max(0, test_nt_aim - test_nt)
    valid_nt_gap = max(0, valid_nt_aim - valid_nt)
    train_nt_gap = max(1, train_nt_aim - train_nt)

    # skip if too large
    if ctg_len > pct_abstain*test_nt_gap:
      test_nt_gap = 0
    if ctg_len > pct_abstain*valid_nt_gap:
      valid_nt_gap = 0

    # compute remaining %
    gap_sum = train_nt_gap + valid_nt_gap + test_nt_gap
    test_pct_gap = test_nt_gap / gap_sum
    valid_pct_gap = valid_nt_gap / gap_sum
    train_pct_gap = train_nt_gap / gap_sum

    # sample train/valid/test
    ri = np.random.choice(range(3), 1, p=[train_pct_gap, valid_pct_gap, test_pct_gap])[0]
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

  print('Contigs divided into')
  print(' Train: %5d contigs, %10d nt (%.4f)' % \
      (len(train_contigs), train_nt, train_nt/total_nt))
  print(' Valid: %5d contigs, %10d nt (%.4f)' % \
      (len(valid_contigs), valid_nt, valid_nt/total_nt))
  print(' Test:  %5d contigs, %10d nt (%.4f)' % \
      (len(test_contigs), test_nt, test_nt/total_nt))

  return train_contigs, valid_contigs, test_contigs


################################################################################
def divide_contigs_chr(contigs, test_chr, valid_chr):
  """Divide list of contigs into train/valid/test lists
     by chromosome."""

  # initialize current train/valid/test nucleotides
  train_nt = 0
  valid_nt = 0
  test_nt = 0

  # initialize train/valid/test contig lists
  train_contigs = []
  valid_contigs = []
  test_contigs = []

  # process contigs
  for ctg in contigs:
    ctg_len = ctg.end - ctg.start

    if ctg.chr == test_chr:
      test_contigs.append(ctg)
      test_nt += ctg_len
    elif ctg.chr == valid_chr:
      valid_contigs.append(ctg)
      valid_nt += ctg_len
    else:
      train_contigs.append(ctg)
      train_nt += ctg_len

  total_nt = train_nt + valid_nt + test_nt

  print('Contigs divided into')
  print(' Train: %5d contigs, %10d nt (%.4f)' % \
      (len(train_contigs), train_nt, train_nt/total_nt))
  print(' Valid: %5d contigs, %10d nt (%.4f)' % \
      (len(valid_contigs), valid_nt, valid_nt/total_nt))
  print(' Test:  %5d contigs, %10d nt (%.4f)' % \
      (len(test_contigs), test_nt, test_nt/total_nt))

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
def rejoin_large_contigs(contigs):
  """ Rejoin large contigs that were broken up before alignment comparison."""

  # split list by chromosome
  chr_contigs = {}
  for ctg in contigs:
    chr_contigs.setdefault(ctg.chr,[]).append(ctg)

  contigs = []
  for chrm in chr_contigs:
    # sort within chromosome
    chr_contigs[chrm].sort(key=lambda x: x.start)

    ctg_ongoing = chr_contigs[chrm][0]
    for i in range(1, len(chr_contigs[chrm])):
      ctg_this = chr_contigs[chrm][i]
      if ctg_ongoing.end == ctg_this.start:
        # join
        # ctg_ongoing.end = ctg_this.end
        ctg_ongoing = ctg_ongoing._replace(end=ctg_this.end)
      else:
        # conclude ongoing
        contigs.append(ctg_ongoing)

        # move to next
        ctg_ongoing = ctg_this

    # conclude final
    contigs.append(ctg_ongoing)

  return contigs


################################################################################
def write_seqs_bed(bed_file, seqs, labels=False):
  '''Write sequences to BED file.'''
  bed_out = open(bed_file, 'w')
  for i in range(len(seqs)):
    line = '%s\t%d\t%d' % (seqs[i].chr, seqs[i].start, seqs[i].end)
    if labels:
      line += '\t%s' % seqs[i].label
    print(line, file=bed_out)
  bed_out.close()

################################################################################
Contig = collections.namedtuple('Contig', ['chr', 'start', 'end'])
ModelSeq = collections.namedtuple('ModelSeq', ['chr', 'start', 'end', 'label'])


################################################################################
if __name__ == '__main__':
  print('starting')
  main()
  print('')
  print('DONE')
  print('')
