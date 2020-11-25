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
import json
import os
import random
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from basenji import genome
from basenji import util
from basenji_data import annotate_unmap, break_large_contigs
from basenji_data import contig_sequences, limit_contigs
from basenji_data import divide_contigs_chr, divide_contigs_pct
from basenji_data import rejoin_large_contigs, write_seqs_bed
from basenji_data import Contig, ModelSeq

try:
  import slurm
except ModuleNotFoundError:
  pass

'''
akita_data.py

Compute model sequences from the genome, extracting DNA Hi-C/uC values.
'''

################################################################################
def main():
  usage = 'usage: %prog [options] <fasta_file> <targets_file>'
  parser = OptionParser(usage)
  parser.add_option('-b', dest='blacklist_bed',
      help='Set blacklist nucleotides to a baseline value.')
  parser.add_option('--break', dest='break_t',
      default=8388608, type='int',
      help='Break in half contigs above length [Default: %default]')
  parser.add_option('-c', '--crop', dest='crop_bp',
      default=0, type='int',
      help='Crop bp off each end [Default: %default]')
  parser.add_option('-d', dest='diagonal_offset',
      default=2, type='int',
      help='Positions on the diagonal to ignore [Default: %default]')
  parser.add_option('-f', dest='folds',
      default=None, type='int',
      help='Generate cross fold split [Default: %default]')
  parser.add_option('-g', dest='gaps_file',
      help='Genome assembly gaps BED [Default: %default]')
  parser.add_option('-k', dest='kernel_stddev',
      default=0, type='int',
      help='Gaussian kernel stddev to smooth values [Default: %default]')
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
      default=128, type='int',
      help='Sequences per TFRecord file [Default: %default]')
  parser.add_option('--restart', dest='restart',
      default=False, action='store_true',
      help='Continue progress from midpoint. [Default: %default]')
  parser.add_option('--sample', dest='sample_pct',
      default=1.0, type='float',
      help='Down-sample the segments')
  parser.add_option('--seed', dest='seed',
      default=44, type='int',
      help='Random seed [Default: %default]')
  parser.add_option('--stride_train', dest='stride_train',
      default=1., type='float',
      help='Stride to advance train sequences [Default: seq_length]')
  parser.add_option('--stride_test', dest='stride_test',
      default=1., type='float',
      help='Stride to advance valid and test sequences [Default: seq_length]')
  parser.add_option('--st', '--split_test', dest='split_test',
      default=False, action='store_true',
      help='Exit after split. [Default: %default]')
  parser.add_option('-t', dest='test_pct_or_chr',
      default=0.05, type='str',
      help='Proportion of the data for testing [Default: %default]')
  parser.add_option('-u', dest='umap_bed',
      help='Unmappable regions in BED format')
  parser.add_option('--umap_midpoints', dest='umap_midpoints',
      help='Regions with midpoints to exclude in BED format. Used for 4C/HiC.')
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
  parser.add_option('--as_obsexp', dest='as_obsexp',
      action="store_true", default=False,
      help='save targets as obsexp profiles')
  parser.add_option('--global_obsexp', dest='global_obsexp',
      action="store_true", default=False,
      help='use pre-calculated by-chromosome obs/exp')
  parser.add_option('--no_log', dest='no_log',
      action="store_true", default=False,
      help='do not take log for obs/exp')

  (options, args) = parser.parse_args()

  if len(args) != 2:
    parser.error('Must provide FASTA and sample coverage labels and paths.')
  else:
    fasta_file = args[0]
    targets_file = args[1]

  random.seed(options.seed)
  np.random.seed(options.seed)

  if options.break_t is not None and options.break_t < options.seq_length:
    print('Maximum contig length --break cannot be less than sequence length.', file=sys.stderr)
    exit(1)

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
    if np.mod(options.seq_length, options.snap) != 0: 
      raise ValueError('seq_length must be a multiple of snap')
    if np.mod(options.stride_train, options.snap) != 0: 
      raise ValueError('stride_train must be a multiple of snap')
    if np.mod(options.stride_test, options.snap) != 0:
      raise ValueError('stride_test must be a multiple of snap')

  if os.path.isdir(options.out_dir) and not options.restart:
    print('Remove output directory %s or use --restart option.' % options.out_dir)
    exit(1)
  elif not os.path.isdir(options.out_dir):
    os.mkdir(options.out_dir)

  # dump options
  with open('%s/options.json' % options.out_dir, 'w') as options_json_out:
    json.dump(options.__dict__, options_json_out, sort_keys=True, indent=4)

  ################################################################
  # define genomic contigs
  ################################################################
  if not options.restart:
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
  # label folds
  if options.folds is not None:
    fold_labels = ['fold%d' % fi for fi in range(options.folds)]
    num_folds = options.folds
  else:
    fold_labels = ['train', 'valid', 'test']
    num_folds = 3

  if not options.restart:
    if options.folds is not None:
      # divide by fold pct
      fold_contigs = divide_contigs_folds(contigs, options.folds)

    else:
      try:
        # convert to float pct
        valid_pct = float(options.valid_pct_or_chr)
        test_pct = float(options.test_pct_or_chr)
        assert(0 <= valid_pct <= 1)
        assert(0 <= test_pct <= 1)

        # divide by pct
        fold_contigs = divide_contigs_pct(contigs, test_pct, valid_pct)

      except (ValueError, AssertionError):
        # divide by chr
        valid_chrs = options.valid_pct_or_chr.split(',')
        test_chrs = options.test_pct_or_chr.split(',')
        fold_contigs = divide_contigs_chr(contigs, test_chrs, valid_chrs)

    # rejoin broken contigs within set
    for fi in range(len(fold_contigs)):
      fold_contigs[fi] = rejoin_large_contigs(fold_contigs[fi])

    # write labeled contigs to BED file
    ctg_bed_file = '%s/contigs.bed' % options.out_dir
    ctg_bed_out = open(ctg_bed_file, 'w')
    for fi in range(len(fold_contigs)):
      for ctg in fold_contigs[fi]:
        line = '%s\t%d\t%d\t%s' % (ctg.chr, ctg.start, ctg.end, fold_labels[fi])
        print(line, file=ctg_bed_out)
    ctg_bed_out.close()

  if options.split_test:
    exit()

  ################################################################
  # define model sequences
  ################################################################
  if not options.restart:
    fold_mseqs = []
    for fi in range(num_folds):
      if fold_labels[fi] in ['valid','test']:
        stride_fold = options.stride_test
      else:
        stride_fold = options.stride_train

      # stride sequences across contig
      fold_mseqs_fi = contig_sequences(fold_contigs[fi], options.seq_length,
                                       stride_fold, options.snap, fold_labels[fi])
      fold_mseqs.append(fold_mseqs_fi)

      # shuffle
      random.shuffle(fold_mseqs[fi])

      # down-sample
      if options.sample_pct < 1.0:
        fold_mseqs[fi] = random.sample(fold_mseqs[fi], int(options.sample_pct*len(fold_mseqs[fi])))

    # merge into one list
    mseqs = [ms for fm in fold_mseqs for ms in fm]


  ################################################################
  # mappability
  ################################################################
  if not options.restart:
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
      # annotate unmappable midpoints for 4C/HiC
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
    print('writing sequences to BED')
    seqs_bed_file = '%s/sequences.bed' % options.out_dir
    write_seqs_bed(seqs_bed_file, mseqs, True)
  else:
    # read from directory
    seqs_bed_file = '%s/sequences.bed' % options.out_dir
    unmap_npy = '%s/mseqs_unmap.npy' % options.out_dir
    mseqs = []
    fold_mseqs = []
    for fi in range(num_folds):
      fold_mseqs.append([])
    for line in open(seqs_bed_file):
      a = line.split()
      msg = ModelSeq(a[0], int(a[1]), int(a[2]), a[3])
      mseqs.append(msg)
      if a[3] == 'train':
        fi = 0
      elif a[3] == 'valid':
        fi = 1
      elif a[3] == 'test':
        fi = 2
      else:
        fi = int(a[3].replace('fold',''))
      fold_mseqs[fi].append(msg)


  ################################################################
  # read sequence coverage values
  ################################################################
  # read target datasets
  targets_df = pd.read_csv(targets_file, index_col=0, sep='\t')

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

    # scale_ti = 1
    # if 'scale' in targets_df.columns:
    #   scale_ti = targets_df['scale'].iloc[ti]

    if options.restart and os.path.isfile(seqs_cov_file):
      print('Skipping existing %s' % seqs_cov_file, file=sys.stderr)
    else:
      cmd = 'akita_data_read.py'
      cmd += ' --crop %d' % options.crop_bp
      cmd += ' -d %s' % options.diagonal_offset
      cmd += ' -k %d' % options.kernel_stddev
      cmd += ' -w %d' % options.pool_width
      if clip_ti is not None:
        cmd += ' --clip %f' % clip_ti
      # cmd += ' -s %f' % scale_ti
      if options.blacklist_bed:
        cmd += ' -b %s' % options.blacklist_bed
      if options.as_obsexp:
        cmd += ' --as_obsexp'
        if options.global_obsexp:
          cmd += ' --global_obsexp'
        if options.no_log:
          cmd += ' --no_log'
      cmd += ' %s' % genome_cov_file
      cmd += ' %s' % seqs_bed_file
      cmd += ' %s' % seqs_cov_file

      if options.run_local:
        # breaks on some OS
        # cmd += ' &> %s.err' % seqs_cov_stem
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

  for fold_set in fold_labels:
    fold_set_indexes = [i for i in range(len(mseqs)) if mseqs[i].label == fold_set]
    fold_set_start = fold_set_indexes[0]
    fold_set_end = fold_set_indexes[-1] + 1

    tfr_i = 0
    tfr_start = fold_set_start
    tfr_end = min(tfr_start+options.seqs_per_tfr, fold_set_end)

    while tfr_start <= fold_set_end:
      tfr_stem = '%s/%s-%d' % (tfr_dir, fold_set, tfr_i)

      cmd = 'basenji_data_write.py'
      cmd += ' -s %d' % tfr_start
      cmd += ' -e %d' % tfr_end

      # do not use      
      # if options.umap_bed is not None:
      #   cmd += ' -u %s' % unmap_npy
      # if options.umap_set is not None:
      #   cmd += ' --umap_set %f' % options.umap_set

      cmd += ' %s' % fasta_file
      cmd += ' %s' % seqs_bed_file
      cmd += ' %s' % seqs_cov_dir
      cmd += ' %s.tfr' % tfr_stem

      if options.run_local:
        # breaks on some OS
        # cmd += ' &> %s.err' % tfr_stem
        write_jobs.append(cmd)
      else:
        j = slurm.Job(cmd,
              name='write_%s-%d' % (fold_set, tfr_i),
              out_file='%s.out' % tfr_stem,
              err_file='%s.err' % tfr_stem,
              queue='standard', mem=15000, time='12:0:0')
        write_jobs.append(j)

      # update
      tfr_i += 1
      tfr_start += options.seqs_per_tfr
      tfr_end = min(tfr_start+options.seqs_per_tfr, fold_set_end)

  if options.run_local:
    util.exec_par(write_jobs, options.processes, verbose=True)
  else:
    slurm.multi_run(write_jobs, options.processes, verbose=True,
                    launch_sleep=1, update_sleep=5)

  ################################################################
  # stats
  ################################################################
  stats_dict = {}
  stats_dict['num_targets'] = targets_df.shape[0]
  stats_dict['seq_length'] = options.seq_length
  stats_dict['pool_width'] = options.pool_width
  stats_dict['crop_bp'] = options.crop_bp
  stats_dict['diagonal_offset'] = options.diagonal_offset

  target1_length = options.seq_length - 2*options.crop_bp
  target1_length = target1_length // options.pool_width
  target1_length = target1_length - options.diagonal_offset
  target_length = target1_length*(target1_length+1) // 2
  stats_dict['target_length'] = target_length

  for fi in range(num_folds):
    stats_dict['%s_seqs' % fold_labels[fi]] = len(fold_mseqs[fi])

  with open('%s/statistics.json' % options.out_dir, 'w') as stats_json_out:
    json.dump(stats_dict, stats_json_out, indent=4)


################################################################################
if __name__ == '__main__':
  main()