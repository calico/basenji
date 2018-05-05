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
import subprocess
import sys
import tempfile
import time

import h5py
import networkx as nx
import numpy as np
import pandas as pd
import pybedtools

import util
import slurm

import basenji.genome as genome

'''
basenji_data.py

Compute model sequences from the genome, extracting DNA coverage values.
'''

################################################################################
def main():
  usage = 'usage: %prog [options] <fasta1_file> <targets1_file> <fasta2_file> <targets2_file>'
  parser = OptionParser(usage)
  parser.add_option('-a', dest='align_net', help='Alignment .net file')
  parser.add_option('-b', dest='break_t',
      default=None, type='int',
      help='Break in half contigs above length [Default: %default]')
  # parser.add_option('-c', dest='clip',
  #     default=None, type='float',
  #     help='Clip target values to have minimum [Default: %default]')
  parser.add_option('-d', dest='sample_pct',
      default=1.0, type='float',
      help='Down-sample the segments')
  parser.add_option('-f', dest='fill_min',
    default=3000000, type='int',
    help='Alignment net fill size minimum [Default: %default]')
  parser.add_option('--g1', dest='gaps1_file',
      help='Genome1 assembly gaps BED [Default: %default]')
  parser.add_option('--g2', dest='gaps2_file',
      help='Genome2 assembly gaps BED [Default: %default]')
  parser.add_option('-l', dest='seq_length',
      default=131072, type='int',
      help='Sequence length [Default: %default]')
  parser.add_option('--local', dest='run_local',
      default=False, action='store_true',
      help='Run jobs locally as opposed to on SLURM [Default: %default]')
  parser.add_option('-o', dest='out_dir',
      default='data_out',
      help='Output directory [Default: %default]')
  parser.add_option('-p', dest='processes',
      default=None, type='int',
      help='Number parallel processes [Default: %default]')
  parser.add_option('--seed', dest='seed',
      default=44, type='int',
      help='Random seed [Default: %default]')
  parser.add_option('--stride_train', dest='stride_train',
      default=1., type='float',
      help='Stride to advance train sequences [Default: seq_length]')
  parser.add_option('--stride_test', dest='stride_test',
      default=1., type='float',
      help='Stride to advance valid and test sequences [Default: seq_length]')
  parser.add_option('-r', dest='seqs_per_tfr',
      default=256, type='int',
      help='Sequences per TFRecord file [Default: %default]')
  parser.add_option('-t', dest='test_pct',
      default=0.05, type='float',
      help='Proportion of the data for testing [Default: %default]')
  parser.add_option('-u', dest='unmap_bed',
      help='Unmappable segments to set to NA')
  parser.add_option('--unmap_t', dest='unmap_t',
      default=0.3, type='float',
      help='Remove sequences with more than this unmappable bin % [Default: %default]')
  parser.add_option('-w', dest='pool_width',
      default=128, type='int',
      help='Sum pool width [Default: %default]')
  parser.add_option('-v', dest='valid_pct',
      default=0.05, type='float',
      help='Proportion of the data for validation [Default: %default]')
  (options, args) = parser.parse_args()

  if len(args) != 4:
    parser.error('Must provide FASTA and sample coverage label and path files for two genomes.')
  else:
    fasta1_file = args[0]
    targets1_file = args[1]
    fasta2_file = args[2]
    targets2_file = args[3]

  random.seed(options.seed)
  np.random.seed(options.seed)

  if not os.path.isdir(options.out_dir):
    os.mkdir(options.out_dir)

  ################################################################
  # define genomic contigs
  ################################################################
  genome1_chr_contigs = genome.load_chromosomes(fasta1_file)
  genome2_chr_contigs = genome.load_chromosomes(fasta2_file)

  # remove gaps
  if options.gaps1_file:
    genome1_chr_contigs = genome.split_contigs(genome1_chr_contigs,
                                               options.gaps1_file)
  if options.gaps2_file:
    genome2_chr_contigs = genome.split_contigs(genome2_chr_contigs,
                                               options.gaps2_file)

  # ditch the chromosomes
  contigs = []
  for chrom in genome1_chr_contigs:
    contigs += [Contig(0, chrom, ctg_start, ctg_end)
                for ctg_start, ctg_end in genome1_chr_contigs[chrom]]
  for chrom in genome2_chr_contigs:
    contigs += [Contig(1, chrom, ctg_start, ctg_end)
                for ctg_start, ctg_end in genome2_chr_contigs[chrom]]

  # filter for large enough
  contigs = [ctg for ctg in contigs if ctg.end - ctg.start >= options.seq_length]

  # break up large contigs
  if options.break_t is not None:
    contigs = break_large_contigs(contigs, options.break_t)

  # down-sample
  if options.sample_pct < 1.0:
    contigs = random.sample(contigs, int(options.sample_pct*len(contigs)))

  # print contigs to BED file
  for genome_i in range(2):
    contigs_i = [ctg for ctg in contigs if ctg.genome == genome_i]
    ctg_bed_file = '%s/contigs%d.bed' % (options.out_dir, genome_i)
    write_seqs_bed(ctg_bed_file, contigs_i)

  ################################################################
  # divide between train/valid/test
  ################################################################

  # connect contigs across genomes by alignment
  contig_components = connect_contigs(contigs, options.align_net, options.fill_min, options.out_dir)

  # divide contig connected components between train/valid/test
  contig_sets = divide_contig_components(contig_components,
                                         options.test_pct,
                                         options.valid_pct)

  train_contigs, valid_contigs, test_contigs = contig_sets

  ################################################################
  # define model sequences
  ################################################################

  # data2 notes
  #
  # when shuffling, I think we're going to want to carefully
  # evenly distribute the sequences. so shuffle within genome,
  # and then interleave.

  # stride sequences across contig
  train_mseqs = contig_sequences(train_contigs, options.seq_length, options.stride_train)
  valid_mseqs = contig_sequences(valid_contigs, options.seq_length, options.stride_test)
  test_mseqs = contig_sequences(test_contigs, options.seq_length, options.stride_test)

  # shuffle
  random.shuffle(train_mseqs)
  random.shuffle(valid_mseqs)
  random.shuffle(test_mseqs)

  # merge
  mseqs = train_mseqs + valid_mseqs + test_mseqs
  mseqs_labels = ['train']*len(train_mseqs) + ['valid']*len(valid_mseqs) + ['test']*len(test_mseqs)


  ################################################################
  # mappability
  ################################################################

  # data2 notes
  #
  # given two unmap bed files, this is easy

  if options.unmap_bed is not None:
    # annotate unmappable positions
    mseqs_unmap = annotate_unmap(mseqs, options.unmap_bed,
                                 options.seq_length, options.pool_width)

    # filter unmappable
    mseqs_map_mask = (mseqs_unmap.mean(axis=1, dtype='float64') < options.unmap_t)
    mseqs = [mseqs[i] for i in range(len(mseqs)) if mseqs_map_mask[i]]
    mseqs_labels = [mseqs_labels[i] for i in range(len(mseqs_labels)) if mseqs_map_mask[i]]
    mseqs_unmap = mseqs_unmap[mseqs_map_mask,:]

    # write to file
    unmap_npy = '%s/mseqs_unmap.npy' % options.out_dir
    np.save(unmap_npy, mseqs_unmap)

  # write sequences to BED
  seqs_bed_file = '%s/sequences.bed' % options.out_dir
  write_seqs_bed(seqs_bed_file, mseqs, mseqs_labels)


  ################################################################
  # read sequence coverage values
  ################################################################

  # data2 notes
  #
  # move this to a function that takes seqs_bed_file and targets_file
  # separately for each genome

  # read target datasets
  targets_df = pd.read_table(targets_file)

  seqs_cov_dir = '%s/seqs_cov' % options.out_dir
  if not os.path.isdir(seqs_cov_dir):
    os.mkdir(seqs_cov_dir)

  read_jobs = []

  for ti in range(targets_df.shape[0]):
    genome_cov_file = targets_df['file'].iloc[ti]
    seqs_cov_stem = '%s/%d' % (seqs_cov_dir, ti)
    seqs_cov_file = '%s.h5' % seqs_cov_stem

    if os.path.isfile(seqs_cov_file):
      print('Skipping existing %s' % seqs_cov_file, file=sys.stderr)
    else:
      cmd = 'basenji_data_read.py'
      cmd += ' -w %d' % options.pool_width
      cmd += ' %s' % genome_cov_file
      cmd += ' %s' % seqs_bed_file
      cmd += ' %s' % seqs_cov_file

      if options.run_local:
        cmd += ' &> %s.err' % seqs_cov_stem
        read_jobs.append(cmd)
      else:
        j = slurm.Job(cmd,
            name='read_t%d' % ti,
            out_file='%s.out' % seqs_cov_stem,
            err_file='%s.err' % seqs_cov_stem,
            queue='standard,tbdisk', mem=15000, time='12:0:0')
        read_jobs.append(j)

  if options.run_local:
    util.exec_par(read_jobs, options.processes, verbose=True)
  else:
    slurm.multi_run(read_jobs, options.processes, verbose=True, sleep_time=1)

  ################################################################
  # write TF Records
  ################################################################

  # data2 notes
  #
  # every TFRecord file should have the same proportion of sequences
  # from each genome.
  # that can be accomplished by sorting mseqs accordingly
  #
  # the TFRecord itself needs a 'genome' int variable added that will be used
  # to gate the final model layer. it can read it out of seqs_bed_file
  #
  # well, that will only work if the 'targets' byteslist can have variable size.

  tfr_dir = '%s/tfrecords' % options.out_dir
  if not os.path.isdir(tfr_dir):
    os.mkdir(tfr_dir)

  write_jobs = []

  for tvt_set in ['train', 'valid', 'test']:
    tvt_set_indexes = [i for i in range(len(mseqs_labels)) if mseqs_labels[i] == tvt_set]
    tvt_set_start = tvt_set_indexes[0]
    tvt_set_end = tvt_set_indexes[-1]

    tfr_i = 0
    tfr_start = tvt_set_start
    tfr_end = min(tfr_start+options.seqs_per_tfr, tvt_set_end)

    while tfr_start <= tvt_set_end:
      tfr_stem = '%s/%s-%d' % (tfr_dir, tvt_set, tfr_i)

      cmd = 'basenji_data_write.py'
      cmd += ' -s %d' % tfr_start
      cmd += ' -e %d' % tfr_end
      if options.unmap_bed is not None:
        cmd += ' -u %s' % unmap_npy

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
              queue='standard,tbdisk', mem=15000, time='12:0:0')
        write_jobs.append(j)

      # update
      tfr_i += 1
      tfr_start += options.seqs_per_tfr
      tfr_end = min(tfr_start+options.seqs_per_tfr, tvt_set_end)

  if options.run_local:
    util.exec_par(write_jobs, options.processes, verbose=True)
  else:
    slurm.multi_run(write_jobs, options.processes, verbose=True, sleep_time=1)


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

    assert(a[3].startswith('chr'))
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

      # add left
      ctg_left = Contig(ctg.genome, ctg.chr, ctg.start, ctg_mid)
      ctg_left_len = ctg_left.end - ctg_left.start
      heapq.heappush(contig_heapq, (-ctg_left_len, ctg_left))

      # add right
      ctg_right = Contig(ctg.genome, ctg.chr, ctg_mid, ctg.end)
      ctg_right_len = ctg_right.end - ctg_right.start
      heapq.heappush(contig_heapq, (-ctg_right_len, ctg_right))

  # return to list
  contigs = [len_ctg[1] for len_ctg in contig_heapq]

  return contigs


################################################################################
def contig_sequences(contigs, seq_length, stride):
  ''' Break up a list of Contig's into a list of ModelSeq's. '''
  mseqs = []

  for ctg in contigs:
    seq_start = ctg.start
    seq_end = seq_start + seq_length

    while seq_end < ctg.end:
      # record sequence
      mseqs.append(ModelSeq(ctg.genome, ctg.chr, seq_start, seq_end))

      # update
      seq_start += int(stride*seq_length)
      seq_end += int(stride*seq_length)

  return mseqs


################################################################################
def connect_contigs(contigs, align_net_file, fill_min, out_dir):
  """Connect contigs across genomes by forming a graph that includes
     net format aligning regions and contigs. Compute contig components
     as connected components of that graph."""

  # construct align net graph and write net BEDs
  graph_contigs_nets = make_net_graph(align_net_file, fill_min, out_dir)

  # add contig nodes
  for ctg in contigs:
    ctg_node = GraphSeq(ctg.genome, False, ctg.chr, ctg.start, ctg.end)
    graph_contigs_nets.add_node(ctg_node)

  # intersect contigs BED w/ nets BED, adding graph edges.
  intersect_contigs_nets(graph_contigs_nets, 0, out_dir)
  intersect_contigs_nets(graph_contigs_nets, 1, out_dir)

  # find connected components
  contig_components = []
  for contig_net_component in nx.connected_components(graph_contigs_nets):
    # extract only the contigs
    cc_contigs = [contig_or_net for contig_or_net in contig_net_component if contig_or_net.net is False]

    if cc_contigs:
      # add to list
      contig_components.append(cc_contigs)

  # write summary stats
  comp_out = open('%s/contig_components.txt' % out_dir, 'w')
  for ctg_comp in contig_components:
    ctg_comp0 = [ctg for ctg in ctg_comp if ctg.genome == 0]
    ctg_comp1 = [ctg for ctg in ctg_comp if ctg.genome == 1]
    ctg_comp0_nt = sum([ctg.end-ctg.start for ctg in ctg_comp0])
    ctg_comp1_nt = sum([ctg.end-ctg.start for ctg in ctg_comp1])
    ctg_comp_nt = ctg_comp0_nt + ctg_comp1_nt
    cols = [len(ctg_comp), len(ctg_comp0), len(ctg_comp1)]
    cols += [ctg_comp0_nt, ctg_comp1_nt, ctg_comp_nt]
    cols = [str(c) for c in cols]
    print('\t'.join(cols), file=comp_out)
  comp_out.close()

  return contig_components


################################################################################
def divide_contig_components(contig_components, test_pct, valid_pct, pct_abstain=0.5):
  """Divide contig connected components into train/valid/test,
     and aiming for the specified nucleotide percentages."""

  # sort contig components descending by length
  length_contig_components = []
  for cc_contigs in contig_components:
    cc_len = sum([ctg.end-ctg.start for ctg in cc_contigs])
    length_contig_components.append((cc_len, cc_contigs))
  length_contig_components.sort(reverse=True)

  # compute total nucleotides
  total_nt = sum([lc[0] for lc in length_contig_components])

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

  # process contigs
  for ctg_comp_len, ctg_comp in length_contig_components:

    # compute gap between current and aim
    test_nt_gap = max(0, test_nt_aim - test_nt)
    valid_nt_gap = max(0, valid_nt_aim - valid_nt)
    train_nt_gap = max(1, train_nt_aim - train_nt)

    # skip if too large
    if ctg_comp_len > pct_abstain*test_nt_gap:
      test_nt_gap = 0
    if ctg_comp_len > pct_abstain*valid_nt_gap:
      valid_nt_gap = 0

    # compute remaining %
    gap_sum = train_nt_gap + valid_nt_gap + test_nt_gap
    test_pct_gap = test_nt_gap / gap_sum
    valid_pct_gap = valid_nt_gap / gap_sum
    train_pct_gap = train_nt_gap / gap_sum

    # sample train/valid/test
    ri = np.random.choice(range(3), 1, p=[train_pct_gap, valid_pct_gap, test_pct_gap])[0]
    if ri == 0:
      for ctg in ctg_comp:
        train_contigs.append(ctg)
      train_nt += ctg_comp_len
    elif ri == 1:
      for ctg in ctg_comp:
        valid_contigs.append(ctg)
      valid_nt += ctg_comp_len
    elif ri == 2:
      for ctg in ctg_comp:
        test_contigs.append(ctg)
      test_nt += ctg_comp_len
    else:
      print('TVT random number beyond 0,1,2', file=sys.stderr)
      exit(1)

  # report genome-specific train/valid/test stats
  report_divide_stats(train_contigs, valid_contigs, test_contigs)

  return train_contigs, valid_contigs, test_contigs


def report_divide_stats(train_contigs, valid_contigs, test_contigs):
  """ Report genome-specific statistics about the division of contigs
      between train/valid/test sets."""

  # compute genome-specific stats
  train_count_genome, train_nt_genome = contig_stats_genome(train_contigs)
  valid_count_genome, valid_nt_genome = contig_stats_genome(valid_contigs)
  test_count_genome, test_nt_genome = contig_stats_genome(test_contigs)

  # sum nt across genomes
  train_nt = sum(train_nt_genome)
  valid_nt = sum(valid_nt_genome)
  test_nt = sum(test_nt_genome)
  total_nt = train_nt + valid_nt + test_nt

  # compute total sum nt per genome
  total_nt_genome = []
  for gi in range(2):
    total_nt_gi = train_nt_genome[gi] + valid_nt_genome[gi] + test_nt_genome[gi]
    total_nt_genome.append(total_nt_gi)

  print('Contigs divided into')
  print(' Train: %5d contigs, %10d nt (%.4f)' % \
       (len(train_contigs), train_nt, train_nt/total_nt))
  for gi in range(2):
    print('  Genome%d: %5d contigs, %10d nt (%.4f)' % \
         (gi, train_count_genome[gi], train_nt_genome[gi], train_nt_genome[gi]/total_nt_genome[gi]))

  print(' Valid: %5d contigs, %10d nt (%.4f)' % \
      (len(valid_contigs), valid_nt, valid_nt/total_nt))
  for gi in range(2):
    print('  Genome%d: %5d contigs, %10d nt (%.4f)' % \
         (gi, valid_count_genome[gi], valid_nt_genome[gi], valid_nt_genome[gi]/total_nt_genome[gi]))

  print(' Test:  %5d contigs, %10d nt (%.4f)' % \
      (len(test_contigs), test_nt, test_nt/total_nt))
  for gi in range(2):
    print('  Genome%d: %5d contigs, %10d nt (%.4f)' % \
         (gi, test_count_genome[gi], test_nt_genome[gi], test_nt_genome[gi]/total_nt_genome[gi]))


def contig_stats_genome(contigs):
  contigs_count_genome = []
  contigs_nt_genome = []

  for gi in range(2):
    contigs_genome = [ctg for ctg in contigs if ctg.genome == gi]
    contigs_nt = [ctg.end-ctg.start for ctg in contigs_genome]

    contigs_count_genome.append(len(contigs_genome))
    contigs_nt_genome.append(sum(contigs_nt))

  return contigs_count_genome, contigs_nt_genome


def make_net_graph(align_net_file, fill_min, out_dir):
  """Construct a Graph with aligned net intervals connected
     by edges."""

  nets1_bed_out = open('%s/nets0.bed' % out_dir, 'w')
  nets2_bed_out = open('%s/nets1.bed' % out_dir, 'w')

  graph_nets = nx.Graph()

  for net_line in open(align_net_file):
    if net_line.startswith('net'):
      net_a = net_line.split()
      chrom1 = net_a[1]

    elif net_line.startswith(' fill'):
      net_a = net_line.split()

      # extract genome1 interval
      start1 = int(net_a[1])
      size1 = int(net_a[2])
      end1 = start1+size1

      # extract genome2 interval
      chrom2 = net_a[3]
      start2 = int(net_a[5])
      size2 = int(net_a[6])
      end2 = start2+size2

      if min(size1, size2) >= fill_min:
        # add edge
        net1_node = GraphSeq(0, True, chrom1, start1, end1)
        net2_node = GraphSeq(1, True, chrom2, start2, end2)
        graph_nets.add_edge(net1_node, net2_node)

        # write interval1
        cols = [chrom1, str(start1), str(end1)]
        print('\t'.join(cols), file=nets1_bed_out)

        # write interval2
        cols = [chrom2, str(start2), str(end2)]
        print('\t'.join(cols), file=nets2_bed_out)

  nets1_bed_out.close()
  nets2_bed_out.close()

  return graph_nets


def write_net_beds(align_net_file, out_dir):
  nets1_bed_out = open('%s/nets0.bed' % out_dir, 'w')
  nets2_bed_out = open('%s/nets1.bed' % out_dir, 'w')

  for net_line in open(align_net_file):
    if net_line.startswith('net'):
      net_a = net_line.split()
      chrom1 = net_a[1]

    elif net_line.startswith(' fill'):
      net_a = net_line.split()

      # write genome1 interval
      start1 = int(net_a[1])
      size1 = int(net_a[2])
      end1 = start1+size1
      cols = [chrom1, str(start1), str(end1)]
      print('\t'.join(cols), file=nets1_bed_out)

      # write genome2 interval
      chrom2 = net_a[3]
      start2 = int(net_a[5])
      size2 = int(net_a[6])
      end2 = start2+size2
      cols = [chrom2, str(start2), str(end2)]
      print('\t'.join(cols), file=nets2_bed_out)

  nets1_bed_out.close()
  nets2_bed_out.close()


def intersect_contigs_nets(graph_contigs_nets, genome_i, out_dir):
  """Intersect the contigs and nets from genome_i, adding the
     overlaps as edges to graph_contigs_nets."""

  contigs_file = '%s/contigs%d.bed' % (out_dir, genome_i)
  nets_file = '%s/nets%d.bed' % (out_dir, genome_i)

  contigs_bed = pybedtools.BedTool(contigs_file)
  nets_bed = pybedtools.BedTool(nets_file)

  for overlap in contigs_bed.intersect(nets_bed, wo=True):
    ctg_chr = overlap[0]
    ctg_start = int(overlap[1])
    ctg_end = int(overlap[2])
    net_chr = overlap[3]
    net_start = int(overlap[4])
    net_end = int(overlap[5])

    # create node objects
    ctg_node = GraphSeq(genome_i, False, ctg_chr, ctg_start, ctg_end)
    net_node = GraphSeq(genome_i, True, net_chr, net_start, net_end)

    # add edge / verify we found nodes
    gcn_size_pre = graph_contigs_nets.number_of_nodes()
    graph_contigs_nets.add_edge(ctg_node, net_node)
    gcn_size_post = graph_contigs_nets.number_of_nodes()
    assert(gcn_size_pre == gcn_size_post)


################################################################################
def divide_contigs(contigs, test_pct, valid_pct, pct_abstain=0.2):
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

  # initialie train/valid/test contig lists
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
Contig = collections.namedtuple('Contig', ['genome', 'chr', 'start', 'end'])
ModelSeq = collections.namedtuple('ModelSeq', ['genome', 'chr', 'start', 'end'])
GraphSeq = collections.namedtuple('GraphSeq', ['genome', 'net', 'chr', 'start', 'end'])


################################################################################
if __name__ == '__main__':
  main()
