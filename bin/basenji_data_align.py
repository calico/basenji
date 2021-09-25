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
import gzip
import heapq
import pdb
import os
import random
import sys
import time

import networkx as nx
import numpy as np
import pybedtools

from basenji import genome
from basenji_data import annotate_unmap, rejoin_large_contigs, write_seqs_bed

'''
basenji_data_align.py

Partition sequences from multiple aligned genomes into train/valid/test splits
that respect homology.
'''

################################################################################
def main():
  usage = 'usage: %prog [options] <align_net> <fasta0_file,fasta1_file>'
  parser = OptionParser(usage)
  parser.add_option('-a', dest='genome_labels',
      default=None, help='Genome labels in output')
  parser.add_option('--break', dest='break_t',
      default=None, type='int',
      help='Break in half contigs above length [Default: %default]')
  parser.add_option('-c','--crop', dest='crop_bp',
      default=0, type='int',
      help='Crop bp off each end [Default: %default]')
  parser.add_option('-d', dest='sample_pct',
      default=1.0, type='float',
      help='Down-sample the segments')
  parser.add_option('-f', dest='folds',
      default=None, type='int',
      help='Generate cross fold split [Default: %default]')
  parser.add_option('-g', dest='gap_files',
      help='Comma-separated list of assembly gaps BED files [Default: %default]')
  parser.add_option('-l', dest='seq_length',
      default=131072, type='int',
      help='Sequence length [Default: %default]')
  parser.add_option('--nf', dest='net_fill_min',
    default=100000, type='int',
    help='Alignment net fill size minimum [Default: %default]')
  parser.add_option('--no', dest='net_olap_min',
    default=1024, type='int',
    help='Alignment net and contig overlap minimum [Default: %default]')
  parser.add_option('-o', dest='out_dir',
      default='align_out',
      help='Output directory [Default: %default]')
  parser.add_option('--seed', dest='seed',
      default=44, type='int',
      help='Random seed [Default: %default]')
  parser.add_option('--snap', dest='snap',
      default=1, type='int',
      help='Snap sequences to multiple of the given value [Default: %default]')
  parser.add_option('--stride', '--stride_train', dest='stride_train',
      default=1., type='float',
      help='Stride to advance train sequences [Default: seq_length]')
  parser.add_option('--stride_test', dest='stride_test',
      default=1., type='float',
      help='Stride to advance valid and test sequences [Default: %default]')
  parser.add_option('-t', dest='test_pct',
      default=0.1, type='float',
      help='Proportion of the data for testing [Default: %default]')
  parser.add_option('-u', dest='umap_beds',
      help='Comma-separated genome unmappable segments to set to NA')
  parser.add_option('--umap_t', dest='umap_t',
      default=0.5, type='float',
      help='Remove sequences with more than this unmappable bin % [Default: %default]')
  parser.add_option('-w', dest='pool_width',
      default=128, type='int',
      help='Sum pool width [Default: %default]')
  parser.add_option('-v', dest='valid_pct',
      default=0.1, type='float',
      help='Proportion of the data for validation [Default: %default]')
  (options, args) = parser.parse_args()

  if len(args) != 2:
    parser.error('Must provide alignment and FASTA files.')
  else:
    align_net_file = args[0]
    fasta_files = args[1].split(',')

  # there is still some source of stochasticity
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

  # check snap
  if options.snap is not None:
    if np.mod(options.seq_length, options.snap) != 0: 
      raise ValueError('seq_length must be a multiple of snap')
    if np.mod(options.stride_train, options.snap) != 0: 
      raise ValueError('stride_train must be a multiple of snap')
    if np.mod(options.stride_test, options.snap) != 0:
      raise ValueError('stride_test must be a multiple of snap')

  # count genomes
  num_genomes = len(fasta_files)

  # parse gap files
  if options.gap_files is not None:
    options.gap_files = options.gap_files.split(',')
    assert(len(options.gap_files) == num_genomes)

  # parse unmappable files
  if options.umap_beds is not None:
    options.umap_beds = options.umap_beds.split(',')
    assert(len(options.umap_beds) == num_genomes)

  # label genomes
  if options.genome_labels is None:
    options.genome_labels = ['genome%d' % (gi+1) for gi in range(num_genomes)]
  else:
    options.genome_labels = options.genome_labels.split(',')
    assert(len(options.genome_labels) == num_genomes)

  # create output directorys
  if not os.path.isdir(options.out_dir):
    os.mkdir(options.out_dir)
  genome_out_dirs = []
  for gi in range(num_genomes):
    gout_dir = '%s/%s' % (options.out_dir, options.genome_labels[gi])
    if not os.path.isdir(gout_dir):
      os.mkdir(gout_dir)
    genome_out_dirs.append(gout_dir)

  ################################################################
  # define genomic contigs
  ################################################################
  genome_chr_contigs = []
  for gi in range(num_genomes):
    genome_chr_contigs.append(genome.load_chromosomes(fasta_files[gi]))

    # remove gaps
    if options.gap_files[gi]:
      genome_chr_contigs[gi] = genome.split_contigs(genome_chr_contigs[gi],
                                                    options.gap_files[gi])

  # ditch the chromosomes
  contigs = []
  for gi in range(num_genomes):
    for chrom in genome_chr_contigs[gi]:
      contigs += [Contig(gi, chrom, ctg_start, ctg_end)
                  for ctg_start, ctg_end in genome_chr_contigs[gi][chrom]]

  # filter for large enough
  seq_tlength = options.seq_length - 2*options.crop_bp
  contigs = [ctg for ctg in contigs if ctg.end - ctg.start >= seq_tlength]

  # break up large contigs
  if options.break_t is not None:
    contigs = break_large_contigs(contigs, options.break_t)

  # print contigs to BED file
  for gi in range(num_genomes):
    contigs_i = [ctg for ctg in contigs if ctg.genome == gi]
    ctg_bed_file = '%s/contigs.bed' % genome_out_dirs[gi]
    write_seqs_bed(ctg_bed_file, contigs_i)

  ################################################################
  # divide between train/valid/test
  ################################################################

  # connect contigs across genomes by alignment
  contig_components = connect_contigs(contigs, align_net_file, options.net_fill_min,
                                      options.net_olap_min, options.out_dir, genome_out_dirs)

  if options.folds is not None:
    # divide by fold
    fold_contigs = divide_components_folds(contig_components, options.folds)

  else:
    # divide by train/valid/test pct
    fold_contigs = divide_components_pct(contig_components, options.test_pct,
                                         options.valid_pct)

  # rejoin broken contigs within set
  for fi in range(len(fold_contigs)):
    fold_contigs[fi] = rejoin_large_contigs(fold_contigs[fi])

  # label folds
  if options.folds is not None:
    fold_labels = ['fold%d' % fi for fi in range(options.folds)]
    num_folds = options.folds
  else:
    fold_labels = ['train', 'valid', 'test']
    num_folds = 3

  if options.folds is None:
    # quantify leakage across sets
    quantify_leakage(align_net_file, fold_contigs[0], fold_contigs[1],
                     fold_contigs[2], options.out_dir)

  ################################################################
  # define model sequences
  ################################################################

  fold_mseqs = []
  for fi in range(num_folds):
    if fold_labels[fi] in ['valid','test']:
      stride_fold = options.stride_test
    else:
      stride_fold = options.stride_train

    # stride sequences across contig
    fold_mseqs_fi = contig_sequences(fold_contigs[fi], seq_tlength,
                                     stride_fold, options.snap, fold_labels[fi])
    fold_mseqs.append(fold_mseqs_fi)

    # shuffle
    random.shuffle(fold_mseqs[fi])

    # down-sample
    if options.sample_pct < 1.0:
      fold_mseqs[fi] = random.sample(fold_mseqs[fi], int(options.sample_pct*len(fold_mseqs[fi])))

  # merge into one list
  mseqs = [ms for fm in fold_mseqs for ms in fm]

  # separate by genome
  mseqs_genome = []
  for gi in range(num_genomes):
    mseqs_gi = [mseqs[si] for si in range(len(mseqs)) if mseqs[si].genome == gi]
    mseqs_genome.append(mseqs_gi)

  ################################################################
  # filter for sufficient mappability
  ################################################################
  for gi in range(num_genomes):
    if options.umap_beds[gi] is not None:
      # annotate unmappable positions
      mseqs_unmap = annotate_unmap(mseqs_genome[gi], options.umap_beds[gi],
                                   seq_tlength, options.pool_width)

      # filter unmappable
      mseqs_map_mask = (mseqs_unmap.mean(axis=1, dtype='float64') < options.umap_t)
      mseqs_genome[gi] = [mseqs_genome[gi][si] for si in range(len(mseqs_genome[gi])) if mseqs_map_mask[si]]
      mseqs_unmap = mseqs_unmap[mseqs_map_mask,:]

      # write to file
      unmap_npy_file = '%s/mseqs_unmap.npy' % genome_out_dirs[gi]
      np.save(unmap_npy_file, mseqs_unmap)

  seqs_bed_files = []
  for gi in range(num_genomes):
    # write sequences to BED
    seqs_bed_files.append('%s/sequences.bed' % genome_out_dirs[gi])
    write_seqs_bed(seqs_bed_files[gi], mseqs_genome[gi], True)



################################################################################
def quantify_leakage(align_net_file, train_contigs, valid_contigs, test_contigs, out_dir):
  """Quanitfy the leakage across sequence sets."""

  def split_genome(contigs):
    genome_contigs = []
    for ctg in contigs:
      while len(genome_contigs) <= ctg.genome:
        genome_contigs.append([])
      genome_contigs[ctg.genome].append((ctg.chr,ctg.start,ctg.end))
    genome_bedtools = [pybedtools.BedTool(ctgs) for ctgs in genome_contigs]
    return genome_bedtools

  def bed_sum(overlaps):
    osum = 0
    for overlap in overlaps:
      osum += int(overlap[2]) - int(overlap[1])
    return osum

  train0_bt, train1_bt = split_genome(train_contigs)
  valid0_bt, valid1_bt = split_genome(valid_contigs)
  test0_bt, test1_bt = split_genome(test_contigs)

  assign0_sums = {}
  assign1_sums = {}

  if os.path.splitext(align_net_file)[-1] == '.gz':
    align_net_open = gzip.open(align_net_file, 'rt')
  else:
    align_net_open = open(align_net_file, 'r')

  for net_line in align_net_open:
    if net_line.startswith('net'):
      net_a = net_line.split()
      chrom0 = net_a[1]

    elif net_line.startswith(' fill'):
      net_a = net_line.split()

      # extract genome1 interval
      start0 = int(net_a[1])
      size0 = int(net_a[2])
      end0 = start0+size0
      align0_bt = pybedtools.BedTool([(chrom0,start0,end0)]) 

      # extract genome2 interval
      chrom1 = net_a[3]
      start1 = int(net_a[5])
      size1 = int(net_a[6])
      end1 = start1+size1
      align1_bt = pybedtools.BedTool([(chrom1,start1,end1)])

      # count interval overlap
      align0_train_bp = bed_sum(align0_bt.intersect(train0_bt))
      align0_valid_bp = bed_sum(align0_bt.intersect(valid0_bt))
      align0_test_bp = bed_sum(align0_bt.intersect(test0_bt))
      align0_max_bp = max(align0_train_bp, align0_valid_bp, align0_test_bp)

      align1_train_bp = bed_sum(align1_bt.intersect(train1_bt))
      align1_valid_bp = bed_sum(align1_bt.intersect(valid1_bt))
      align1_test_bp = bed_sum(align1_bt.intersect(test1_bt))
      align1_max_bp = max(align1_train_bp, align1_valid_bp, align1_test_bp)

      # assign to class
      if align0_max_bp == 0:
        assign0 = None
      elif align0_train_bp == align0_max_bp:
        assign0 = 'train'
      elif align0_valid_bp == align0_max_bp:
        assign0 = 'valid'
      elif align0_test_bp == align0_max_bp:
        assign0 = 'test'
      else:
        print('Bad logic')
        exit(1)

      if align1_max_bp == 0:
        assign1 = None
      elif align1_train_bp == align1_max_bp:
        assign1 = 'train'
      elif align1_valid_bp == align1_max_bp:
        assign1 = 'valid'
      elif align1_test_bp == align1_max_bp:
        assign1 = 'test'
      else:
        print('Bad logic')
        exit(1)

      # increment
      assign0_sums[(assign0,assign1)] = assign0_sums.get((assign0,assign1),0) + align0_max_bp
      assign1_sums[(assign0,assign1)] = assign1_sums.get((assign0,assign1),0) + align1_max_bp

  # sum contigs
  splits0_bp = {}
  splits0_bp['train'] = bed_sum(train0_bt)
  splits0_bp['valid'] = bed_sum(valid0_bt)
  splits0_bp['test'] = bed_sum(test0_bt)
  splits1_bp = {}
  splits1_bp['train'] = bed_sum(train1_bt)
  splits1_bp['valid'] = bed_sum(valid1_bt)
  splits1_bp['test'] = bed_sum(test1_bt)

  leakage_out = open('%s/leakage.txt' % out_dir, 'w')
  print('Genome0', file=leakage_out)
  for split0 in ['train','valid','test']:
    print('  %5s: %10d nt' % (split0, splits0_bp[split0]), file=leakage_out)
    for split1 in ['train','valid','test',None]:
      ss_bp = assign0_sums.get((split0,split1),0)
      print('    %5s: %10d (%.5f)' % (split1, ss_bp, ss_bp/splits0_bp[split0]), file=leakage_out)
  print('\nGenome1', file=leakage_out)
  for split1 in ['train','valid','test']:
    print('  %5s: %10d nt' % (split1, splits1_bp[split1]), file=leakage_out)
    for split0 in ['train','valid','test',None]:
      ss_bp = assign1_sums.get((split0,split1),0)
      print('    %5s: %10d (%.5f)' % (split0, ss_bp, ss_bp/splits1_bp[split1]), file=leakage_out)
  leakage_out.close()



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
def contig_sequences(contigs, seq_length, stride, snap=1, label=None):
  ''' Break up a list of Contig's into a list of model length
       and stride sequence contigs.'''
  mseqs = []

  for ctg in contigs:
    seq_start = int(np.ceil(ctg.start/snap)*snap)
    seq_end = seq_start + seq_length

    while seq_end < ctg.end:
      # record sequence
      mseqs.append(ModelSeq(ctg.genome, ctg.chr, seq_start, seq_end, label))

      # update
      seq_start += stride
      seq_end += stride

  return mseqs


################################################################################
def connect_contigs(contigs, align_net_file, net_fill_min, net_olap_min, out_dir, genome_out_dirs):
  """Connect contigs across genomes by forming a graph that includes
     net format aligning regions and contigs. Compute contig components
     as connected components of that graph."""

  # construct align net graph and write net BEDs
  if align_net_file is None:
    graph_contigs_nets = nx.Graph()
  else:
    graph_contigs_nets = make_net_graph(align_net_file, net_fill_min, out_dir)

  # add contig nodes
  for ctg in contigs:
    ctg_node = GraphSeq(ctg.genome, False, ctg.chr, ctg.start, ctg.end)
    graph_contigs_nets.add_node(ctg_node)

  # intersect contigs BED w/ nets BED, adding graph edges.
  intersect_contigs_nets(graph_contigs_nets, 0, out_dir, genome_out_dirs[0], net_olap_min)
  intersect_contigs_nets(graph_contigs_nets, 1, out_dir, genome_out_dirs[1], net_olap_min)

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
def contig_stats_genome(contigs):
  """Compute contig statistics within each genome."""
  contigs_count_genome = []
  contigs_nt_genome = []

  contigs_genome_found = True
  gi = 0
  while contigs_genome_found:
    contigs_genome = [ctg for ctg in contigs if ctg.genome == gi]

    if len(contigs_genome) == 0:
      contigs_genome_found = False

    else:
      contigs_nt = [ctg.end-ctg.start for ctg in contigs_genome]

      contigs_count_genome.append(len(contigs_genome))
      contigs_nt_genome.append(sum(contigs_nt))

      gi += 1

  return contigs_count_genome, contigs_nt_genome


################################################################################
def divide_components_folds(contig_components, folds):
  """Divide contig connected components into cross fold lists."""

  # sort contig components descending by length
  length_contig_components = []
  for cc_contigs in contig_components:
    cc_len = sum([ctg.end-ctg.start for ctg in cc_contigs])
    length_contig_components.append((cc_len, cc_contigs))
  length_contig_components.sort(reverse=True)

  # compute total nucleotides
  total_nt = sum([lc[0] for lc in length_contig_components])

  # compute aimed fold nucleotides
  fold_nt_aim = int(np.ceil(total_nt / folds))

  # initialize current fold nucleotides
  fold_nt = np.zeros(folds)

  # initialize fold contig lists
  fold_contigs = []
  for fi in range(folds):
    fold_contigs.append([])

  # process contigs
  for ctg_comp_len, ctg_comp in length_contig_components:
    # compute gap between current and aim
    fold_nt_gap = fold_nt_aim - fold_nt
    fold_nt_gap = np.clip(fold_nt_gap, 0, np.inf)

    # compute sample probability
    fold_prob = fold_nt_gap / fold_nt_gap.sum()

    # sample train/valid/test
    fi = np.random.choice(folds, p=fold_prob)
    fold_nt[fi] += ctg_comp_len
    for ctg in ctg_comp:
      fold_contigs[fi].append(ctg)
    
  # report genome-specific train/valid/test stats
  report_divide_stats(fold_contigs)

  return fold_contigs


################################################################################
def divide_components_pct(contig_components, test_pct, valid_pct, pct_abstain=0.5):
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

    # collect contigs (sorted is required for deterministic sequence order)
    if ri == 0:
      for ctg in sorted(ctg_comp):
        train_contigs.append(ctg)
      train_nt += ctg_comp_len
    elif ri == 1:
      for ctg in sorted(ctg_comp):
        valid_contigs.append(ctg)
      valid_nt += ctg_comp_len
    elif ri == 2:
      for ctg in sorted(ctg_comp):
        test_contigs.append(ctg)
      test_nt += ctg_comp_len
    else:
      print('TVT random number beyond 0,1,2', file=sys.stderr)
      exit(1)

  # report genome-specific train/valid/test stats
  report_divide_stats([train_contigs, valid_contigs, test_contigs])

  return train_contigs, valid_contigs, test_contigs


################################################################################
def intersect_contigs_nets(graph_contigs_nets, genome_i, out_dir, genome_out_dir, min_olap=128):
  """Intersect the contigs and nets from genome_i, adding the
     overlaps as edges to graph_contigs_nets."""

  contigs_file = '%s/contigs.bed' % genome_out_dir
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
    olap_len = int(overlap[6])

    if olap_len > min_olap:
      # create node objects
      ctg_node = GraphSeq(genome_i, False, ctg_chr, ctg_start, ctg_end)
      net_node = GraphSeq(genome_i, True, net_chr, net_start, net_end)

      # add edge / verify we found nodes
      gcn_size_pre = graph_contigs_nets.number_of_nodes()
      graph_contigs_nets.add_edge(ctg_node, net_node)
      gcn_size_post = graph_contigs_nets.number_of_nodes()
      assert(gcn_size_pre == gcn_size_post)


################################################################################
def make_net_graph(align_net_file, net_fill_min, out_dir):
  """Construct a Graph with aligned net intervals connected
     by edges."""

  graph_nets = nx.Graph()

  nets1_bed_out = open('%s/nets0.bed' % out_dir, 'w')
  nets2_bed_out = open('%s/nets1.bed' % out_dir, 'w')

  if os.path.splitext(align_net_file)[-1] == '.gz':
    align_net_open = gzip.open(align_net_file, 'rt')
  else:
    align_net_open = open(align_net_file, 'r')

  for net_line in align_net_open:
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

      if min(size1, size2) >= net_fill_min:
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


################################################################################
def rejoin_large_contigs(contigs):
  """ Rejoin large contigs that were broken up before alignment comparison."""

  # split list by genome/chromosome
  gchr_contigs = {}
  for ctg in contigs:
    gchr = (ctg.genome, ctg.chr)
    gchr_contigs.setdefault(gchr,[]).append(ctg)

  contigs = []
  for gchr in gchr_contigs:
    # sort within chromosome
    gchr_contigs[gchr].sort(key=lambda x: x.start)
    # gchr_contigs[gchr] = sorted(gchr_contigs[gchr], key=lambda ctg: ctg.start)

    ctg_ongoing = gchr_contigs[gchr][0]
    for i in range(1, len(gchr_contigs[gchr])):
      ctg_this = gchr_contigs[gchr][i]
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
def report_divide_stats(fold_contigs):
  """ Report genome-specific statistics about the division of contigs
      between sets."""

  fold_counts_genome = []
  fold_nts_genome = [] 
  for fi in range(len(fold_contigs)):
    fcg, fng = contig_stats_genome(fold_contigs[fi])
    fold_counts_genome.append(fcg)
    fold_nts_genome.append(fng)
  num_genomes = len(fold_counts_genome[0])  

  # sum nt across genomes
  fold_nts = [sum(fng) for fng in fold_nts_genome]
  total_nt = sum(fold_nts)

  # compute total sum nt per genome
  total_nt_genome = []
  print('Total nt')
  for gi in range(num_genomes):
    total_nt_gi = sum([fng[gi] for fng in fold_nts_genome])
    total_nt_genome.append(total_nt_gi)
    print('  Genome%d: %10d nt' % (gi, total_nt_gi))

  # label folds and guess that 3 is train/valid/test
  fold_labels = []
  if len(fold_contigs) == 3:
    fold_labels = ['Train','Valid','Test']
  else:
    fold_labels = ['Fold%d' % fi for fi in range(len(fold_contigs))]

  print('Contigs divided into')
  for fi in range(len(fold_contigs)):
    print(' %s: %5d contigs, %10d nt (%.4f)' % \
         (fold_labels[fi], len(fold_contigs[fi]), fold_nts[fi], fold_nts[fi]/total_nt))
    for gi in range(num_genomes):
      print('  Genome%d: %5d contigs, %10d nt (%.4f)' % \
           (gi, fold_counts_genome[fi][gi], fold_nts_genome[fi][gi], fold_nts_genome[fi][gi]/total_nt_genome[gi]))


################################################################################
def report_divide_stats_v1(train_contigs, valid_contigs, test_contigs):
  """ Report genome-specific statistics about the division of contigs
      between train/valid/test sets."""

  # compute genome-specific stats
  train_count_genome, train_nt_genome = contig_stats_genome(train_contigs)
  valid_count_genome, valid_nt_genome = contig_stats_genome(valid_contigs)
  test_count_genome, test_nt_genome = contig_stats_genome(test_contigs)
  num_genomes = len(train_count_genome)

  # sum nt across genomes
  train_nt = sum(train_nt_genome)
  valid_nt = sum(valid_nt_genome)
  test_nt = sum(test_nt_genome)
  total_nt = train_nt + valid_nt + test_nt

  # compute total sum nt per genome
  total_nt_genome = []
  for gi in range(num_genomes):
    total_nt_gi = train_nt_genome[gi] + valid_nt_genome[gi] + test_nt_genome[gi]
    total_nt_genome.append(total_nt_gi)

  print('Contigs divided into')
  print(' Train: %5d contigs, %10d nt (%.4f)' % \
       (len(train_contigs), train_nt, train_nt/total_nt))
  for gi in range(num_genomes):
    print('  Genome%d: %5d contigs, %10d nt (%.4f)' % \
         (gi, train_count_genome[gi], train_nt_genome[gi], train_nt_genome[gi]/total_nt_genome[gi]))

  print(' Valid: %5d contigs, %10d nt (%.4f)' % \
      (len(valid_contigs), valid_nt, valid_nt/total_nt))
  for gi in range(num_genomes):
    print('  Genome%d: %5d contigs, %10d nt (%.4f)' % \
         (gi, valid_count_genome[gi], valid_nt_genome[gi], valid_nt_genome[gi]/total_nt_genome[gi]))

  print(' Test:  %5d contigs, %10d nt (%.4f)' % \
      (len(test_contigs), test_nt, test_nt/total_nt))
  for gi in range(num_genomes):
    print('  Genome%d: %5d contigs, %10d nt (%.4f)' % \
         (gi, test_count_genome[gi], test_nt_genome[gi], test_nt_genome[gi]/total_nt_genome[gi]))


################################################################################
Contig = collections.namedtuple('Contig', ['genome', 'chr', 'start', 'end'])
ModelSeq = collections.namedtuple('ModelSeq', ['genome', 'chr', 'start', 'end', 'label'])
GraphSeq = collections.namedtuple('GraphSeq', ['genome', 'net', 'chr', 'start', 'end'])

################################################################################
if __name__ == '__main__':
  main()
