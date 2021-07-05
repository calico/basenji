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
from collections import OrderedDict
import multiprocessing
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd
import pyBigWig
import pysam

from basenji import dna_io
from basenji import gff
from basenji import gene

"""basenji_hdf5_genes.py

Tile a set of genes and save the result in HDF5 for Basenji processing.

Notes:
 -At the moment, I'm excluding target measurements, but that could be included
  if I want to measure accuracy on specific genes.
"""


################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <fasta_file> <gtf_file> <hdf5_file>'
  parser = OptionParser(usage)
  parser.add_option(
      '-g',
      dest='genome_file',
      default=None,
      help='Chromosome lengths file [Default: %default]')
  parser.add_option(
      '-l',
      dest='seq_length',
      default=1024,
      type='int',
      help='Sequence length [Default: %default]')
  parser.add_option(
      '-c',
      dest='center_t',
      default=0.333,
      type='float',
      help=
      'Center proportion in which TSSs are required to be [Default: %default]')
  parser.add_option(
      '-p',
      dest='processes',
      default=1,
      type='int',
      help='Number parallel processes to load data [Default: %default]')
  parser.add_option(
      '-t',
      dest='target_wigs_file',
      default=None,
      help='Store target values, extracted from this list of WIG files')
  parser.add_option(
      '-w',
      dest='pool_width',
      type='int',
      default=1,
      help='Average pooling width [Default: %default]')
  parser.add_option(
      '--w5',
      dest='w5',
      default=False,
      action='store_true',
      help='Coverage files are w5 rather than BigWig [Default: %default]')
  (options, args) = parser.parse_args()

  if len(args) != 3:
    parser.error('Must provide genes as GTF, genome FASTA, and output HDF5')
  else:
    fasta_file = args[0]
    gtf_file = args[1]
    hdf5_file = args[2]

  if options.target_wigs_file is not None:
    check_wigs(options.target_wigs_file)

  ################################################################
  # organize TSS's by chromosome

  # read transcripts
  transcripts = gff.read_genes(gtf_file, key_id='transcript_id')

  # read transcript --> gene mapping
  transcript_genes = gff.t2g(gtf_file, feature='exon')

  # make gene --> strand mapping
  gene_strand = {}
  for tx_id in transcripts:
    gene_strand[transcript_genes[tx_id]] = transcripts[tx_id].strand

  # cluster TSSs by gene
  gene_tss = cluster_tss(transcript_genes, transcripts, options.pool_width/2)

  # hash TSS's by chromosome
  gene_chrom = {}
  for tx_id in transcripts:
    gene_id = transcript_genes[tx_id]
    gene_chrom[gene_id] = transcripts[tx_id].chrom

  chrom_tss = {}
  for gene_id in gene_tss:
    for tss_pos in gene_tss[gene_id]:
      chrom_tss.setdefault(gene_chrom[gene_id],[]).append((tss_pos,gene_id))

  # sort TSS's by chromosome
  for chrom in chrom_tss:
    chrom_tss[chrom].sort()


  ################################################################
  # determine segments / map transcripts

  # open fasta (to verify chromosome presence)
  fasta = pysam.Fastafile(fasta_file)

  chrom_sizes = OrderedDict()
  for line in open(options.genome_file):
    a = line.split()
    if a[0] in fasta.references:
      chrom_sizes[a[0]] = int(a[1])
    elif a[0] in chrom_tss:
      print('FASTA missing chromosome - %s' % a[0], file=sys.stderr)
      del chrom_tss[a[0]]

  merge_distance = options.center_t * options.seq_length

  seq_coords = []
  tss_list = []

  # ordering by options.genome_file allows for easier
  #  bigwig output in downstream scripts.

  for chrom in chrom_sizes:
    ctss = chrom_tss.get(chrom,[])

    left_i = 0
    while left_i < len(ctss):
      # left TSS
      left_tss = ctss[left_i][0]

      # right TSS
      right_i = left_i
      while right_i+1 < len(ctss) and ctss[right_i+1][0] - left_tss < merge_distance:
        right_i += 1
      right_tss = ctss[right_i][0]

      # determine segment midpoint
      seg_mid = (left_tss + right_tss) // 2

      # extend
      seg_start = seg_mid - options.seq_length//2
      seg_end = seg_start + options.seq_length

      # rescue
      if seg_start < 0 or seg_end >= chrom_sizes[chrom]:
        if chrom_sizes[chrom] == options.seq_length:
          seg_start = 0
          seg_end = options.seq_length
        elif chrom_sizes[chrom] > options.seq_length:
          # also rescuable but not important right now
          pass

      # save segment
      if seg_start >= 0 and seg_end <= chrom_sizes[chrom]:
        seq_coords.append((chrom,seg_start,seg_end))

        # annotate TSS to indexes
        seq_index = len(seq_coords)-1
        for i in range(left_i,right_i+1):
          tss_pos, gene_id = ctss[i]
          tss = gene.TSS('TSS%d'%len(tss_list), gene_id, chrom, tss_pos, seq_index, True, gene_strand[gene_id])
          tss_list.append(tss)

      # update
      left_i = right_i + 1


  ################################################################
  # extract target values

  if options.target_wigs_file:
    t0 = time.time()

    # get wig files and labels
    target_wigs_df = pd.read_table(options.target_wigs_file, index_col=0)
    target_wigs = OrderedDict()
    target_labels = []
    for i in range(target_wigs_df.shape[0]):
      target_wig_series = target_wigs_df.iloc[i]
      target_wigs[target_wig_series.identifier] = target_wig_series.file
      target_labels.append(target_wig_series.description)

    # initialize multiprocessing pool
    pool = multiprocessing.Pool(options.processes)

    # bigwig_read parameters
    bwt_params = [(wig_file, tss_list, seq_coords, options.pool_width)
                  for wig_file in target_wigs.values()]

    # pull the target values in parallel
    if options.w5:
      tss_targets = pool.starmap(wig5_tss_targets, bwt_params)
    else:
      tss_targets = pool.starmap(bigwig_tss_targets, bwt_params)

    # convert to array
    tss_targets = np.transpose(np.array(tss_targets))

  ################################################################
  # extract sequences

  seqs_1hot = []

  for chrom, start, end in seq_coords:
    seq = fasta.fetch(chrom, start, end)
    seqs_1hot.append(dna_io.dna_1hot(seq))

  seqs_1hot = np.array(seqs_1hot)

  fasta.close()

  ################################################################
  # save to HDF5

  # write to HDF5
  hdf5_out = h5py.File(hdf5_file, 'w')

  # store pooling
  hdf5_out.create_dataset('pool_width', data=options.pool_width, dtype='int')

  # store gene sequences
  hdf5_out.create_dataset('seqs_1hot', data=seqs_1hot, dtype='bool')

  # store genesequence coordinates
  seq_chrom = np.array([sc[0] for sc in seq_coords], dtype='S')
  seq_start = np.array([sc[1] for sc in seq_coords])
  seq_end = np.array([sc[2] for sc in seq_coords])

  hdf5_out.create_dataset('seq_chrom', data=seq_chrom)
  hdf5_out.create_dataset('seq_start', data=seq_start)
  hdf5_out.create_dataset('seq_end', data=seq_end)

  # store TSSs
  tss_id = np.array([tss.identifier for tss in tss_list], dtype='S')
  tss_gene = np.array([tss.gene_id for tss in tss_list], dtype='S')
  tss_chrom = np.array([tss.chrom for tss in tss_list], dtype='S')
  tss_pos = np.array([tss.pos for tss in tss_list])
  tss_seq = np.array([tss.gene_seq for tss in tss_list])
  tss_strand = np.array([tss.strand for tss in tss_list], dtype='S')

  hdf5_out.create_dataset('tss_id', data=tss_id)
  hdf5_out.create_dataset('tss_gene', data=tss_gene)
  hdf5_out.create_dataset('tss_chrom', data=tss_chrom)
  hdf5_out.create_dataset('tss_pos', data=tss_pos)
  hdf5_out.create_dataset('tss_seq', data=tss_seq)
  hdf5_out.create_dataset('tss_strand', data=tss_strand)

  # store targets
  if options.target_wigs_file:
    # ids
    target_ids = np.array([tl for tl in target_wigs.keys()], dtype='S')
    hdf5_out.create_dataset('target_ids', data=target_ids)

    # labels
    target_labels = np.array(target_labels, dtype='S')
    hdf5_out.create_dataset('target_labels', data=target_labels)

    # values
    hdf5_out.create_dataset('tss_targets', data=tss_targets, dtype='float16')

  hdf5_out.close()


################################################################################
def bigwig_tss_targets(wig_file, tss_list, seq_coords, pool_width=1):
  ''' Read gene target values from a bigwig
  Args:
    wig_file: Bigwig filename
    tss_list: list of TSS instances
    seq_coords: list of (chrom,start,end) sequence coordinates
    pool_width: average pool adjacent nucleotides of this width
  Returns:
    tss_targets:
  '''

  # initialize target values
  tss_targets = np.zeros(len(tss_list), dtype='float16')

  # open wig
  wig_in = pyBigWig.open(wig_file)

  # warn about missing chromosomes just once
  warned_chroms = set()

  # for each TSS
  for tss_i in range(len(tss_list)):
    tss = tss_list[tss_i]

    # extract sequence coordinates
    seq_chrom, seq_start, seq_end = seq_coords[tss.gene_seq]

    # determine bin coordinates
    tss_bin = (tss.pos - seq_start) // pool_width
    bin_start = seq_start + tss_bin*pool_width
    bin_end = bin_start + pool_width

    # pull values
    try:
      tss_targets[tss_i] = np.array(wig_in.values(seq_chrom, bin_start, bin_end), dtype='float32').sum()

    except RuntimeError:
      if seq_chrom not in warned_chroms:
        print("WARNING: %s doesn't see %s (%s:%d-%d). Setting to all zeros. No additional warnings will be offered for %s" % (wig_file,tss.identifier,seq_chrom,seq_start,seq_end,seq_chrom), file=sys.stderr)
        warned_chroms.add(seq_chrom)

    # check NaN
    if np.isnan(tss_targets[tss_i]):
      print('WARNING: %s (%s:%d-%d) pulled NaN from %s. Setting to zero.' % (tss.identifier, seq_chrom, seq_start, seq_end, wig_file), file=sys.stderr)
      tss_targets[tss_i] = 0

  # close wig file
  wig_in.close()

  return tss_targets


################################################################################
def check_wigs(target_wigs_file):
  target_wigs_df = pd.read_table(target_wigs_file, index_col=0)
  for wig_file in target_wigs_df.file:
    if not os.path.isfile(wig_file):
      print('Cannot find %s' % wig_file, file=sys.stderr)
      exit(1)


################################################################################
def cluster_tss(transcript_genes, transcripts, merge_distance):
  ''' Cluster transcript TSSs and return a dict mapping gene_id
       to a TSS list. '''

  # hash gene_id to all TSSs
  gene_tss_all = {}
  for tx_id in transcript_genes:
    gene_id = transcript_genes[tx_id]
    gene_tss_all.setdefault(gene_id,[]).append(transcripts[tx_id].tss())

  # initialize gene TSS dict
  gene_tss = {}

  # for each gene
  for gene_id in gene_tss_all:
    # initialize TSS cluster summary stats
    cluster_mean = []
    cluster_n = []

    # for each sorted TSS
    for tss_pos in sorted(gene_tss_all[gene_id]):
      # if it's first, add it
      if len(cluster_mean) == 0:
        cluster_mean.append(tss_pos)
        cluster_n.append(1)

      else:
        # if it's close to the previous
        if tss_pos - cluster_mean[-1] < merge_distance:
          # merge
          cluster_mean[-1] = (cluster_mean[-1]*cluster_n[-1] + tss_pos) / (cluster_n[-1]+1)
          cluster_n[-1] += 1

        else:
          # create a new cluster
          cluster_mean.append(tss_pos)
          cluster_n.append(1)

    # map gene_id to TSS cluster means (and correct for GFF to BED index)
    gene_tss[gene_id] = [int(cm) for cm in cluster_mean]

  return gene_tss


################################################################################
def wig5_tss_targets(w5_file, tss_list, seq_coords, pool_width=1):
  ''' Read gene target values from a bigwig
  Args:
    w5_file: wiggle HDF5 filename
    tss_list: list of TSS instances
    seq_coords: list of (chrom,start,end) sequence coordinates
    pool_width: average pool adjacent nucleotides of this width
  Returns:
    tss_targets:
  '''

  # initialize target values
  tss_targets = np.zeros(len(tss_list), dtype='float16')

  # open wig h5
  w5_in = h5py.File(w5_file)

  # warn about missing chromosomes just once
  warned_chroms = set()

  # for each TSS
  for tss_i in range(len(tss_list)):
    tss = tss_list[tss_i]

    # extract sequence coordinates
    seq_chrom, seq_start, seq_end = seq_coords[tss.gene_seq]

    # determine bin coordinates
    tss_bin = (tss.pos - seq_start) // pool_width
    bin_start = seq_start + tss_bin*pool_width
    bin_end = bin_start + pool_width

    # pull values
    try:
      tss_targets[tss_i] = w5_in[seq_chrom][bin_start:bin_end].sum(dtype='float32')

    except RuntimeError:
      if seq_chrom not in warned_chroms:
        print("WARNING: %s doesn't see %s (%s:%d-%d). Setting to all zeros. No additional warnings will be offered for %s" % (w5_file,tss.identifier,seq_chrom,seq_start,seq_end,seq_chrom), file=sys.stderr)
        warned_chroms.add(seq_chrom)

    # check NaN
    if np.isnan(tss_targets[tss_i]):
      print('WARNING: %s (%s:%d-%d) pulled NaN from %s. Setting to zero.' % (tss.identifier, seq_chrom, seq_start, seq_end, w5_file), file=sys.stderr)
      tss_targets[tss_i] = 0

  # close w5 file
  w5_in.close()

  return tss_targets


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
