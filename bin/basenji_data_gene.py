#!/usr/bin/env python
from optparse import OptionParser
import json
import os
import pdb
import random
import sys

import networkx as nx
import numpy as np
import pysam
import pandas as pd
import tensorflow as tf

from basenji.dna_io import dna_1hot
import gff

"""
basenji_data_gene.py

Write TF Records for gene TSS with expression measurements.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <fasta> <tss_gff> <expr_file>'
  parser = OptionParser(usage)
  parser.add_option('-c', dest='cluster_gene_distance',
      default=1000, type='int',
      help='Cluster genes into the same split within this distance [Default: %default]')
  parser.add_option('-g', dest='gene_index',
      default='gene_name',
      help='Key to match TSS GFF to expression table [Default: %default]')
  parser.add_option('-l', dest='seq_length',
      default=65536, type='int',
      help='Sequence length [Default: %default]')
  parser.add_option('-o', dest='out_dir',
      default='genes_out')
  parser.add_option('-n', dest='n_allowed_pct',
      default=0.25, type='float',
      help='Proportion of sequence allowed to be Ns on one side [Default: %default]')
  parser.add_option('-r', dest='seqs_per_tfr',
      default=256, type='int',
      help='Sequences per TFRecord file [Default: %default]')
  parser.add_option('-s', dest='sqrt',
      default=False, action='store_true',
      help='Square root the expression values [Default: %default]')
  parser.add_option('-t', dest='test_pct_or_chr',
      default=0.05, type='str',
      help='Proportion of the data for testing [Default: %default]')
  parser.add_option('-v', dest='valid_pct_or_chr',
      default=0.05, type='str',
      help='Proportion of the data for validation [Default: %default]')
  (options, args) = parser.parse_args()

  if len(args) != 3:
    parser.error('')
  else:
    fasta_file = args[0]
    tss_gff_file = args[1]
    expr_file = args[2]

  if os.path.isdir(options.out_dir):
    print('Remove output directory %s.' % options.out_dir)
    exit(1)
  else:
    os.mkdir(options.out_dir)

  ################################################################
  # read genes and targets

  genes_df = gff_df(tss_gff_file, options.gene_index)
  expr_df = pd.read_csv(expr_file, index_col=0)

  # filter for shared genes
  shared_genes = set(genes_df.index) & set(expr_df.index)
  print('Shared %d genes of %d described and %d quantified' %  \
    (len(shared_genes), genes_df.shape[0], expr_df.shape[0]))

  genes_mask = np.array([gene in shared_genes for gene in genes_df.index])
  genes_df = genes_df.loc[genes_mask].copy()
  expr_mask = np.array([gene in shared_genes for gene in expr_df.index])
  expr_df = expr_df.loc[expr_mask].copy()

  if options.sqrt:
    expr_df = np.sqrt(expr_df)

  ################################################################
  # divide between train/valid/test

  # permute genes
  genes_df = genes_df.sample(frac=1)
  
  try:
    # convert to float pct
    valid_pct = float(options.valid_pct_or_chr)
    test_pct = float(options.test_pct_or_chr)
    assert(0 <= valid_pct <= 1)
    assert(0 <= test_pct <= 1)

    # divide by pct
    tvt_indexes = divide_genes_pct(genes_df, test_pct, valid_pct, options.cluster_gene_distance)

  except (ValueError, AssertionError):
    # divide by chr
    valid_chrs = options.valid_pct_or_chr.split(',')
    test_chrs = options.test_pct_or_chr.split(',')
    tvt_indexes = divide_genes_chr(genes_df, test_chrs, valid_chrs)

  # write gene sets
  train_index, valid_index, test_index = tvt_indexes
  genes_df.iloc[train_index].to_csv('%s/genes_train.csv' % options.out_dir, sep='\t')
  genes_df.iloc[valid_index].to_csv('%s/genes_valid.csv' % options.out_dir, sep='\t')
  genes_df.iloc[test_index].to_csv('%s/genes_test.csv' % options.out_dir, sep='\t')

  # write targets
  targets_df = pd.DataFrame({'identifier':expr_df.columns,
                             'description': expr_df.columns})
  targets_df.index.name = 'index'
  targets_df.to_csv('%s/targets.txt' % options.out_dir, sep='\t')

  ################################################################
  # write TFRecords

  tfr_dir = '%s/tfrecords' % options.out_dir 
  os.mkdir(tfr_dir)

  # open FASTA
  fasta_open = pysam.Fastafile(fasta_file)

  # define options
  tf_opts = tf.io.TFRecordOptions('ZLIB')

  tvt_tuples = [('train',train_index), ('valid',valid_index), ('test',test_index)]
  for set_label, set_index in tvt_tuples:
    genes_set_df = genes_df.iloc[set_index].copy()
    expr_set_df = expr_df.iloc[set_index].copy()

    num_set = genes_set_df.shape[0]
    num_set_tfrs = int(np.ceil(num_set / options.seqs_per_tfr))

    si = 0
    for tfr_i in range(num_set_tfrs):
      tfr_file = '%s/%s-%d.tfr' % (tfr_dir, set_label, tfr_i)
      print(tfr_file)
      with tf.io.TFRecordWriter(tfr_file, tf_opts) as writer:
        ti = 0
        while ti < options.seqs_per_tfr and si < num_set:
          gene = genes_set_df.iloc[si]
          seq_chrm = gene.chr
          mid_pos = (gene.start + gene.end) // 2
          seq_start = mid_pos - options.seq_length//2
          seq_end = seq_start + options.seq_length

          # left over
          if seq_start < 0:
            n_requested = -seq_start
            if n_requested/options.seq_length < options.n_allowed_pct:
              seq_dna = ''.join([random.choice('ACGT') for i in range(n_requested)])
              seq_dna += fasta_open.fetch(seq_chrm, 0, seq_end)
              print('Allowing %s with %d left Ns' % (gene.name, n_requested))
            else:
              seq_dna = ''
          else:
            seq_dna = fasta_open.fetch(seq_chrm, seq_start, seq_end)

          # right over
          if len(seq_dna) < options.seq_length:
            if len(seq_dna) > 0:
              n_requested = options.seq_length - len(seq_dna)
            if n_requested/options.seq_length < options.n_allowed_pct:
              seq_dna += ''.join([random.choice('ACGT') for i in range(n_requested)])
              print('Allowing %s with %d right Ns' % (gene.name, n_requested))
            else:
              print('Skipping %s with %d Ns' % (gene.name, n_requested))

          # write
          if len(seq_dna) == options.seq_length:
            if gene.strand == '-':
              seq_dna = rc(seq_dna)

            # one hot code
            seq_1hot = dna_1hot(seq_dna)

            # get targets
            targets = expr_set_df.iloc[si].values
            targets = targets.reshape((1,-1))
            targets = targets.astype('float16')

            # make example
            example = tf.train.Example(features=tf.train.Features(feature={
              'sequence': _bytes_feature(seq_1hot.flatten().tostring()),
              'target': _bytes_feature(targets.flatten().tostring())}))

            # write
            writer.write(example.SerializeToString())

          ti += 1
          si += 1

  fasta_open.close()

  ################################################################
  # stats

  stats_dict = {}
  stats_dict['num_targets'] = targets_df.shape[0]
  stats_dict['train_seqs'] = len(train_index)
  stats_dict['valid_seqs'] = len(valid_index)
  stats_dict['test_seqs'] = len(test_index)
  stats_dict['seq_length'] = options.seq_length
  stats_dict['target_length'] = 1

  with open('%s/statistics.json' % options.out_dir, 'w') as stats_json_out:
    json.dump(stats_dict, stats_json_out, indent=4)


################################################################################
def divide_genes_chr(genes_df, test_chrs, valid_chrs):
  """Divide genes into train/valid/test lists by chromosome."""

  # initialize train/valid/test genes lists
  train_index = []
  valid_index = []
  test_index = []

  # process contigs
  gi = 0
  for gene in genes_df.itertuples(index=False):
    if gene.chr in test_chrs:
      test_index.append(gi)      
    elif gene.chr in valid_chrs:
      valid_index.append(gi)
    else:
      train_index.append(gi)
    gi += 1

  train_n = len(train_index)
  valid_n = len(valid_index)
  test_n = len(test_index)
  genes_n = genes_df.shape[0]

  print('Genes divided into')
  print(' Train: %5d genes, (%.4f)' % \
      (train_n, train_n/genes_n))
  print(' Valid: %5d genes, (%.4f)' % \
      (valid_n, valid_n/genes_n))
  print(' Test:  %5d genes, (%.4f)' % \
      (test_n, test_n/genes_n))

  return train_index, valid_index, test_index


################################################################################
def cluster_genes(genes_df, cluster_gene_distance):
  """Cluster genes within a specific distance into a graph."""

  # add numeric index
  genes_df['index_num'] = np.arange(genes_df.shape[0])

  # initialize undirected graph
  genes_graph = nx.Graph()

  # add all genes
  for gi in range(genes_df.shape[0]):
    genes_graph.add_node(gi)

  # for each chromosome
  for chrm in set(genes_df.chr):

    # sort chromosome genes
    genes_chr_df = genes_df[genes_df.chr==chrm].sort_values('start')

    # maintain list of open genes that may still be within cluster distance
    open_genes = []

    # for each chromosome gene
    for gene in genes_chr_df.itertuples():

      # initialize a list of still open genes
      still_open_genes = [gene]

      for ogene in open_genes:
        if gene.start - ogene.start < cluster_gene_distance:
          # add edge between genes
          genes_graph.add_edge(ogene.index_num, gene.index_num)

          # ogene remains open
          still_open_genes.append(ogene)

      # set new open gene list
      open_genes = still_open_genes

  # remove numeric index
  del genes_df['index_num']

  return genes_graph


################################################################################
def divide_genes_pct(genes_df, test_pct, valid_pct, cluster_gene_distance):
  """Divide genes into train/valid/test lists by percentage."""

  # make gene graph
  genes_graph = cluster_genes(genes_df, cluster_gene_distance)

  # initialize train/valid/test genes lists
  train_index = []
  valid_index = []
  test_index = []

  # process connected componenets
  for genes_cc in nx.connected_components(genes_graph):
    r = random.random()
    if r < test_pct:
      test_index += genes_cc
    elif r < test_pct+valid_pct:
      valid_index += genes_cc
    else:
      train_index += genes_cc

  train_n = len(train_index)
  valid_n = len(valid_index)
  test_n = len(test_index)
  genes_n = genes_df.shape[0]

  print('Genes divided into')
  print(' Train: %5d genes, (%.4f)' % \
      (train_n, train_n/genes_n))
  print(' Valid: %5d genes, (%.4f)' % \
      (valid_n, valid_n/genes_n))
  print(' Test:  %5d genes, (%.4f)' % \
      (test_n, test_n/genes_n))

  return train_index, valid_index, test_index


################################################################################
def gff_df(gff_file, gene_index):
  """Read GFF w/ keys into DataFrame."""

  chrms = []
  starts = []
  ends = []
  strands = []
  gtf_lists = {}
  for line in open(gff_file):
    a = line.split('\t')
    chrms.append(a[0])
    starts.append(int(a[3]))
    ends.append(int(a[3]))
    strands.append(a[5])
    for kv in gff.gtf_kv(a[-1]).items():
      gtf_lists.setdefault(kv[0],[]).append(kv[1])

  df = pd.DataFrame({
    'chr':chrms,
    'start':starts,
    'end':ends,
    'strand':strands
    })

  for k, kl in gtf_lists.items():
    df[k] = kl

  df.set_index(gene_index, inplace=True)

  return df


def rc(seq):
  return seq.translate(str.maketrans("ATCGatcg","TAGCtagc"))[::-1]

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
