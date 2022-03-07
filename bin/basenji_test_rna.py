#!/usr/bin/env python
# Copyright 2021 Calico LLC
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

from optparse import OptionParser
import json
import os
import time

import h5py
import numpy as np
import pandas as pd
import pybedtools
from scipy.stats import pearsonr
from sklearn.metrics import explained_variance_score
import tensorflow as tf

from basenji import bed
from basenji import dataset
from basenji import seqnn
from basenji import trainer
import pygene
from quantile_normalization import quantile_normalize

'''
basenji_test_rna.py

Additional test metrics for RNA-seq tracks.
'''

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <model_file> <data_dir> <genes_gtf>'
  parser = OptionParser(usage)
  parser.add_option('--head', dest='head_i',
      default=0, type='int',
      help='Parameters head [Default: %default]')
  parser.add_option('-o', dest='out_dir',
      default='rna_out',
      help='Output directory for predictions [Default: %default]')
  parser.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Average the fwd and rc predictions [Default: %default]')
  parser.add_option('--shifts', dest='shifts',
      default='0',
      help='Ensemble prediction shifts [Default: %default]')
  parser.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')
  parser.add_option('--split', dest='split_label',
      default='test',
      help='Dataset split label for eg TFR pattern [Default: %default]')
  parser.add_option('--tfr', dest='tfr_pattern',
      default=None,
      help='TFR pattern string appended to data_dir/tfrecords for subsetting [Default: %default]')
  (options, args) = parser.parse_args()

  if len(args) != 4:
    parser.error('Must provide parameters, model, data directory, and genes GTF')
  else:
    params_file = args[0]
    model_file = args[1]
    data_dir = args[2]
    genes_gtf_file = args[3]

  if not os.path.isdir(options.out_dir):
    os.mkdir(options.out_dir)

  # parse shifts to integers
  options.shifts = [int(shift) for shift in options.shifts.split(',')]

  #######################################################
  # inputs

  # read targets
  if options.targets_file is None:
    options.targets_file = '%s/targets.txt' % data_dir
  targets_df = pd.read_csv(options.targets_file, index_col=0, sep='\t')

  # slice RNA
  rna_indexes = []
  for ti, desc in enumerate(targets_df.description):
    if desc.startswith('RNA:') or desc.startswith('scRNA:'):
      rna_indexes.append(ti)
  targets_rna_df = targets_df.iloc[rna_indexes]
  num_targets = targets_rna_df.shape[0]

  # read model parameters
  with open(params_file) as params_open:
    params = json.load(params_open)
  params_model = params['model']
  params_train = params['train']
  
  # construct eval data
  eval_data = dataset.SeqDataset(data_dir,
    split_label=options.split_label,
    batch_size=params_train['batch_size'],
    mode='eval',
    tfr_pattern=options.tfr_pattern)

  # initialize model
  seqnn_model = seqnn.SeqNN(params_model)
  seqnn_model.restore(model_file, options.head_i)
  seqnn_model.build_ensemble(options.rc, options.shifts)
  seqnn_model.build_slice(rna_indexes)

  #######################################################
  # predict
  
  # compute predictions
  test_preds_rna = seqnn_model.predict(eval_data).astype('float16')

  # read targets
  test_targets = eval_data.numpy(return_inputs=False)
  test_targets_rna = test_targets[...,rna_indexes]
  
  #######################################################
  # sequence intervals

  # read data parameters
  with open('%s/statistics.json'%data_dir) as data_open:
    data_stats = json.load(data_open)
    crop_bp = data_stats['crop_bp']
    pool_width = data_stats['pool_width']
    target_length = data_stats['target_length']

  # read sequence positions
  seqs_df = pd.read_csv('%s/sequences.bed'%data_dir, sep='\t',
    names=['chr','start','end','split'])
  seqs_df = seqs_df[seqs_df.split == options.split_label]

  # write intervals BED
  intervals_bed_file = '%s/intervals.bed' % options.out_dir
  intervals_bed_out = open(intervals_bed_file, 'w')
  for si, seq in enumerate(seqs_df.itertuples()):
    bin_start = seq.start + crop_bp
    for bi in range(target_length):
      bin_end = bin_start + pool_width
      cols = [seq.chr, str(bin_start), str(bin_end), '%d/%d' % (si,bi)]
      print('\t'.join(cols), file=intervals_bed_out)
      bin_start = bin_end
  intervals_bed_out.close()

  #######################################################
  # genes

  # read genes
  genes_gtf = pygene.GTF(genes_gtf_file)

  # write gene spans
  genes_bed_file = '%s/genes_all.bed' % options.out_dir
  genes_bed_out = open(genes_bed_file, 'w')
  gene_lengths = {}
  for gene_id, gene in genes_gtf.genes.items():
    start, end = gene.span()
    gene_lengths[gene_id] = end - start
    cols = [gene.chrom, str(start), str(end), gene_id]
    print('\t'.join(cols), file=genes_bed_out)
  genes_bed_out.close()

  # find overlapping genes
  genes1_bt = pybedtools.BedTool(genes_bed_file)
  genes2_bt = pybedtools.BedTool(genes_bed_file)
  overlapping_genes = set()
  for overlap in genes1_bt.intersect(genes2_bt, wo=True):
    gene1_id = overlap[3]
    gene2_id = overlap[7]
    if gene1_id != gene2_id:
      overlapping_genes.add(gene1_id)
      overlapping_genes.add(gene2_id)

  # filter for nonoverlapping genes
  ugenes_bed_file = '%s/genes_unique.bed' % options.out_dir
  ugenes_bed_out = open(ugenes_bed_file, 'w')
  for line in open(genes_bed_file):
    gene_id = line.split()[-1]
    if gene_id not in overlapping_genes:
      print(line, end='', file=ugenes_bed_out)
  ugenes_bed_out.close()

  #######################################################
  # intersect genes w/ preds, targets

  intervals_bt = pybedtools.BedTool(intervals_bed_file)
  genes_bt = pybedtools.BedTool(ugenes_bed_file)

  # hash preds/targets by gene_id
  gene_preds_dict = {}
  gene_targets_dict = {}

  for overlap in genes_bt.intersect(intervals_bt, wo=True):
    gene_id = overlap[3]
    si, bi = overlap[7].split('/')
    si = int(si)
    bi = int(bi)
    gene_preds_dict.setdefault(gene_id,[]).append(test_preds_rna[si,bi])
    gene_targets_dict.setdefault(gene_id,[]).append(test_targets_rna[si,bi])

  # aggregate gene bin values into arrays
  gene_targets = []
  gene_preds = []

  for gene_id in sorted(gene_targets_dict.keys()):
    gene_preds_gi = np.array(gene_preds_dict[gene_id])
    gene_targets_gi = np.array(gene_targets_dict[gene_id])

    gene_preds_gi = gene_preds_gi.mean(axis=0) * gene_lengths[gene_id]
    gene_targets_gi = gene_targets_gi.mean(axis=0) * gene_lengths[gene_id]

    gene_preds.append(gene_preds_gi)
    gene_targets.append(gene_targets_gi)

  gene_targets = np.array(gene_targets)
  gene_preds = np.array(gene_preds)

  # quantile and mean normalize
  gene_targets_norm = quantile_normalize(gene_targets)
  gene_targets_norm = gene_targets_norm - gene_targets_norm.mean(axis=-1, keepdims=True)
  gene_preds_norm = quantile_normalize(gene_preds)
  gene_preds_norm = gene_preds_norm - gene_preds_norm.mean(axis=-1, keepdims=True)

  #######################################################
  # accuracy stats

  acc_pearsonr = []
  acc_r2 = []
  acc_npearsonr = []
  acc_nr2 = []
  for ti in range(num_targets):
    r_ti = pearsonr(gene_targets[:,ti], gene_preds[:,ti])[0]
    acc_pearsonr.append(r_ti)
    r2_ti = explained_variance_score(gene_targets[:,ti], gene_preds[:,ti])
    acc_r2.append(r2_ti)
    nr_ti = pearsonr(gene_targets_norm[:,ti], gene_preds_norm[:,ti])[0]
    acc_npearsonr.append(nr_ti)
    nr2_ti = explained_variance_score(gene_targets_norm[:,ti], gene_preds_norm[:,ti])
    acc_nr2.append(nr2_ti)

  acc_df = pd.DataFrame({
    'identifier': targets_rna_df.identifier,
    'pearsonr': acc_pearsonr,
    'r2': acc_r2,
    'pearsonr_norm': acc_npearsonr,
    'r2_norm': acc_nr2,
    'description': targets_rna_df.description
    })
  acc_df.to_csv('%s/acc.txt' % options.out_dir, sep='\t')

  print('%d genes' % gene_targets.shape[0])
  print('PearsonR: %.4f' % np.mean(acc_df.pearsonr))
  print('R2:       %.4f' % np.mean(acc_df.r2))
  print('Normalized PearsonR: %.4f' % np.mean(acc_df.pearsonr_norm))
  print('Normalized R2:       %.4f' % np.mean(acc_df.r2_norm))


def genes_aggregate(genes_bed_file, values_bedgraph):
  values_bt = pybedtools.BedTool(values_bedgraph)
  genes_bt = pybedtools.BedTool(genes_bed_file)

  gene_values = {}

  for overlap in genes_bt.intersect(values_bt, wo=True):
    gene_id = overlap[3]
    value = overlap[7]
    gene_values[gene_id] = gene_values.get(gene_id,0) + value

  return gene_values

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
