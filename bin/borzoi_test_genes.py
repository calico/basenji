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
import gc
import json
import pdb
import os
import time

from intervaltree import IntervalTree
import numpy as np
import pandas as pd
import pybedtools
import pyranges as pr
from qnorm import quantile_normalize
from scipy.stats import pearsonr
from sklearn.metrics import explained_variance_score

import pygene
from basenji import dataset
from basenji import seqnn
from basenji_sad import untransform_preds1
from borzoi_sed import targets_prep_strand

'''
borzoi_test_genes.py

Measure accuracy at gene-level.
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
      default='testg_out',
      help='Output directory for predictions [Default: %default]')
  parser.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Average the fwd and rc predictions [Default: %default]')
  parser.add_option('--shifts', dest='shifts',
      default='0',
      help='Ensemble prediction shifts [Default: %default]')
  parser.add_option('--span', dest='span',
      default=False, action='store_true',
      help='Aggregate entire gene span [Default: %default]')
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

  # prep strand
  targets_strand_df = targets_prep_strand(targets_df)
  num_targets = targets_df.shape[0]
  num_targets_strand = targets_strand_df.shape[0]

  # read model parameters
  with open(params_file) as params_open:
    params = json.load(params_open)
  params_model = params['model']
  params_train = params['train']

  # set strand pairs (using new indexing)
  orig_new_index = dict(zip(targets_df.index, np.arange(targets_df.shape[0])))
  targets_strand_pair = np.array([orig_new_index[ti] for ti in targets_df.strand_pair])
  params_model['strand_pair'] = [targets_strand_pair]
  
  # construct eval data
  eval_data = dataset.SeqDataset(data_dir,
    split_label=options.split_label,
    batch_size=params_train['batch_size'],
    mode='eval',
    tfr_pattern=options.tfr_pattern)

  # initialize model
  seqnn_model = seqnn.SeqNN(params_model)
  seqnn_model.restore(model_file, options.head_i)
  seqnn_model.build_slice(targets_df.index)
  seqnn_model.build_ensemble(options.rc, options.shifts)
  
  #######################################################
  # sequence intervals

  # read data parameters
  with open('%s/statistics.json'%data_dir) as data_open:
    data_stats = json.load(data_open)
    crop_bp = data_stats['crop_bp']
    pool_width = data_stats['pool_width']

  # read sequence positions
  seqs_df = pd.read_csv('%s/sequences.bed'%data_dir, sep='\t',
    names=['Chromosome','Start','End','Name'])
  seqs_df = seqs_df[seqs_df.Name == options.split_label]
  seqs_pr = pr.PyRanges(seqs_df)

  #######################################################
  # make gene BED

  t0 = time.time()
  print('Making gene BED...', end='')
  genes_bed_file = '%s/genes.bed' % options.out_dir
  if options.span:
    make_genes_span(genes_bed_file, genes_gtf_file, options.out_dir)
  else:
    make_genes_exon(genes_bed_file, genes_gtf_file, options.out_dir)

  genes_pr = pr.read_bed(genes_bed_file)
  print('DONE in %ds' % (time.time()-t0))

  # count gene normalization lengths
  gene_lengths = {}
  gene_strand = {}
  for line in open(genes_bed_file):
    a = line.rstrip().split('\t')
    gene_id = a[3]
    gene_seg_len = int(a[2]) - int(a[1])
    gene_lengths[gene_id] = gene_lengths.get(gene_id,0) + gene_seg_len
    gene_strand[gene_id] = a[5]

  #######################################################
  # intersect genes w/ preds, targets

  # intersect seqs, genes
  t0 = time.time()
  print('Intersecting sequences w/ genes...', end='')
  seqs_genes_pr = seqs_pr.join(genes_pr)
  print('DONE in %ds' % (time.time()-t0), flush=True)

  # hash preds/targets by gene_id
  gene_preds_dict = {}
  gene_targets_dict = {}

  si = 0
  for x, y in eval_data.dataset:
    # predict only if gene overlaps
    yh = None
    y = y.numpy()[...,targets_df.index]

    t0 = time.time()
    print('Sequence %d...' % si, end='')
    for bsi in range(x.shape[0]):
      seq = seqs_df.iloc[si+bsi]

      cseqs_genes_df = seqs_genes_pr[seq.Chromosome].df
      if cseqs_genes_df.shape[0] == 0:
        # empty. no genes on this chromosome
        seq_genes_df = cseqs_genes_df
      else:
        seq_genes_df = cseqs_genes_df[cseqs_genes_df.Start == seq.Start]

      for _, seq_gene in seq_genes_df.iterrows():
        gene_id = seq_gene.Name_b
        gene_start = seq_gene.Start_b
        gene_end = seq_gene.End_b
        seq_start = seq_gene.Start

        # clip boundaries
        gene_seq_start = max(0, gene_start - seq_start)
        gene_seq_end = max(0, gene_end - seq_start)

        # requires >50% overlap
        bin_start = int(np.round(gene_seq_start / pool_width))
        bin_end = int(np.round(gene_seq_end / pool_width))

        # predict
        if yh is None:
          yh = seqnn_model(x)

        # slice gene region
        yhb = yh[bsi,bin_start:bin_end].astype('float16')
        yb = y[bsi,bin_start:bin_end].astype('float16')

        if len(yb) > 0:  
          gene_preds_dict.setdefault(gene_id,[]).append(yhb)
          gene_targets_dict.setdefault(gene_id,[]).append(yb)
    
    # advance sequence table index
    si += x.shape[0]
    print('DONE in %ds' % (time.time()-t0), flush=True)
    if si % 128 == 0:
      gc.collect()

  # aggregate gene bin values into arrays
  gene_targets = []
  gene_preds = []
  gene_ids = sorted(gene_targets_dict.keys())
  gene_within = []
  gene_wvar = []

  for gene_id in gene_ids:
    gene_preds_gi = np.concatenate(gene_preds_dict[gene_id], axis=0).astype('float32')
    gene_targets_gi = np.concatenate(gene_targets_dict[gene_id], axis=0).astype('float32')

    # slice strand
    if gene_strand[gene_id] == '+':
      gene_strand_mask = (targets_df.strand != '-').to_numpy()
    else:
      gene_strand_mask = (targets_df.strand != '+').to_numpy()
    gene_preds_gi = gene_preds_gi[:,gene_strand_mask]
    gene_targets_gi = gene_targets_gi[:,gene_strand_mask]

    if gene_targets_gi.shape[0] == 0:
      print(gene_id, gene_targets_gi.shape, gene_preds_gi.shape)

    # untransform
    gene_preds_gi = untransform_preds1(gene_preds_gi, targets_strand_df)
    gene_targets_gi = untransform_preds1(gene_targets_gi, targets_strand_df)

    # compute within gene correlation before dropping length axis
    gene_corr_gi = np.zeros(num_targets_strand)
    for ti in range(num_targets_strand):
      if gene_preds_gi[:,ti].var() > 1e-6 and gene_targets_gi[:,ti].var() > 1e-6:
        preds_log = np.log2(gene_preds_gi[:,ti]+1)
        targets_log = np.log2(gene_targets_gi[:,ti]+1)
        gene_corr_gi[ti] = pearsonr(preds_log, targets_log)[0]
        # gene_corr_gi[ti] = pearsonr(gene_preds_gi[:,ti], gene_targets_gi[:,ti])[0]
      else:
        gene_corr_gi[ti] = np.nan
    gene_within.append(gene_corr_gi)
    gene_wvar.append(gene_targets_gi.var(axis=0))

    # TEMP: save gene preds/targets
    # os.makedirs('%s/gene_within' % options.out_dir, exist_ok=True)
    # np.save('%s/gene_within/%s_preds.npy' % (options.out_dir, gene_id), gene_preds_gi.astype('float16'))
    # np.save('%s/gene_within/%s_targets.npy' % (options.out_dir, gene_id), gene_targets_gi.astype('float16'))

    # mean coverage
    gene_preds_gi = gene_preds_gi.mean(axis=0)
    gene_targets_gi = gene_targets_gi.mean(axis=0)

    # scale by gene length
    gene_preds_gi *= gene_lengths[gene_id]
    gene_targets_gi *= gene_lengths[gene_id]

    gene_preds.append(gene_preds_gi)
    gene_targets.append(gene_targets_gi)

  gene_targets = np.array(gene_targets)
  gene_preds = np.array(gene_preds)
  gene_within = np.array(gene_within)
  gene_wvar = np.array(gene_wvar)

  # log2 transform
  gene_targets = np.log2(gene_targets+1)
  gene_preds = np.log2(gene_preds+1)

  # quantile and mean normalize
  gene_targets_norm = quantile_normalize(gene_targets, ncpus=2)
  gene_targets_norm = gene_targets_norm - gene_targets_norm.mean(axis=-1, keepdims=True)
  gene_preds_norm = quantile_normalize(gene_preds, ncpus=2)
  gene_preds_norm = gene_preds_norm - gene_preds_norm.mean(axis=-1, keepdims=True)

  # save values
  genes_targets_df = pd.DataFrame(gene_targets, index=gene_ids,
                                  columns=targets_strand_df.identifier)
  genes_targets_df.to_csv('%s/gene_targets.tsv' % options.out_dir, sep='\t')
  genes_preds_df = pd.DataFrame(gene_preds, index=gene_ids,
                                columns=targets_strand_df.identifier)
  genes_preds_df.to_csv('%s/gene_preds.tsv' % options.out_dir, sep='\t')
  genes_within_df = pd.DataFrame(gene_within, index=gene_ids,
                                 columns=targets_strand_df.identifier)
  genes_within_df.to_csv('%s/gene_within.tsv' % options.out_dir, sep='\t')
  genes_var_df = pd.DataFrame(gene_wvar, index=gene_ids,
                              columns=targets_strand_df.identifier)
  genes_var_df.to_csv('%s/gene_var.tsv' % options.out_dir, sep='\t')

  #######################################################
  # accuracy stats

  wvar_t = np.percentile(gene_wvar, 80, axis=0)

  acc_pearsonr = []
  acc_r2 = []
  acc_npearsonr = []
  acc_nr2 = []
  acc_wpearsonr = []
  for ti in range(num_targets_strand):
    r_ti = pearsonr(gene_targets[:,ti], gene_preds[:,ti])[0]
    acc_pearsonr.append(r_ti)
    r2_ti = explained_variance_score(gene_targets[:,ti], gene_preds[:,ti])
    acc_r2.append(r2_ti)
    nr_ti = pearsonr(gene_targets_norm[:,ti], gene_preds_norm[:,ti])[0]
    acc_npearsonr.append(nr_ti)
    nr2_ti = explained_variance_score(gene_targets_norm[:,ti], gene_preds_norm[:,ti])
    acc_nr2.append(nr2_ti)
    var_mask = (gene_wvar[:,ti] > wvar_t[ti])
    wr_ti = gene_within[var_mask].mean()
    acc_wpearsonr.append(wr_ti)

  acc_df = pd.DataFrame({
    'identifier': targets_strand_df.identifier,
    'pearsonr': acc_pearsonr,
    'r2': acc_r2,
    'pearsonr_norm': acc_npearsonr,
    'r2_norm': acc_nr2,
    'pearsonr_gene': acc_wpearsonr,
    'description': targets_strand_df.description
    })
  acc_df.to_csv('%s/acc.txt' % options.out_dir, sep='\t')

  print('%d genes' % gene_targets.shape[0])
  print('Overall PearsonR:     %.4f' % np.mean(acc_df.pearsonr))
  print('Overall R2:           %.4f' % np.mean(acc_df.r2))
  print('Normalized PearsonR:  %.4f' % np.mean(acc_df.pearsonr_norm))
  print('Normalized R2:        %.4f' % np.mean(acc_df.r2_norm))
  print('Within-gene PearsonR: %.4f' % np.mean(acc_df.pearsonr_gene))


def genes_aggregate(genes_bed_file, values_bedgraph):
  """Aggregate values across genes.

  Args:
    genes_bed_file (str): BED file of genes.
    values_bedgraph (str): BedGraph file of values.

  Returns:
    gene_values (dict): Dictionary of gene values.
  """
  values_bt = pybedtools.BedTool(values_bedgraph)
  genes_bt = pybedtools.BedTool(genes_bed_file)

  gene_values = {}

  for overlap in genes_bt.intersect(values_bt, wo=True):
    gene_id = overlap[3]
    value = overlap[7]
    gene_values[gene_id] = gene_values.get(gene_id,0) + value

  return gene_values


def make_genes_exon(genes_bed_file: str, genes_gtf_file: str, out_dir: str):
  """Make a BED file with each genes' exons, excluding exons overlapping
    across genes.
  
  Args:
    genes_bed_file (str): Output BED file of genes.
    genes_gtf_file (str): Input GTF file of genes.
    out_dir (str): Output directory for temporary files.
  """
  # read genes
  genes_gtf = pygene.GTF(genes_gtf_file)

  # write gene exons
  agenes_bed_file = '%s/genes_all.bed' % out_dir
  agenes_bed_out = open(agenes_bed_file, 'w')
  for gene_id, gene in genes_gtf.genes.items():
    # collect exons
    gene_intervals = IntervalTree()
    for tx_id, tx in gene.transcripts.items():
      for exon in tx.exons:
        gene_intervals[exon.start-1:exon.end] = True

    # union
    gene_intervals.merge_overlaps()

    # write
    for interval in sorted(gene_intervals):
      cols = [gene.chrom, str(interval.begin), str(interval.end), gene_id, '.', gene.strand]
      print('\t'.join(cols), file=agenes_bed_out)
  agenes_bed_out.close()

  # find overlapping exons
  genes1_bt = pybedtools.BedTool(agenes_bed_file)
  genes2_bt = pybedtools.BedTool(agenes_bed_file)
  overlapping_exons = set()
  for overlap in genes1_bt.intersect(genes2_bt, s=True, wo=True):
    gene1_id = overlap[3]
    gene1_start = int(overlap[1])
    gene1_end = int(overlap[2])
    overlapping_exons.add((gene1_id,gene1_start,gene1_end))
      
    gene2_id = overlap[9]
    gene2_start = int(overlap[7])
    gene2_end = int(overlap[8])
    overlapping_exons.add((gene2_id,gene2_start,gene2_end))

  # filter for nonoverlapping exons
  genes_bed_out = open(genes_bed_file, 'w')
  for line in open(agenes_bed_file):
    a = line.split()
    start = int(a[1])
    end = int(a[2])
    gene_id = a[-1]
    if (gene_id,start,end) not in overlapping_exons:
      print(line, end='', file=genes_bed_out)
  genes_bed_out.close()


def make_genes_span(genes_bed_file: str, genes_gtf_file: str, out_dir: str, stranded: bool=True):
  """Make a BED file with the span of each gene.
  
  Args:
    genes_bed_file (str): Output BED file of genes.
    genes_gtf_file (str): Input GTF file of genes.
    out_dir (str): Output directory for temporary files.
    stranded (bool): Perform stranded intersection.
  """
  # read genes
  genes_gtf = pygene.GTF(genes_gtf_file)

  # write all gene spans
  agenes_bed_file = '%s/genes_all.bed' % out_dir
  agenes_bed_out = open(agenes_bed_file, 'w')
  for gene_id, gene in genes_gtf.genes.items():
    start, end = gene.span()
    cols = [gene.chrom, str(start-1), str(end), gene_id, '.', gene.strand]
    print('\t'.join(cols), file=agenes_bed_out)
  agenes_bed_out.close()

  # find overlapping genes
  genes1_bt = pybedtools.BedTool(agenes_bed_file)
  genes2_bt = pybedtools.BedTool(agenes_bed_file)
  overlapping_genes = set()
  for overlap in genes1_bt.intersect(genes2_bt, s=stranded, wo=True):
    gene1_id = overlap[3]
    gene2_id = overlap[7]
    if gene1_id != gene2_id:
      overlapping_genes.add(gene1_id)
      overlapping_genes.add(gene2_id)

  # filter for nonoverlapping genes
  genes_bed_out = open(genes_bed_file, 'w')
  for line in open(agenes_bed_file):
    gene_id = line.split()[-1]
    if gene_id not in overlapping_genes:
      print(line, end='', file=genes_bed_out)
  genes_bed_out.close()

  
################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
