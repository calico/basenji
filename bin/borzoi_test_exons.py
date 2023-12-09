#!/usr/bin/env python
# Copyright 2022 Calico LLC
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
import pickle
import pdb
import os
import time

import numpy as np
import pandas as pd
import pyranges as pr
from scipy.stats import pearsonr

import pygene
from basenji import dataset
from basenji import seqnn
from basenji_sad import untransform_preds1

'''
borzoi_test_exons.py

Measure accuracy at exon-level, focusing on tissue-specificity.
'''

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <model_file> <data_dir> <exons_gff>'
  parser = OptionParser(usage)
  parser.add_option('-b', dest='exons_bed',
      default=None,
      help='Internal variable exons BED [Default: %default]')
  parser.add_option('-e', dest='exon_end',
       default=2, type='int',
       help='Ignore exons this many exons at the 5p or 3p end [Default: %default]')
  parser.add_option('--head', dest='head_i',
      default=0, type='int',
      help='Parameters head [Default: %default]')
  parser.add_option('-o', dest='out_dir',
      default='teste_out',
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
    exons_gff_file = args[3]

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

  # attach strand
  targets_strand = []
  for ti, identifier in enumerate(targets_df.identifier):
    if targets_df.index[ti] == targets_df.strand_pair.iloc[ti]:
      targets_strand.append('.')
    else:
      targets_strand.append(identifier[-1])
  targets_df['strand'] = targets_strand

  # collapse stranded
  strand_mask = (targets_df.strand != '-')
  targets_strand_df = targets_df[strand_mask]

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

  # seqnn_model.strand_pair = [np.arange(seqnn_model.num_targets())]
  # seqnn_model.strand_pair[0][targets_df.index] = np.array(targets_df.strand_pair)
  # seqnn_model.build_ensemble(options.rc, options.shifts)
  # seqnn_model.build_slice(targets_df.index)
  
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

  exons_pr = pr.read_gtf(exons_gff_file)
  exons_pr = exons_pr[exons_pr.Feature == 'exonic_part']
  exons_pr = exons_pr.drop('Source')
  exons_pr = exons_pr.drop('Score')

  # filter for sufficient length
  exon_lens = np.array(exons_pr.End - exons_pr.Start)
  len_mask = (exon_lens >= eval_data.pool_width)
  exons_pr = exons_pr[len_mask]

  # count gene normalization lengths
  exon_lengths = {}
  gene_strand = {}
  coords_exonid = {}
  for line in open(exons_gff_file):
    a = line.rstrip().split('\t')
    if a[2] == 'exonic_part':
      chrm = a[0]
      start = int(a[3])
      end = int(a[4])
      kv = pygene.gtf_kv(a[-1])
      gene_id = kv['gene_id']
      gene_strand[gene_id] = a[6]
      exon_id = '%s/%s' % (gene_id, kv['exonic_part_number'])
      exon_seg_len = end - start + 1
      exon_lengths[exon_id] = exon_lengths.get(exon_id,0) + exon_seg_len
      coords_exonid[(chrm,start,end)] = exon_id

  #######################################################
  # intersect genes w/ preds, targets

  # intersect seqs, exons
  seqs_exons_pr = seqs_pr.join(exons_pr)

  # hash preds/targets by gene_id/exon
  exon_preds_dict = {}
  exon_targets_dict = {}
  gene_preds_dict = {}
  gene_targets_dict = {}

  si = 0
  for x, y in eval_data.dataset:
    # predict only if gene overlaps
    yh = None
    y = y.numpy()[...,targets_df.index]
    y = untransform_preds1(y, targets_df, unscale=True)

    t0 = time.time()
    print('Sequence %d...' % si, end='')
    for bsi in range(x.shape[0]):
      seq = seqs_df.iloc[si+bsi]

      cseqs_exons_df = seqs_exons_pr[seq.Chromosome].df
      if cseqs_exons_df.shape[0] == 0:
        # empty. no genes on this chromosome
        seq_exons_df = cseqs_exons_df
      else:
        seq_exons_df = cseqs_exons_df[cseqs_exons_df.Start == seq.Start]

      for _, seq_exon in seq_exons_df.iterrows():
        gene_id = seq_exon.gene_id
        exon_id = '%s/%s' % (gene_id, seq_exon.exonic_part_number)
        exon_start = seq_exon.Start_b
        exon_end = seq_exon.End_b
        seq_start = seq_exon.Start

        # clip boundaries
        exon_seq_start = max(0, exon_start - seq_start)
        exon_seq_end = max(0, exon_end - seq_start)

        # requires >50% overlap
        bin_start = int(np.round(exon_seq_start / pool_width))
        bin_end = int(np.round(exon_seq_end / pool_width))

        # predict
        if yh is None:
          yh = seqnn_model(x)
          print(yh.max(), " untransformed to ", end='')
          yh = untransform_preds1(yh, targets_df, unscale=True)
          print(yh.max())

        # slice gene region
        yhb = yh[bsi,bin_start:bin_end].astype('float16')
        yb = y[bsi,bin_start:bin_end].astype('float16')

        if np.isinf(yhb).any() or np.isnan(yhb).any():
          print(f"WARNING: {gene_id} {exon_id} has NaN/Inf values")

        if len(yb) > 0:
          # save
          exon_preds_dict.setdefault(exon_id,[]).append(yhb)
          exon_targets_dict.setdefault(exon_id,[]).append(yb)
          gene_preds_dict.setdefault(gene_id,[]).append(yhb)
          gene_targets_dict.setdefault(gene_id,[]).append(yb)
    
    # advance sequence table index
    si += x.shape[0]
    print('DONE in %ds.' % (time.time()-t0))
    if si % 128 == 0:
      gc.collect()


  #######################################################
  # aggregate gene values

  # candidate for modularization, shared with borzoi_test_genes.py

  gene_ids = sorted(gene_targets_dict.keys())
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

    # mean coverage
    gene_preds_gi = np.mean(gene_preds_gi, axis=0)
    gene_targets_gi = np.mean(gene_targets_gi, axis=0)

    # save back into dict
    gene_preds_dict[gene_id] = gene_preds_gi
    gene_targets_dict[gene_id] = gene_targets_gi

  gc.collect()

  # TEMP
  with open('%s/gene_preds.pkl' % options.out_dir, 'wb') as pkl_out:
    pickle.dump(gene_preds_dict, pkl_out)
  with open('%s/gene_targets.pkl' % options.out_dir, 'wb') as pkl_out:
    pickle.dump(gene_targets_dict, pkl_out)

  # determine pseudocount
  #  (per track pseudocounts contain zeros up to 50th percentile)
  gene_targets_matrix = np.array(list(gene_targets_dict.values()))
  gene_targets_quart = np.percentile(gene_targets_matrix, 25, axis=0)
  pseudocount = np.median(gene_targets_quart)
  print('Gene pseudocount %f' % pseudocount)

  #######################################################
  # filter for internal exons

  internal_exons = set()
  if options.exons_bed is None:
    # hash genes to their exon numbers
    gene_exon_nums = {}
    exon_ids = sorted(exon_targets_dict.keys())
    for ei, exon_id in enumerate(exon_ids):
      gene_id, exon_num = exon_id.split('/')
      exon_num = int(exon_num)
      gene_exon_nums.setdefault(gene_id,[]).append(exon_num)
    
    for gene_id, exon_nums in gene_exon_nums.items():
      exon_nums = sorted(exon_nums)

      exon_buffer_end = len(exon_nums)-options.exon_end+1
      for egi in range(options.exon_end, exon_buffer_end):
        egn = exon_nums[egi]
        exon_num = str(egn).zfill(3)
        exon_id = '%s/%s' % (gene_id, exon_num)
        internal_exons.add(exon_id)

  else:
    # read directly from BED
    for line in open(options.exons_bed):
      a = line.split('\t')
      chrm = a[0]
      start = int(a[1])
      end = int(a[2])
      exon_id = coords_exonid.get((chrm,start,end), None)
      if exon_id in exon_targets_dict:
        internal_exons.add(exon_id)

  #######################################################
  # aggregate exon bins and normalize by gene

  exonr_targets = []
  exonr_preds = []
  exonr_ge = []
  exon_ids = []

  for exon_id in sorted(internal_exons):
    gene_id = exon_id.split('/')[0]
    exon_preds_ei = np.concatenate(exon_preds_dict[exon_id], axis=0).astype('float32')
    exon_targets_ei = np.concatenate(exon_targets_dict[exon_id], axis=0).astype('float32')

    # slice strand
    if gene_strand[gene_id] == '+':
      gene_strand_mask = (targets_df.strand != '-').to_numpy()
    else:
      gene_strand_mask = (targets_df.strand != '+').to_numpy()
    exon_preds_ei = exon_preds_ei[:,gene_strand_mask]
    exon_targets_ei = exon_targets_ei[:,gene_strand_mask]

    if exon_targets_ei.shape[0] == 0:
      print(exon_id, exon_targets_ei.shape, exon_preds_ei.shape)

    # mean coverage
    exon_preds_ei = exon_preds_ei.mean(axis=0)
    exon_targets_ei = exon_targets_ei.mean(axis=0)

    """ v1
    # skip exons expressed less than pseudocount in all samples
    exon_max = exon_targets_ei.max()
    gene_min = gene_targets_dict[gene_id].min()
    if exon_max > 2*pseudocount and gene_min > 2*pseudocount:
      # require exon expressed near gene level
      exon_gene_ratio = exon_targets_ei / (gene_targets_dict[gene_id] + 0.2*pseudocount)
      if exon_gene_ratio.mean() > 0.5:
        # add pseudocount
        exon_preds_ei += pseudocount
        exon_targets_ei += pseudocount

        # normalize by gene
        exon_preds_ei /= (gene_preds_dict[gene_id]+pseudocount)
        exon_targets_ei /= (gene_targets_dict[gene_id]+pseudocount)

        # append to array lists
        exon_preds.append(exon_preds_ei)
        exon_targets.append(exon_targets_ei)
        exon_ids.append(exon_id)
    """
    # normalize by gene
    exonr_preds_ei = exon_preds_ei / (gene_preds_dict[gene_id]+pseudocount)
    exonr_targets_ei = exon_targets_ei / (gene_targets_dict[gene_id]+pseudocount)

    # append to array lists
    exonr_preds.append(exonr_preds_ei)
    exonr_targets.append(exonr_targets_ei)
    exonr_ge.append(gene_preds_dict[gene_id])
    exon_ids.append(exon_id)

  exonr_targets = np.array(exonr_targets)
  exonr_preds = np.array(exonr_preds)
  exonr_ge = np.array(exonr_ge)

  #######################################################
  # normalize by adjacent exons
  """
  # hash exons to their indexes
  exon_index = dict([(exon_ids[i],i) for i in range(len(exon_ids))])

  # hash genes to their exon numbers
  gene_exon_nums = {}
  for ei, exon_id in enumerate(exon_ids):
    gene_id, exon_num = exon_id.split('/')
    exon_num = int(exon_num)
    gene_exon_nums.setdefault(gene_id,[]).append(exon_num)
  gene_ids = sorted(gene_exon_nums.keys())

  # determine pseudocount
  exon_targets_quart = np.percentile(exon_targets, 33, axis=0)
  pseudocount = np.median(exon_targets_quart)
  print('Exon normalization pseudocount %f' % pseudocount)

  # for each gene
  valid_exons = np.zeros(exon_targets.shape[0], dtype='bool')
  for gene_id, exon_nums in gene_exon_nums.items():
    # for central exons
    for egi in range(2, len(exon_nums)-1):
      egn = exon_nums[egi]
      exon_num = str(egn).zfill(3)
      exon_id = '%s/%s' % (gene_id, exon_num)
      ei = exon_index[exon_id]

      legn = exon_nums[egi-1]
      left_exon_num = str(legn).zfill(3)
      left_exon_id = '%s/%s' % (gene_id, left_exon_num)
      lei = exon_index[left_exon_id]
      
      regn = exon_nums[egi+1]
      right_exon_num = str(regn).zfill(3)
      right_exon_id = '%s/%s' % (gene_id, right_exon_num)
      rei = exon_index[right_exon_id]

      # normalize by adjacent exons
      exon_targets_norm = pseudocount + np.maximum(exon_targets[lei], exon_targets[rei])
      exon_targets[ei] /= exon_targets_norm
      exon_preds_norm = pseudocount + np.maximum(exon_preds[lei], exon_preds[rei])
      exon_preds[ei] /= exon_preds_norm
      valid_exons[ei] = True
  """

  #######################################################
  # accuracy stats

  # save values
  exonr_targets_df = pd.DataFrame(exonr_targets,
                                  index=exon_ids,
                                  columns=targets_strand_df.identifier)
  exonr_targets_df.to_csv('%s/exonr_targets.tsv.gz' % options.out_dir, sep='\t')
  exonr_preds_df = pd.DataFrame(exonr_preds,
                                index=exon_ids,
                                columns=targets_strand_df.identifier)
  exonr_preds_df.to_csv('%s/exonr_preds.tsv.gz' % options.out_dir, sep='\t')
  exonr_ge_df = pd.DataFrame(exonr_ge,
                             index=exon_ids,
                             columns=targets_strand_df.identifier)
  exonr_ge_df.to_csv('%s/exonr_ge.tsv.gz' % options.out_dir, sep='\t')

  """ need help choosing thresholds
  # compute exon variances
  exon_var = exon_targets.var(axis=-1)
  exon_var = np.nan_to_num(exon_var)

  # compute exon accuracies across targets
  exon_pearsonr = []
  for ei in range(exon_targets.shape[0]):
    if exon_var[ei] > 0:
      r_ei = pearsonr(exon_targets[ei], exon_preds[ei])[0]
    else:
      r_ei = np.nan
    exon_pearsonr.append(r_ei)
  exon_pearsonr = np.array(exon_pearsonr)

  # save accuracy table
  exon_acc_df = pd.DataFrame({
    'Variance': exon_var,
    'PearsonR': exon_pearsonr},
    index=exon_ids)
  exon_acc_df.to_csv('%s/exon_acc.tsv.gz' % options.out_dir, sep='\t')

  # summarize at variance levels
  print('Top %%\tExons \tPearsonR')
  for var_pct in [1, 2, 5, 10, 20, 50]:
    var_t = np.percentile(exon_var, 100-var_pct)
    var_mask = (exon_var >= var_t)
    print('%5d\t%6d\t%.4f' % (var_pct, var_mask.sum(), exon_pearsonr[var_mask].mean()))
  """


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
