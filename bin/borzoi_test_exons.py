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
import pdb
import os
import time

import h5py
from intervaltree import IntervalTree
import numpy as np
import pandas as pd
import pybedtools
from scipy.stats import pearsonr
from sklearn.metrics import explained_variance_score
import tensorflow as tf
from tqdm import tqdm

from basenji import bed
from basenji import dataset
from basenji import seqnn
from basenji import trainer
import pickle

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
    genes_bed_file = args[3]

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
    if targets_df.strand_pair.iloc[ti] == ti:
      targets_strand.append('.')
    else:
      targets_strand.append(identifier[-1])
  targets_df['strand'] = targets_strand

  # collapse stranded
  strand_mask = (targets_df.strand != '-')
  targets_strand_df = targets_df[strand_mask]

  # count targets
  num_targets = targets_df.shape[0]
  num_targets_strand = targets_strand_df.shape[0]

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
  seqnn_model.build_slice(targets_df.index)
  seqnn_model.build_ensemble(options.rc, options.shifts)
  
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

  #######################################################
  # intersect genes w/ preds, targets

  genes_bt = pybedtools.BedTool(genes_bed_file)

  # hash preds/targets by gene_id
  gene_preds_dict = {}
  gene_targets_dict = {}

  si = 0
  for x, y in tqdm(eval_data.dataset):
    # predict only if gene overlaps
    yh = None
    y = y.numpy()

    # assemble sequence bedtool
    seq_bed_lines = []
    for bsi in range(x.shape[0]):
      seq = seqs_df.iloc[si+bsi]
      seq_bed_lines.append('%s %d %d %d' % (seq.chr, seq.start, seq.end, bsi))
    seq_bedt = pybedtools.BedTool('\n'.join(seq_bed_lines), from_string=True)

    for overlap in genes_bt.intersect(seq_bedt, wo=True):
      gene_id = overlap[3]
      gene_start = int(overlap[1])
      gene_end = int(overlap[2])
      seq_start = int(overlap[7])
      bsi = int(overlap[9])

      if yh is None:
        # predict
        yh = seqnn_model.predict(x)
        
      # clip boundaries
      gene_seq_start = max(0, gene_start - seq_start)
      gene_seq_end = max(0, gene_end - seq_start)

      # requires >50% overlap
      bin_start = int(np.round(gene_seq_start / pool_width))
      bin_end = int(np.round(gene_seq_end / pool_width))

      # slice gene region
      yhb = yh[bsi,bin_start:bin_end].astype('float16')
      yb = y[bsi,bin_start:bin_end].astype('float16')
      if len(yb) > 0:  
        gene_preds_dict.setdefault(gene_id,[]).append(yhb)
        gene_targets_dict.setdefault(gene_id,[]).append(yb)
    
    # advance sequence table index
    si += x.shape[0]

    # values_len_mean = np.mean([len(v) for v in gene_preds_dict.values()])
    # print(len(gene_preds_dict), values_len_mean, flush=True)
  
  with open("%s/gene_preds_dict.pickle"%options.out_dir, "wb") as file:
    pickle.dump(gene_preds_dict, file, pickle.HIGHEST_PROTOCOL)

  with open("%s/gene_targets_dict.pickle"%options.out_dir, "wb") as file:
    pickle.dump(gene_targets_dict, file, pickle.HIGHEST_PROTOCOL)


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
