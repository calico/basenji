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

from __future__ import print_function
from optparse import OptionParser
import json
import os
import pdb
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import scale
import tensorflow as tf

from basenji import dataset
from basenji import seqnn
from basenji import trainer
from quantile_normalization import quantile_normalize

"""
basenji_train_head.py

Train a new head on top of an existing Basenji model.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <model_file> <data_dir>'
  parser = OptionParser(usage)
  parser.add_option('-a', dest='alpha',
      default=1, type='float',
      help='Ridge alpha [Default: %default]')
  parser.add_option('-g', dest='save_gene_expr',
      default=False, action='store_true',
      help='Save predicted and actual gene expression tables [Default: %default]')
  parser.add_option('-o', dest='out_dir',
      default='head_out',
      help='Output directory for new head training [Default: %default]')
  parser.add_option('-q', dest='quantile_normalize',
      default=False, action='store_true',
      help='Quantile normalize test predictions and targets [Default: %default]')
  parser.add_option('-m', dest='skip_model_save',
      default=False, action='store_true',
      help='Skip model save [Default: %default]')
  parser.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Average the fwd and rc predictions [Default: %default]')
  parser.add_option('--shifts', dest='shifts',
      default='0',
      help='Ensemble prediction shifts [Default: %default]')
  (options, args) = parser.parse_args()

  if len(args) != 3:
    parser.error('Must provide parameters, model, and test data directory')
  else:
    params_file = args[0]
    model_file = args[1]
    data_dir = args[2]

  os.makedirs(options.out_dir, exist_ok=True)

  # parse shifts to integers
  options.shifts = [int(shift) for shift in options.shifts.split(',')]

  #######################################################
  # prepare model

  # read model parameters
  with open(params_file) as params_open:
    params = json.load(params_open)
  params_model = params['model']
  params_train = params['train']
  
  #######################################################
  # predict

  preds_h5_file = '%s/preds.h5' % options.out_dir
  targets_h5_file = '%s/targets.h5' % options.out_dir

  if os.path.isfile(preds_h5_file):
    print('Predictions/targets found.')

  else:
    # initialize model
    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(model_file, trunk=True)
    seqnn_model.build_embed(-1)
    seqnn_model.build_ensemble(options.rc, options.shifts)

    preds_h5 = h5py.File(preds_h5_file, 'w')
    targets_h5 = h5py.File(targets_h5_file, 'w')

    for split_label in ['train','valid','test']:
      # construct data
      split_data = dataset.SeqDataset(data_dir,
        split_label=split_label,
        batch_size=params_train['batch_size'],
        mode='eval')

      # compute predictions
      split_preds = seqnn_model.predict(split_data)
      split_preds = split_preds.squeeze()

      # get targets
      split_targets = split_data.numpy(return_inputs=False)
      split_targets = split_targets.squeeze()

      # save HDF5
      preds_h5.create_dataset(split_label, data=split_preds)
      targets_h5.create_dataset(split_label, data=split_targets)

    preds_h5.close()
    targets_h5.close()


  #######################################################
  # regression

  preds_h5 = h5py.File(preds_h5_file, 'r')
  targets_h5 = h5py.File(targets_h5_file, 'r')

  X_train = preds_h5['train'][:].astype('float32')
  X_valid = preds_h5['valid'][:].astype('float32')
  X = np.concatenate([X_train,X_valid], axis=0)

  y_train = targets_h5['train'][:].astype('float32')
  y_valid = targets_h5['valid'][:].astype('float32')
  y = np.concatenate([y_train,y_valid], axis=0)
  
  # train
  ridge_model = Ridge(alpha=options.alpha)
  # ridge_model = ElasticNet(alpha=options.alpha, l1_ratio=0.5, max_iter=2000)
  ridge_model.fit(X, y)

  # test
  X_test = preds_h5['test'][:].astype('float32')
  y_test = targets_h5['test'][:].astype('float32')
  yh_test = ridge_model.predict(X_test)
  num_genes, num_targets = y_test.shape

  if options.save_gene_expr:
    yh = ridge_model.predict(X)
    save_gene_expr(np.concatenate([yh,yh_test], axis=0),
                   np.concatenate([y,y_test], axis=0),
                   data_dir, options.out_dir)

  # normalize
  if options.quantile_normalize:
    y_test = quantile_normalize(y_test)
    yh_test = quantile_normalize(yh_test)

  # tissue metrics
  acc_out = open('%s/acc_tissue.txt' % options.out_dir, 'w')
  acc_pearsonr = []
  acc_spearmanr = []
  acc_r2 = [] 

  for ti in range(num_targets):
    pr = pearsonr(y_test[:,ti], yh_test[:,ti])[0]
    sr = spearmanr(y_test[:,ti], yh_test[:,ti])[0]
    r2 = explained_variance_score(y_test[:,ti], yh_test[:,ti])
    print('%d\t%.4f\t%.4f\t%.4f' % (ti, pr, sr, r2), file=acc_out)
    acc_pearsonr.append(pr)
    acc_spearmanr.append(sr)
    acc_r2.append(r2)

  print('Tissue, across genes:')
  print('PearsonR: %.4f' % np.mean(acc_pearsonr))
  print('SpearmanR: %.4f' % np.mean(acc_spearmanr))
  print('R2: %.4f' % np.mean(acc_r2))
  print('')
  acc_out.close()

  # gene metrics
  acc_out = open('%s/acc_gene.txt' % options.out_dir, 'w')
  acc_pearsonr = []
  acc_spearmanr = []
  acc_r2 = [] 

  for gi in range(num_genes):
    pr = pearsonr(y_test[gi,:], yh_test[gi,:])[0]
    sr = spearmanr(y_test[gi,:], yh_test[gi,:])[0]
    r2 = explained_variance_score(y_test[gi,:], yh_test[gi,:])
    print('%d\t%.4f\t%.4f\t%.4f' % (gi, pr, sr, r2), file=acc_out)
    acc_pearsonr.append(pr)
    acc_spearmanr.append(sr)
    acc_r2.append(r2)

  print('Gene, across tissues:')
  print('PearsonR: %.4f' % np.mean(acc_pearsonr))
  print('SpearmanR: %.4f' % np.mean(acc_spearmanr))
  print('R2: %.4f' % np.mean(acc_r2))
  print('')
  acc_out.close()

  preds_h5.close()
  targets_h5.close()


  #######################################################
  # manipulate model

  if not options.skip_model_save:
    # initialize model  
    head_model = seqnn.SeqNN(params_model)
    head_model.restore(model_file, trunk=True)

    # set weights
    final_dense = head_model.get_dense_layer(-1)
    head_weights = [ridge_model.coef_.T]
    head_weights += [ridge_model.intercept_]
    final_dense.set_weights(head_weights)

    # save new model
    head_model.save('%s/model_best.h5' % options.out_dir)


def save_gene_expr(yh, y, data_dir, out_dir):
  # collect gene names
  train_genes = []
  valid_genes = []
  test_genes = []
  genes_bed_file = '%s/genes.bed' % data_dir
  for line in open(genes_bed_file):
    a = line.split('\t')
    gene_id, split_label = a[3].split('/')
    if split_label == 'train':
      train_genes.append(gene_id)
    elif split_label == 'valid':
      valid_genes.append(gene_id)
    else:
      test_genes.append(gene_id)

  # collect targets
  targets_df = pd.read_csv('%s/targets.txt'%data_dir, index_col=0, sep='\t')

  # write predictions
  gpreds_df = pd.DataFrame(yh,
    index=train_genes+valid_genes+test_genes,
    columns=targets_df.identifier)
  gpreds_df.to_csv('%s/gene_preds.tsv'%out_dir, sep='\t')

  # write predictions
  gexpr_df = pd.DataFrame(y,
    index=train_genes+valid_genes+test_genes,
    columns=targets_df.identifier)
  gexpr_df.to_csv('%s/gene_expr.tsv'%out_dir, sep='\t')

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
