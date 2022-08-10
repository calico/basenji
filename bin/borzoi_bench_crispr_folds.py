#!/usr/bin/env python
# Copyright 2019 Calico LLC

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
from optparse import OptionParser, OptionGroup
import glob
import h5py
import pdb
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from borzoi_bench_crispr import accuracy_stats

"""
borzoi_bench_crispr_folds.py

Benchmark Borzoi model replicates on CRISPR enhancer scoring task.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <data_dir>'
  parser = OptionParser(usage)

  # crispr
  crispr_options = OptionGroup(parser, 'basenji_bench_crispr.py options')
  crispr_options.add_option('-c', dest='crispr_dir',
      default='/home/drk/seqnn/data/crispr')
  crispr_options.add_option('-f', dest='genome_fasta',
      default='%s/data/hg38.fa' % os.environ['BASENJIDIR'],
      help='Genome FASTA for sequences [Default: %default]')
  crispr_options.add_option('-o', dest='out_dir',
      default='crispr', help='Output directory [Default: %default]')
  crispr_options.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Ensemble forward and reverse complement predictions [Default: %default]')
  crispr_options.add_option('--shifts', dest='shifts',
      default='0', type='str',
      help='Ensemble prediction shifts [Default: %default]')
  crispr_options.add_option('--span', dest='span',
      default=False, action='store_true',
      help='Aggregate entire gene span [Default: %default]')
  crispr_options.add_option('--sum', dest='sum_targets',
      default=False, action='store_true',
      help='Sum targets for single output [Default: %default]')
  crispr_options.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')
  parser.add_option_group(crispr_options)

  # cross-fold
  fold_options = OptionGroup(parser, 'cross-fold options')
  fold_options.add_option('--crosses', dest='crosses',
      default=1, type='int',
      help='Number of cross-fold rounds [Default:%default]')
  fold_options.add_option('-d', dest='data_head',
      default=None, type='int',
      help='Index for dataset/head [Default: %default]')
  # fold_options.add_option('-r', dest='restart',
  #     default=False, action='store_true',
  #     help='Restart a partially completed job [Default: %default]')
  parser.add_option_group(fold_options)

  (options, args) = parser.parse_args()

  if len(args) != 2:
    parser.error('Must provide parameters file and cross-fold directory')
  else:
    params_file = args[0]
    exp_dir = args[1]

  #######################################################
  # prep work

  # count folds
  num_folds = 0
  fold0_dir = '%s/f%dc0' % (exp_dir, num_folds)
  model_file = '%s/train/model_best.h5' % fold0_dir
  if options.data_head is not None:
    model_file = '%s/train/model%d_best.h5' % (fold0_dir, options.data_head)
  while os.path.isfile(model_file):
    num_folds += 1
    fold0_dir = '%s/f%dc0' % (exp_dir, num_folds)
    model_file = '%s/train/model_best.h5' % fold0_dir
    if options.data_head is not None:
      model_file = '%s/train/model%d_best.h5' % (fold0_dir, options.data_head)
  print('Found %d folds' % num_folds)
  if num_folds == 0:
    exit(1)

  ################################################################
  # satg

  for ci in range(options.crosses):
    for fi in range(num_folds):
      it_dir = '%s/f%dc%d' % (exp_dir, fi, ci)

      # update output directory
      it_crispr_dir = '%s/%s' % (it_dir, options.out_dir)
      os.makedirs(it_crispr_dir, exist_ok=True)

      # choose model
      model_file = '%s/train/model_best.h5' % it_dir
      if options.data_head is not None:
        model_file = '%s/train/model%d_best.h5' % (it_dir, options.data_head)

      cmd = 'borzoi_bench_crispr.py %s %s' % (params_file, model_file)
      cmd += ' %s' % options_string(options, crispr_options, it_crispr_dir)
      print(cmd)
      subprocess.call(cmd, shell=True)


  ################################################################
  # ensemble / metrics
  
  ensemble_dir = '%s/ensemble' % exp_dir
  if not os.path.isdir(ensemble_dir):
    os.mkdir(ensemble_dir)

  crispr_dir = '%s/%s' % (ensemble_dir, options.out_dir)
  if not os.path.isdir(crispr_dir):
    os.mkdir(crispr_dir)

  for crispr_table_tsv in glob.glob('%s/crispr_table.tsv' % options.crispr_dir):
    job_base = os.path.split(crispr_table_tsv)[-2]

    # collect site scores
    site_scores = []
    for ci in range(options.crosses):
      for fi in range(num_folds):
        it_dir = '%s/f%dc%d' % (exp_dir, fi, ci)
        it_crispr_dir = '%s/%s' % (it_dir, options.out_dir)
        score_file = '%s/%s/site_scores.npy' % (it_crispr_dir, job_base)
        site_scores.append(np.load(score_file))

    # ensemble
    ens_dir = '%s/%s' % (crispr_dir, job_base)
    os.makedirs(ens_dir, exist_ok=True)
    site_scores = np.array(site_scores).mean(axis=0, dtype='float32')

    # ensemble
    ens_score_file = '%s/site_scores.npy' % ens_dir
    np.save(ens_score_file, site_scores)

    # score sites
    crispr_df = pd.read_csv(crispr_table_tsv, sep='\t')
    crispr_df['score'] = site_scores

    # compute stats
    accuracy_stats(crispr_df, ens_dir)


def ensemble_scores_h5(ensemble_h5_file, scores_files):
  # open ensemble
  ensemble_h5 = h5py.File(ensemble_h5_file, 'w')

  # transfer base
  base_keys = ['seqs','gene','chr','start','end','strand']
  sad_stats = []
  sad_shapes = []
  scores0_h5 = h5py.File(scores_files[0], 'r')
  for key in scores0_h5.keys():
    if key in base_keys:
      ensemble_h5.create_dataset(key, data=scores0_h5[key])
    else:
      sad_stats.append(key)
      sad_shapes.append(scores0_h5[key].shape)
  scores0_h5.close()

  # average stats
  num_folds = len(scores_files)
  for si, sad_stat in enumerate(sad_stats):
    # initialize ensemble array
    sad_values = np.zeros(shape=sad_shapes[si], dtype='float32')

    # read and add folds
    for scores_file in scores_files:
      with h5py.File(scores_file, 'r') as scores_h5:
        sad_values += scores_h5[sad_stat][:].astype('float32')
    
    # normalize and downcast
    sad_values /= num_folds
    sad_values = sad_values.astype('float16')

    # save
    ensemble_h5.create_dataset(sad_stat, data=sad_values)

  ensemble_h5.close()


def options_string(options, group_options, rep_dir):
  options_str = ''

  for opt in group_options.option_list:
    opt_str = opt.get_opt_string()
    opt_value = options.__dict__[opt.dest]

    # wrap askeriks in ""
    if type(opt_value) == str and opt_value.find('*') != -1:
      opt_value = '"%s"' % opt_value

    # no value for bools
    elif type(opt_value) == bool:
      if not opt_value:
        opt_str = ''
      opt_value = ''

    # skip Nones
    elif opt_value is None:
      opt_str = ''
      opt_value = ''

    # modify
    elif opt.dest == 'out_dir':
      opt_value = rep_dir

    options_str += ' %s %s' % (opt_str, opt_value)

  return options_str


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
