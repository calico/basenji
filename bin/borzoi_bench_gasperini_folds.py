#!/usr/bin/env python
# Copyright 2022 Calico LLC

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
import h5py
import pickle
import pdb
import os
import subprocess

import numpy as np
import pandas as pd

import slurm
from basenji.gene import gtf_kv
from borzoi_bench_crispr import accuracy_stats, complete_h5, score_sites
from borzoi_satg_gene_multi import collect_h5

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
  crispr_options.add_option('-b', dest='bench_dir',
      default='/home/drk/seqnn/data/crispr/gasperini')
  crispr_options.add_option('-f', dest='genome_fasta',
      default='%s/data/hg38.fa' % os.environ['BASENJIDIR'],
      help='Genome FASTA for sequences [Default: %default]')
  crispr_options.add_option('-o', dest='out_dir',
      default='gasperini', help='Output directory [Default: %default]')
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
  fold_options.add_option('-c', dest='crosses',
      default=1, type='int',
      help='Number of cross-fold rounds [Default:%default]')
  fold_options.add_option('-d', dest='data_head',
      default=None, type='int',
      help='Index for dataset/head [Default: %default]')
  fold_options.add_option('-e', dest='conda_env',
      default='tf210',
      help='Anaconda environment [Default: %default]')
  fold_options.add_option('-p', dest='processes',
      default=None, type='int',
      help='Number of processes to parallelize satg. [Default: %default]')
  fold_options.add_option('-q', dest='queue',
      default='standard',
      help='SLURM queue on which to run the jobs [Default: %default]')
  parser.add_option_group(fold_options)

  (options, args) = parser.parse_args()

  if len(args) != 2:
    parser.error('Must provide parameters file and cross-fold directory')
  else:
    params_file = args[0]
    exp_dir = args[1]

  # crispr files
  crispr_gtf_file = '%s/crispr_genes.gtf' % options.bench_dir
  crispr_table_tsv = '%s/crispr_table.tsv' % options.bench_dir

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

  if options.queue == 'standard':
    num_gpu = 0
    num_cpu = 8
  else:
    num_gpu = 1
    num_cpu = 4

  leaf_out_dir = options.out_dir

  ################################################################
  # satg
  
  jobs = []
          
  for ci in range(options.crosses):
    for fi in range(num_folds):
      it_dir = '%s/f%dc%d' % (exp_dir, fi, ci)

      # choose model
      model_file = '%s/train/model_best.h5' % it_dir
      if options.data_head is not None:
        model_file = '%s/train/model%d_best.h5' % (it_dir, options.data_head)

      # update output directory
      it_crispr_dir = '%s/%s' % (it_dir, leaf_out_dir)
      os.makedirs(it_crispr_dir, exist_ok=True)

      if os.path.isfile(f'{it_crispr_dir}/scores.h5'):
        print(f'{it_crispr_dir} scores computed')
      else:
        # pickle options
        options_pkl_file = '%s/options.pkl' % it_crispr_dir
        options_pkl = open(options_pkl_file, 'wb')
        options.out_dir = it_crispr_dir
        pickle.dump(options, options_pkl)
        options_pkl.close()

        # parallelize genes
        for pi in range(options.processes):
          satg_pi_h5f = '%s/job%d/scores.h5' % (it_crispr_dir, pi)
          if not complete_h5(satg_pi_h5f):
            satg_cmd = '. /home/drk/anaconda3/etc/profile.d/conda.sh;'
            satg_cmd += ' conda activate %s;' % options.conda_env

            satg_cmd += ' borzoi_satg_gene.py %s %s %s %s %d' % (
                options_pkl_file, params_file, model_file, crispr_gtf_file, pi)
            name = 'gasp-f%dc%dp%d' % (fi,ci,pi)
            outf = '%s/job%d.out' % (it_crispr_dir, pi)
            errf = '%s/job%d.err' % (it_crispr_dir, pi)
            j = slurm.Job(satg_cmd, name,
                outf, errf,
                queue=options.queue,
                cpu=num_cpu, gpu=num_gpu,
                mem=120000, time='7-0:0:0')
            jobs.append(j)

    slurm.multi_run(jobs, verbose=True, launch_sleep=10, update_sleep=60)
  

  ################################################################
  # aggregate processes / metrics

  for ci in range(options.crosses):
    for fi in range(num_folds):
      it_dir = '%s/f%dc%d' % (exp_dir, fi, ci)
      it_crispr_dir = '%s/%s' % (it_dir, leaf_out_dir)

      if os.path.isfile(f'{it_crispr_dir}/scores.h5'):
        print(f'{it_crispr_dir} scores aggregated')
      else:
        # aggregate processes
        collect_h5(it_crispr_dir, options.processes, 'grads')

      # read sequences and reference scores
      satg_h5f = '%s/scores.h5' % it_crispr_dir
      with h5py.File(satg_h5f, 'r') as satg_h5:
        seq_starts = satg_h5['start'][:]
        
        gene_ids = [gene_id.decode('UTF-8') for gene_id in satg_h5['gene']]
        gene_ids = [trim_dot(gid) for gid in gene_ids]
        geneid_i = dict(zip(gene_ids, np.arange(len(gene_ids))))
        
        num_seqs, seq_len, _, num_targets = satg_h5['grads'].shape
        grads_ref = np.zeros((num_seqs,seq_len), dtype='float16')
        for si in range(num_seqs):
          grads_si = satg_h5['grads'][si].sum(axis=-1)
          seq_1hot = satg_h5['seqs'][si]
          n_mask = (seq_1hot.sum(axis=-1) == 0)
          seq_1hot[n_mask] = np.array([1,0,0,0])
          grads_ref[si] = grads_si[seq_1hot]

      # hash gene names to indexes
      gene_i = {}
      for line in open(crispr_gtf_file):
        a = line.split('\t')
        kv = gtf_kv(a[8])
        gene_id = trim_dot(kv['gene_id'])
        if gene_id in geneid_i:
            gene_i[kv['gene_name']] = geneid_i[gene_id]

      # score sites
      crispr_df = pd.read_csv(crispr_table_tsv, sep='\t')
      crispr_df['score'] = score_sites(crispr_df, gene_i, grads_ref, seq_starts)
      # crispr_df['score'] = score_sites(crispr_df, geneid_i, grads_ref, seq_starts)
      np.save('%s/site_scores.npy' % it_crispr_dir, crispr_df['score'])

      # compute stats
      accuracy_stats(crispr_df, it_crispr_dir)


  ################################################################
  # ensemble / metrics
  
  ensemble_dir = '%s/ensemble' % exp_dir
  os.makedirs(ensemble_dir, exist_ok=True)

  ens_crispr_dir = '%s/%s' % (ensemble_dir, leaf_out_dir)
  os.makedirs(ens_crispr_dir, exist_ok=True)

  # collect site scores
  site_scores = []
  for ci in range(options.crosses):
    for fi in range(num_folds):
      it_crispr_dir = '%s/f%dc%d/%s' % (exp_dir, fi, ci, leaf_out_dir)
      score_file = '%s/site_scores.npy' % it_crispr_dir
      site_scores.append(np.load(score_file))

  # ensemble
  site_scores = np.array(site_scores).mean(axis=0, dtype='float32')

  # ensemble
  ens_score_file = '%s/site_scores.npy' % ens_crispr_dir
  np.save(ens_score_file, site_scores)

  # score sites
  crispr_table_tsv = '%s/crispr_table.tsv' % options.bench_dir
  crispr_df = pd.read_csv(crispr_table_tsv, sep='\t')
  crispr_df['score'] = site_scores

  # compute stats
  accuracy_stats(crispr_df, ens_crispr_dir)


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


def trim_dot(gene_id):
  dot_i = gene_id.rfind('.')
  if dot_i == -1:
    return gene_id
  else:
    return gene_id[:dot_i]
    

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
