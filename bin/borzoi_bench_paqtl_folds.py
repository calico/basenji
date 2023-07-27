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
import json
import pdb
import os
import shutil
import sys

import numpy as np
import pandas as pd

import slurm
#import util

from basenji_test_folds import stat_tests

"""
borzoi_bench_paqtl_folds.py

Benchmark Basenji model replicates on GTEx paQTL classification task.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <data_dir>'
  parser = OptionParser(usage)

  # sed
  sed_options = OptionGroup(parser, 'borzoi_sed_paqtl_cov.py options')
  sed_options.add_option('-f', dest='genome_fasta',
      default='%s/data/hg38.fa' % os.environ['BASENJIDIR'],
      help='Genome FASTA for sequences [Default: %default]')
  sed_options.add_option('-g', dest='genes_gtf',
      default='%s/genes/gencode41/gencode41_basic_nort.gtf' % os.environ['HG38'],
      help='GTF for gene definition [Default %default]')
  sed_options.add_option('--apafile', dest='apa_file',
      default='polyadb_human_v3.csv.gz')
  sed_options.add_option('-o',dest='out_dir',
      default='paqtl',
      help='Output directory for tables and plots [Default: %default]')
  sed_options.add_option('-p', dest='processes',
      default=None, type='int',
      help='Number of processes, passed by multi script')
  sed_options.add_option('--pseudo', dest='cov_pseudo',
      default=50, type='float',
      help='Coverage pseudocount [Default: %default]')
  sed_options.add_option('--cov', dest='cov_min',
      default=100, type='float',
      help='Coverage pseudocount [Default: %default]')
  sed_options.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Average forward and reverse complement predictions [Default: %default]')
  sed_options.add_option('--shifts', dest='shifts',
      default='0', type='str',
      help='Ensemble prediction shifts [Default: %default]')
  sed_options.add_option('--stats', dest='sed_stats',
      default='REF,ALT',
      help='Comma-separated list of stats to save. [Default: %default]')
  sed_options.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')
  parser.add_option_group(sed_options)

  # classify
  class_options = OptionGroup(parser, 'basenji_bench_classify.py options')
  class_options.add_option('--msl', dest='msl',
      default=1, type='int',
      help='Random forest min_samples_leaf [Default: %default]')
  parser.add_option_group(class_options)

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
  fold_options.add_option('--name', dest='name',
      default='paqtl', help='SLURM name prefix [Default: %default]')
  fold_options.add_option('--max_proc', dest='max_proc',
      default=None, type='int',
      help='Maximum concurrent processes [Default: %default]')
  fold_options.add_option('-q', dest='queue',
      default='geforce',
      help='SLURM queue on which to run the jobs [Default: %default]')
  fold_options.add_option('-r', dest='restart',
      default=False, action='store_true',
      help='Restart a partially completed job [Default: %default]')
  fold_options.add_option('--vcf', dest='vcf_dir',
      default='/home/jlinder/seqnn/data/qtl_cat/paqtl_pip90ea')
  parser.add_option_group(fold_options)

  (options, args) = parser.parse_args()

  if len(args) != 2:
    print(len(args))
    print(args)
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

  sed_stats = options.sed_stats.split(',')

  # merge study/tissue variants
  mpos_vcf_file = '%s/pos_merge.vcf' % options.vcf_dir
  mneg_vcf_file = '%s/neg_merge.vcf' % options.vcf_dir

  ################################################################
  # SNP scores

  # command base
  cmd_base = '. /home/drk/anaconda3/etc/profile.d/conda.sh;'
  cmd_base += ' conda activate %s;' % options.conda_env
  cmd_base += ' echo $HOSTNAME;'

  jobs = []

  for ci in range(options.crosses):
    for fi in range(num_folds):
      it_dir = '%s/f%dc%d' % (exp_dir, fi, ci)
      name = '%s-f%dc%d' % (options.name, fi, ci)

      # update output directory
      it_out_dir = '%s/%s' % (it_dir, options.out_dir)
      os.makedirs(it_out_dir, exist_ok=True)

      model_file = '%s/train/model_best.h5' % it_dir
      if options.data_head is not None:
        model_file = '%s/train/model%d_best.h5' % (it_dir, options.data_head)

      cmd_fold = '%s time borzoi_sed_paqtl_cov.py %s %s' % (cmd_base, params_file, model_file)

      # positive job
      job_out_dir = '%s/merge_pos' % it_out_dir
      if not options.restart or not os.path.isfile('%s/sed.h5'%job_out_dir):
        cmd_job = '%s %s' % (cmd_fold, mpos_vcf_file)
        cmd_job += ' %s' % options_string(options, sed_options, job_out_dir)
        j = slurm.Job(cmd_job, '%s_pos' % name,
            '%s.out'%job_out_dir, '%s.err'%job_out_dir,
            queue=options.queue, gpu=1,
            mem=30000, time='7-0:0:0')
        jobs.append(j)

      # negative job
      job_out_dir = '%s/merge_neg' % it_out_dir
      if not options.restart or not os.path.isfile('%s/sed.h5'%job_out_dir):
        cmd_job = '%s %s' % (cmd_fold, mneg_vcf_file)
        cmd_job += ' %s' % options_string(options, sed_options, job_out_dir)
        j = slurm.Job(cmd_job, '%s_neg' % name,
            '%s.out'%job_out_dir, '%s.err'%job_out_dir,
            queue=options.queue, gpu=1,
            mem=30000, time='7-0:0:0')
        jobs.append(j)
        
  slurm.multi_run(jobs, max_proc=options.max_proc, verbose=True,
                  launch_sleep=10, update_sleep=60)

  ################################################################
  # split study/tissue variants

  for ci in range(options.crosses):
    for fi in range(num_folds):
      it_dir = '%s/f%dc%d' % (exp_dir, fi, ci)
      it_out_dir = '%s/%s' % (it_dir, options.out_dir)

      # split positives
      split_sed(it_out_dir, 'pos', options.vcf_dir, sed_stats)

      # split negatives
      split_sed(it_out_dir, 'neg', options.vcf_dir, sed_stats)

  ################################################################
  # ensemble
  
  ensemble_dir = '%s/ensemble' % exp_dir
  if not os.path.isdir(ensemble_dir):
    os.mkdir(ensemble_dir)

  sqtl_dir = '%s/%s' % (ensemble_dir, options.out_dir)
  if not os.path.isdir(sqtl_dir):
    os.mkdir(sqtl_dir)

  for pos_vcf in glob.glob('%s/*_pos.vcf' % options.vcf_dir):
    neg_vcf = pos_vcf.replace('_pos.','_neg.')
    pos_base = os.path.splitext(os.path.split(pos_vcf)[1])[0]
    neg_base = os.path.splitext(os.path.split(neg_vcf)[1])[0]

    # collect SED files
    sed_pos_files = []
    sed_neg_files = []
    for ci in range(options.crosses):
      for fi in range(num_folds):
        it_dir = '%s/f%dc%d' % (exp_dir, fi, ci)
        it_out_dir = '%s/%s' % (it_dir, options.out_dir)
        
        sed_pos_file = '%s/%s/sed.h5' % (it_out_dir, pos_base)
        sed_pos_files.append(sed_pos_file)

        sed_neg_file = '%s/%s/sed.h5' % (it_out_dir, neg_base)
        sed_neg_files.append(sed_neg_file)

    # ensemble
    ens_pos_dir = '%s/%s' % (sqtl_dir, pos_base)
    os.makedirs(ens_pos_dir, exist_ok=True)
    ens_pos_file = '%s/sed.h5' % (ens_pos_dir)
    if not os.path.isfile(ens_pos_file):
      ensemble_sed_h5(ens_pos_file, sed_pos_files, sed_stats)

    ens_neg_dir = '%s/%s' % (sqtl_dir, neg_base)
    os.makedirs(ens_neg_dir, exist_ok=True)
    ens_neg_file = '%s/sed.h5' % (ens_neg_dir)
    if not os.path.isfile(ens_neg_file):
      ensemble_sed_h5(ens_neg_file, sed_neg_files, sed_stats)
  
  ################################################################
  # fit classifiers

  cmd_base = 'basenji_bench_classify.py -i 100 -p 2 -r 44 -s --stat COVR'
  cmd_base += ' --msl %d' % options.msl

  jobs = []
  for ci in range(options.crosses):
    for fi in range(num_folds):
      it_dir = '%s/f%dc%d' % (exp_dir, fi, ci)
      it_out_dir = '%s/%s' % (it_dir, options.out_dir)

      for sqtl_pos_vcf in glob.glob('%s/*_pos.vcf' % options.vcf_dir):
        tissue = os.path.splitext(os.path.split(sqtl_pos_vcf)[1])[0][:-4]
        sed_pos = '%s/%s_pos/sed.h5' % (it_out_dir, tissue)
        sed_neg = '%s/%s_neg/sed.h5' % (it_out_dir, tissue)
        class_out_dir = '%s/%s_class' % (it_out_dir, tissue)

        if not options.restart or not os.path.isfile('%s/stats.txt' % class_out_dir):
          cmd_class = '%s -o %s %s %s' % (cmd_base, class_out_dir, sed_pos, sed_neg)
          j = slurm.Job(cmd_class, tissue,
              '%s.out'%class_out_dir, '%s.err'%class_out_dir,
              queue='standard', cpu=2,
              mem=22000, time='1-0:0:0')
          jobs.append(j)

  # ensemble
  for sqtl_pos_vcf in glob.glob('%s/*_pos.vcf' % options.vcf_dir):
    tissue = os.path.splitext(os.path.split(sqtl_pos_vcf)[1])[0][:-4]
    sed_pos = '%s/%s_pos/sed.h5' % (sqtl_dir, tissue)
    sed_neg = '%s/%s_neg/sed.h5' % (sqtl_dir, tissue)
    class_out_dir = '%s/%s_class' % (sqtl_dir, tissue)

    if not options.restart or not os.path.isfile('%s/stats.txt' % class_out_dir):
      cmd_class = '%s -o %s %s %s' % (cmd_base, class_out_dir, sed_pos, sed_neg)
      j = slurm.Job(cmd_class, tissue,
          '%s.out'%class_out_dir, '%s.err'%class_out_dir,
          queue='standard', cpu=2,
          mem=22000, time='1-0:0:0')
      jobs.append(j)

  slurm.multi_run(jobs, verbose=True)


def complete_h5(h5_file, sed_stats):
  if os.path.isfile(h5_file):
    try:
      with h5py.File(h5_file, 'r') as h5_open:
        for ss in sed_stats:
          sed = h5_open[ss][:]
          if (sed != 0).sum() == 0:
            return False  
        return True
    except:
      return False
  else:
    return False


def ensemble_sed_h5(ensemble_h5_file, scores_files, sed_stats):
  # open ensemble
  ensemble_h5 = h5py.File(ensemble_h5_file, 'w')

  # transfer base
  sed_shapes = {}
  with h5py.File(scores_files[0], 'r') as scores0_h5:
    for key in scores0_h5.keys():
      if key not in sed_stats:
        ensemble_h5.create_dataset(key, data=scores0_h5[key])
      else:
        sed_shapes[key] = scores0_h5[key].shape

  # average stats
  num_folds = len(scores_files)
  for si, sed_stat in enumerate(sed_stats):
    # initialize ensemble array
    sed_values = np.zeros(shape=sed_shapes[sed_stat], dtype='float32')

    # read and add folds
    for scores_file in scores_files:
      with h5py.File(scores_file, 'r') as scores_h5:
        sed_values += scores_h5[sed_stat][:].astype('float32')
    
    # normalize and downcast
    sed_values /= num_folds
    sed_values = sed_values.astype('float16')

    # save
    ensemble_h5.create_dataset(sed_stat, data=sed_values)

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


def split_sed(it_out_dir, posneg, vcf_dir, sed_stats):
  """Split merged VCF predictions in HDF5 into tissue-specific
     predictions in HDF5, aggregating statistics over genes."""

  merge_h5_file = '%s/merge_%s/sed.h5' % (it_out_dir, posneg)
  merge_h5 = h5py.File(merge_h5_file, 'r')

  # hash snp indexes
  snp_i = {}
  for i in range(merge_h5['snp'].shape[0]):
    snp = merge_h5['snp'][i].decode('UTF-8')
    snp_i.setdefault(snp,[]).append(i)

  # for each tissue VCF
  vcf_glob = '%s/*_%s.vcf' % (vcf_dir, posneg)
  for tissue_vcf_file in glob.glob(vcf_glob):
    tissue_label = tissue_vcf_file.split('/')[-1]
    tissue_label = tissue_label.replace('_pos.vcf','')
    tissue_label = tissue_label.replace('_neg.vcf','')

    # initialize HDF5 arrays
    sed_si = []
    sed_snp = []
    sed_chr = []
    sed_pos = []
    sed_ref = []
    sed_alt = []
    sed_scores = {}
    for ss in sed_stats:
      sed_scores[ss] = []

    # fill HDF5 arrays with ordered SNPs
    for line in open(tissue_vcf_file):
      if not line.startswith('#'):
        snp = line.split()[2]
        i0 = snp_i[snp][0]
        sed_si.append(merge_h5['si'][i0])
        sed_snp.append(merge_h5['snp'][i0])
        sed_chr.append(merge_h5['chr'][i0])
        sed_pos.append(merge_h5['pos'][i0])
        sed_ref.append(merge_h5['ref_allele'][i0])
        sed_alt.append(merge_h5['alt_allele'][i0])

        for ss in sed_stats:
          # take max over each gene
          #  (may not be appropriate for all stats!)
          sed_scores_si = np.array([merge_h5[ss][i] for i in snp_i[snp]])
          sed_scores[ss].append(sed_scores_si.max(axis=0))

    # write tissue HDF5
    tissue_dir = '%s/%s_%s' % (it_out_dir, tissue_label, posneg)
    os.makedirs(tissue_dir, exist_ok=True)
    with h5py.File('%s/sed.h5' % tissue_dir, 'w') as tissue_h5:

      # write SNP indexes
      tissue_h5.create_dataset('si',
        data=np.array(sed_si, dtype='uint32'))

      # write genes
      # tissue_h5.create_dataset('gene',
      #   data=np.array(sed_gene, 'S'))

      # write SNPs
      tissue_h5.create_dataset('snp',
        data=np.array(sed_snp, 'S'))

      # write SNP chr
      tissue_h5.create_dataset('chr',
        data=np.array(sed_chr, 'S'))

      # write SNP pos
      tissue_h5.create_dataset('pos',
        data=np.array(sed_pos, dtype='uint32'))

      # write ref allele
      tissue_h5.create_dataset('ref_allele',
        data=np.array(sed_ref, dtype='S'))

      # write alt allele
      tissue_h5.create_dataset('alt_allele',
        data=np.array(sed_alt, dtype='S'))

      # write targets
      tissue_h5.create_dataset('target_ids', data=merge_h5['target_ids'])
      tissue_h5.create_dataset('target_labels', data=merge_h5['target_labels'])

      # write sed stats
      for ss in sed_stats:
        tissue_h5.create_dataset(ss,
          data=np.array(sed_scores[ss], dtype='float16'))

  merge_h5.close()


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
