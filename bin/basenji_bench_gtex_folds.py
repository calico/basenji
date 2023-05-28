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
import json
import pickle
import pdb
import os
import shutil
import sys

import h5py
import numpy as np
import pandas as pd

import slurm

from basenji_sad_multi import collect_h5

"""
basenji_bench_gtex_folds.py

Benchmark Basenji model replicates on GTEx eQTL classification task.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <data_dir>'
  parser = OptionParser(usage)

  # sad options
  sad_options = OptionGroup(parser, 'basenji_sad.py options')
  sad_options.add_option('-f', dest='genome_fasta',
      default='%s/data/hg38.fa' % os.environ['BASENJIDIR'],
      help='Genome FASTA for sequences [Default: %default]')
  sad_options.add_option('-n', dest='norm_file',
      default=None,
      help='Normalize SAD scores')
  sad_options.add_option('-o',dest='out_dir',
      default='gtex',
      help='Output directory for tables and plots [Default: %default]')
  sad_options.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Average forward and reverse complement predictions [Default: %default]')
  sad_options.add_option('--shifts', dest='shifts',
      default='0', type='str',
      help='Ensemble prediction shifts [Default: %default]')
  sad_options.add_option('--stats', dest='sad_stats',
      default='SAD',
      help='Comma-separated list of stats to save. [Default: %default]')
  sad_options.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')
  sad_options.add_option('--ti', dest='track_indexes',
      default=None, type='str',
      help='Comma-separated list of target indexes to output BigWig tracks')
  sad_options.add_option('--threads', dest='threads',
      default=False, action='store_true',
      help='Run CPU math and output in a separate thread [Default: %default]')
  sad_options.add_option('-u', dest='penultimate',
      default=False, action='store_true',
      help='Compute SED in the penultimate layer [Default: %default]')
  parser.add_option_group(sad_options)

  # classify
  class_options = OptionGroup(parser, 'basenji_bench_classify.py options')
  class_options.add_option('--cn', dest='class_name',
      default=None,
      help='Classifier name extension [Default: %default]')  
  class_options.add_option('--ct', dest='class_targets_file',
      default=None,
      help='Targets slice for the classifier stage [Default: %default]')
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
  fold_options.add_option('-g', dest='gtex_vcf_dir',
      default='/home/drk/seqnn/data/gtex_fine/susie_pip90')
  fold_options.add_option('--name', dest='name',
      default='gtex', help='SLURM name prefix [Default: %default]')
  fold_options.add_option('--max_proc', dest='max_proc',
      default=None, type='int',
      help='Maximum concurrent processes [Default: %default]')
  fold_options.add_option('-p', dest='processes',
      default=None, type='int',
      help='Number of processes, passed by multi script. \
            (Unused, but needs to appear as dummy.)')
  fold_options.add_option('-q', dest='queue',
      default='geforce',
      help='SLURM queue on which to run the jobs [Default: %default]')
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

  # extract output subdirectory name
  gtex_out_dir = options.out_dir

  # split SNP stats
  sad_stats = options.sad_stats.split(',')

  # merge study/tissue variants
  mpos_vcf_file = '%s/pos_merge.vcf' % options.gtex_vcf_dir
  mneg_vcf_file = '%s/neg_merge.vcf' % options.gtex_vcf_dir

  ################################################################
  # SAD

  # SAD command base
  cmd_base = '. /home/drk/anaconda3/etc/profile.d/conda.sh;'
  cmd_base += ' conda activate %s;' % options.conda_env
  cmd_base += ' echo $HOSTNAME;'

  jobs = []

  for ci in range(options.crosses):
    for fi in range(num_folds):
      it_dir = '%s/f%dc%d' % (exp_dir, fi, ci)
      name = '%s-f%dc%d' % (options.name, fi, ci)

      # update output directory
      it_out_dir = '%s/%s' % (it_dir, gtex_out_dir)
      os.makedirs(it_out_dir, exist_ok=True)

      # choose model
      model_file = '%s/train/model_best.h5' % it_dir
      if options.data_head is not None:
        model_file = '%s/train/model%d_best.h5' % (it_dir, options.data_head)

      ########################################
      # negative jobs
      
      # pickle options
      options.out_dir = '%s/merge_neg' % it_out_dir
      os.makedirs(options.out_dir, exist_ok=True)
      options_pkl_file = '%s/options.pkl' % options.out_dir
      options_pkl = open(options_pkl_file, 'wb')
      pickle.dump(options, options_pkl)
      options_pkl.close()

      # create base fold command
      cmd_fold = '%s time basenji_sad.py %s %s %s' % (
        cmd_base, options_pkl_file, params_file, model_file)

      for pi in range(options.processes):
        sad_file = '%s/job%d/sad.h5' % (options.out_dir, pi)
        if not complete_h5(sad_file, sad_stats):
          cmd_job = '%s %s %d' % (cmd_fold, mneg_vcf_file, pi)
          j = slurm.Job(cmd_job, '%s_neg%d' % (name,pi),
              '%s/job%d.out' % (options.out_dir,pi),
              '%s/job%d.err' % (options.out_dir,pi),
              '%s/job%d.sb' % (options.out_dir,pi),
              queue=options.queue, gpu=1, cpu=2,
              mem=30000, time='7-0:0:0')
          jobs.append(j)

      ########################################
      # positive jobs
      
      # pickle options
      options.out_dir = '%s/merge_pos' % it_out_dir
      os.makedirs(options.out_dir, exist_ok=True)
      options_pkl_file = '%s/options.pkl' % options.out_dir
      options_pkl = open(options_pkl_file, 'wb')
      pickle.dump(options, options_pkl)
      options_pkl.close()

      # create base fold command
      cmd_fold = '%s time basenji_sad.py %s %s %s' % (
        cmd_base, options_pkl_file, params_file, model_file)

      for pi in range(options.processes):
        sad_file = '%s/job%d/sad.h5' % (options.out_dir, pi)
        if not complete_h5(sad_file, sad_stats):
          cmd_job = '%s %s %d' % (cmd_fold, mpos_vcf_file, pi)
          j = slurm.Job(cmd_job, '%s_pos%d' % (name,pi),
              '%s/job%d.out' % (options.out_dir,pi),
              '%s/job%d.err' % (options.out_dir,pi),
              '%s/job%d.sb' % (options.out_dir,pi),
              queue=options.queue, gpu=1, cpu=2,
              mem=30000, time='7-0:0:0')
          jobs.append(j)

  slurm.multi_run(jobs, max_proc=options.max_proc, verbose=True,
                  launch_sleep=10, update_sleep=60)

  #######################################################
  # collect output

  for ci in range(options.crosses):
    for fi in range(num_folds):
      it_out_dir = '%s/f%dc%d/%s' % (exp_dir, fi, ci, gtex_out_dir)

      # collect negatives
      neg_out_dir = '%s/merge_neg' % it_out_dir
      if not os.path.isfile('%s/sad.h5' % neg_out_dir):
        collect_h5('sad.h5', neg_out_dir, options.processes)

      # collect positives
      pos_out_dir = '%s/merge_pos' % it_out_dir
      if not os.path.isfile('%s/sad.h5' % pos_out_dir):
        collect_h5('sad.h5', pos_out_dir, options.processes)

  ################################################################
  # split study/tissue variants

  for ci in range(options.crosses):
    for fi in range(num_folds):
      it_out_dir = '%s/f%dc%d/%s' % (exp_dir, fi, ci, gtex_out_dir)
      print(it_out_dir)

      # split positives
      split_sad(it_out_dir, 'pos', options.gtex_vcf_dir, sad_stats)

      # split negatives
      split_sad(it_out_dir, 'neg', options.gtex_vcf_dir, sad_stats)

  ################################################################
  # ensemble
  
  ensemble_dir = '%s/ensemble' % exp_dir
  if not os.path.isdir(ensemble_dir):
    os.mkdir(ensemble_dir)

  gtex_dir = '%s/%s' % (ensemble_dir, gtex_out_dir)
  if not os.path.isdir(gtex_dir):
    os.mkdir(gtex_dir)

  for gtex_pos_vcf in glob.glob('%s/*_pos.vcf' % options.gtex_vcf_dir):
    gtex_neg_vcf = gtex_pos_vcf.replace('_pos.','_neg.')
    pos_base = os.path.splitext(os.path.split(gtex_pos_vcf)[1])[0]
    neg_base = os.path.splitext(os.path.split(gtex_neg_vcf)[1])[0]

    # collect SAD files
    sad_pos_files = []
    sad_neg_files = []
    for ci in range(options.crosses):
      for fi in range(num_folds):
        it_dir = '%s/f%dc%d' % (exp_dir, fi, ci)
        it_out_dir = '%s/%s' % (it_dir, gtex_out_dir)
        
        sad_pos_file = '%s/%s/sad.h5' % (it_out_dir, pos_base)
        sad_pos_files.append(sad_pos_file)

        sad_neg_file = '%s/%s/sad.h5' % (it_out_dir, neg_base)
        sad_neg_files.append(sad_neg_file)

    # ensemble
    ens_pos_dir = '%s/%s' % (gtex_dir, pos_base)
    os.makedirs(ens_pos_dir, exist_ok=True)
    ens_pos_file = '%s/sad.h5' % (ens_pos_dir)
    if not os.path.isfile(ens_pos_file):
      ensemble_sad_h5(ens_pos_file, sad_pos_files)

    ens_neg_dir = '%s/%s' % (gtex_dir, neg_base)
    os.makedirs(ens_neg_dir, exist_ok=True)
    ens_neg_file = '%s/sad.h5' % (ens_neg_dir)
    if not os.path.isfile(ens_neg_file):
      ensemble_sad_h5(ens_neg_file, sad_neg_files)


  ################################################################
  # fit classifiers
  ################################################################

  cmd_base = 'basenji_bench_classify.py -i 100 -p 2 -r 44 -s'
  cmd_base += ' --msl %d' % options.msl

  if options.class_targets_file is not None:
    cmd_base += ' -t %s' % options.class_targets_file

  jobs = []
  for ci in range(options.crosses):
    for fi in range(num_folds):
      it_dir = '%s/f%dc%d' % (exp_dir, fi, ci)
      it_out_dir = '%s/%s' % (it_dir, gtex_out_dir)

      for gtex_pos_vcf in glob.glob('%s/*_pos.vcf' % options.gtex_vcf_dir):
        tissue = os.path.splitext(os.path.split(gtex_pos_vcf)[1])[0][:-4]
        sad_pos = '%s/%s_pos/sad.h5' % (it_out_dir, tissue)
        sad_neg = '%s/%s_neg/sad.h5' % (it_out_dir, tissue)
        for sad_stat in sad_stats:
          class_out_dir = '%s/%s_class-%s' % (it_out_dir, tissue, sad_stat)
          if options.class_name is not None:
            class_out_dir += '-%s' % options.class_name
          if not os.path.isfile('%s/stats.txt' % class_out_dir):
            cmd_class = '%s -o %s --stat %s' % (cmd_base, class_out_dir, sad_stat)
            cmd_class += ' %s %s' % (sad_pos, sad_neg)
            j = slurm.Job(cmd_class, tissue,
                '%s.out'%class_out_dir, '%s.err'%class_out_dir,
                queue='standard', cpu=2,
                mem=22000, time='1-0:0:0')
            jobs.append(j)

  # ensemble
  for gtex_pos_vcf in glob.glob('%s/*_pos.vcf' % options.gtex_vcf_dir):
    tissue = os.path.splitext(os.path.split(gtex_pos_vcf)[1])[0][:-4]
    sad_pos = '%s/%s_pos/sad.h5' % (gtex_dir, tissue)
    sad_neg = '%s/%s_neg/sad.h5' % (gtex_dir, tissue)
    for sad_stat in sad_stats:
      class_out_dir = '%s/%s_class-%s' % (gtex_dir, tissue, sad_stat)
      if options.class_name is not None:
        class_out_dir += '-%s' % options.class_name
      if not os.path.isfile('%s/stats.txt' % class_out_dir):
        cmd_class = '%s -o %s --stat %s' % (cmd_base, class_out_dir, sad_stat)
        cmd_class += ' %s %s' % (sad_pos, sad_neg)
        j = slurm.Job(cmd_class, tissue,
            '%s.out'%class_out_dir, '%s.err'%class_out_dir,
            queue='standard', cpu=2,
            mem=22000, time='1-0:0:0')
        jobs.append(j)

  slurm.multi_run(jobs, verbose=True)


def complete_h5(h5_file, sad_stats):
  if os.path.isfile(h5_file):
    try:
      with h5py.File(h5_file, 'r') as h5_open:
        for ss in sad_stats:
          sad = h5_open[ss][:]
          if (sad != 0).sum() == 0:
            return False  
        return True
    except:
      return False
  else:
    return False


def ensemble_sad_h5(ensemble_h5_file, scores_files):
  # open ensemble
  ensemble_h5 = h5py.File(ensemble_h5_file, 'w')

  # transfer base
  base_keys = ['alt_allele','chr','pos','ref_allele','snp','target_ids','target_labels']
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


def split_sad(it_out_dir, posneg, vcf_dir, sad_stats):
  """Split merged VCF predictions in HDF5 into tissue-specific
     predictions in HDF5."""

  merge_h5_file = '%s/merge_%s/sad.h5' % (it_out_dir, posneg)
  merge_h5 = h5py.File(merge_h5_file, 'r')

  # read merged data
  snps = [snp.decode('UTF-8') for snp in merge_h5['snp']]
  merge_scores = {}
  for ss in sad_stats:
    merge_scores[ss] = merge_h5[ss][:]

  # hash snp indexes
  snp_si = dict(zip(snps, np.arange(len(snps))))

  # for each tissue VCF
  vcf_glob = '%s/*_%s.vcf' % (vcf_dir, posneg)
  for tissue_vcf_file in glob.glob(vcf_glob):
    tissue_label = tissue_vcf_file.split('/')[-1]
    tissue_label = tissue_label.replace('_pos.vcf','')
    tissue_label = tissue_label.replace('_neg.vcf','')

    # initialize HDF5 arrays
    sad_snp = []
    sad_chr = []
    sad_pos = []
    sad_ref = []
    sad_alt = []
    sad_scores = {}
    for ss in sad_stats:
      sad_scores[ss] = []

    # fill HDF5 arrays with ordered SNPs
    for line in open(tissue_vcf_file):
      if not line.startswith('#'):
        a = line.split()
        chrm, pos, snp, ref, alt = a[:5]
        sad_snp.append(snp)
        sad_chr.append(chrm)
        sad_pos.append(int(pos))
        sad_ref.append(ref)
        sad_alt.append(alt)

        for ss in sad_stats:
          si = snp_si[snp]
          sad_scores[ss].append(merge_scores[ss][si])

    # write tissue HDF5
    tissue_dir = '%s/%s_%s' % (it_out_dir, tissue_label, posneg)
    os.makedirs(tissue_dir, exist_ok=True)
    with h5py.File('%s/sad.h5' % tissue_dir, 'w') as tissue_h5:
      # write SNPs
      tissue_h5.create_dataset('snp',
        data=np.array(sad_snp, 'S'))

      # write SNP chr
      tissue_h5.create_dataset('chr',
        data=np.array(sad_chr, 'S'))

      # write SNP pos
      tissue_h5.create_dataset('pos',
        data=np.array(sad_pos, dtype='uint32'))

      # write ref allele
      tissue_h5.create_dataset('ref_allele',
        data=np.array(sad_ref, dtype='S'))

      # write alt allele
      tissue_h5.create_dataset('alt_allele',
        data=np.array(sad_alt, dtype='S'))

      # write targets
      tissue_h5.create_dataset('target_ids', data=merge_h5['target_ids'])
      tissue_h5.create_dataset('target_labels', data=merge_h5['target_labels'])

      # write sed stats
      for ss in sad_stats:
        tissue_h5.create_dataset(ss,
          data=np.array(sad_scores[ss], dtype='float16'))

  merge_h5.close()

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
