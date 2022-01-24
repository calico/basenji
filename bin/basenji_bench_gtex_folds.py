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
import util

from basenji_test_folds import stat_tests

"""
basenji_bench_phylop_folds.py

Benchmark Basenji model replicates on BED PhyloP task.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <exp_dir> <params_file> <data_dir>'
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
  sad_options.add_option('--pseudo', dest='log_pseudo',
      default=1, type='float',
      help='Log2 pseudocount [Default: %default]')
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
  class_options.add_option('--msl', dest='msl',
      default=1, type='int',
      help='Random forest min_samples_leaf [Default: %default]')
  parser.add_option_group(class_options)

  # cross-fold
  fold_options = OptionGroup(parser, 'cross-fold options')
  # fold_options.add_option('-a', '--alt', dest='alternative',
  #     default='two-sided', help='Statistical test alternative [Default: %default]')
  fold_options.add_option('-c', dest='crosses',
      default=1, type='int',
      help='Number of cross-fold rounds [Default:%default]')
  fold_options.add_option('-d', dest='data_head',
      default=None, type='int',
      help='Index for dataset/head [Default: %default]')
  fold_options.add_option('-e', dest='conda_env',
      default='tf2.6',
      help='Anaconda environment [Default: %default]')
  fold_options.add_option('-g', dest='gtex_vcf_dir',
      default='/home/drk/seqnn/data/gtex_fine/susie_pip90')
  # fold_options.add_option('--label_exp', dest='label_exp',
  #     default='Experiment', help='Experiment label [Default: %default]')
  # fold_options.add_option('--label_ref', dest='label_ref',
  #     default='Reference', help='Reference label [Default: %default]')
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
      default='gtx1080ti',
      help='SLURM queue on which to run the jobs [Default: %default]')
  fold_options.add_option('-r', dest='restart',
      default=False, action='store_true',
      help='Restart a partially completed job [Default: %default]')
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
  # SAD

  jobs = []

  for ci in range(options.crosses):
    for fi in range(num_folds):
      it_dir = '%s/f%dc%d' % (exp_dir, fi, ci)
      name = '%s-f%dc%d' % (options.name, fi, ci)

      # update output directory
      it_out_dir = '%s/%s' % (it_dir, options.out_dir)
      os.makedirs(it_out_dir, exist_ok=True)

      # SAD command base
      cmd_base = '. /home/drk/anaconda3/etc/profile.d/conda.sh;'
      cmd_base += ' conda activate %s;' % options.conda_env
      cmd_base += ' echo $HOSTNAME;'

      model_file = '%s/train/model_best.h5' % it_dir
      if options.data_head is not None:
        model_file = '%s/train/model%d_best.h5' % (it_dir, options.data_head)

      cmd_base += ' basenji_sad.py %s %s' % (params_file, model_file)

      for gtex_pos_vcf in glob.glob('%s/*_pos.vcf' % options.gtex_vcf_dir):
        # positive job 
        job_base = os.path.splitext(os.path.split(gtex_pos_vcf)[1])[0]
        sad_out_dir = '%s/%s' % (it_out_dir, job_base)
        if not options.restart or not os.path.isfile('%s/sad.h5'%sad_out_dir):
          cmd_sad = '%s %s' % (cmd_base, gtex_pos_vcf)
          cmd_sad += ' %s' % options_string(options, sad_options, sad_out_dir)
          name = '%s_%s' % (options.name, job_base)
          j = slurm.Job(cmd_sad, name,
              '%s.out'%sad_out_dir, '%s.err'%sad_out_dir,
              queue=options.queue, gpu=1,
              mem=22000, time='1-0:0:0')
          jobs.append(j)

        # negative job 
        gtex_neg_vcf = gtex_pos_vcf.replace('_pos.','_neg.')
        job_base = os.path.splitext(os.path.split(gtex_neg_vcf)[1])[0]
        sad_out_dir = '%s/%s' % (it_out_dir, job_base)
        if not options.restart or not os.path.isfile('%s/sad.h5'%sad_out_dir):
          cmd_sad = '%s %s' % (cmd_base, gtex_neg_vcf)
          cmd_sad += ' %s' % options_string(options, sad_options, sad_out_dir)
          name = '%s_%s' % (options.name, job_base)
          j = slurm.Job(cmd_sad, name,
              '%s.out'%sad_out_dir, '%s.err'%sad_out_dir,
              queue=options.queue, gpu=1,
              mem=22000, time='1-0:0:0')
          jobs.append(j)
        
  slurm.multi_run(jobs, max_proc=options.max_proc, verbose=True,
                  launch_sleep=10, update_sleep=60)

  ################################################################
  # ensemble
  ################################################################
  ensemble_dir = '%s/ensemble' % exp_dir
  if not os.path.isdir(ensemble_dir):
    os.mkdir(ensemble_dir)

  gtex_dir = '%s/%s' % (ensemble_dir, options.out_dir)
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
        it_out_dir = '%s/%s' % (it_dir, options.out_dir)
        
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

  jobs = []
  for ci in range(options.crosses):
    for fi in range(num_folds):
      it_dir = '%s/f%dc%d' % (exp_dir, fi, ci)
      it_out_dir = '%s/%s' % (it_dir, options.out_dir)

      for gtex_pos_vcf in glob.glob('%s/*_pos.vcf' % options.gtex_vcf_dir):
        tissue = os.path.splitext(os.path.split(gtex_pos_vcf)[1])[0][:-4]
        sad_pos = '%s/%s_pos/sad.h5' % (it_out_dir, tissue)
        sad_neg = '%s/%s_neg/sad.h5' % (it_out_dir, tissue)
        class_out_dir = '%s/%s_class' % (it_out_dir, tissue)

        if not options.restart or not os.path.isfile('%s/stats.txt' % class_out_dir):
          cmd_class = '%s -o %s %s %s' % (cmd_base, class_out_dir, sad_pos, sad_neg)
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
    class_out_dir = '%s/%s_class' % (gtex_dir, tissue)

    if not options.restart or not os.path.isfile('%s/stats.txt' % class_out_dir):
      cmd_class = '%s -o %s %s %s' % (cmd_base, class_out_dir, sad_pos, sad_neg)
      j = slurm.Job(cmd_class, tissue,
          '%s.out'%class_out_dir, '%s.err'%class_out_dir,
          queue='standard', cpu=2,
          mem=22000, time='1-0:0:0')
      jobs.append(j)

  slurm.multi_run(jobs, verbose=True)


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


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
