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
borzoi_borzoi_trip_folds.py

Benchmark Basenji model replicates on TRIP prediction task.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <data_dir> <promoter_file> <insertions_file>'
  parser = OptionParser(usage)

  # trip
  trip_options = OptionGroup(parser, 'borzoi_trip.py options')
  trip_options.add_option('-f', dest='genome_fasta',
      default='%s/data/hg38.fa' % os.environ['BASENJIDIR'],
      help='Genome FASTA for sequences [Default: %default]')
  trip_options.add_option('-o',dest='out_dir',
      default='trip',
      help='Output directory for tables and plots [Default: %default]')
  trip_options.add_option('--site', dest='site',
      default=False, action='store_true',
      help='Return the insertion site without the promoter [Default: %default]')
  trip_options.add_option('--reporter', dest='reporter',
      default=False, action='store_true',
      help='Insert the flanking piggyback reporter with the promoter [Default: %default]')
  trip_options.add_option('--reporter_bare', dest='reporter_bare',
      default=False, action='store_true',
      help='Insert the flanking piggyback reporter with the promoter (no terminal repeats) [Default: %default]')
  trip_options.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Average forward and reverse complement predictions [Default: %default]')
  trip_options.add_option('--shifts', dest='shifts',
      default='0', type='str',
      help='Ensemble prediction shifts [Default: %default]')
  trip_options.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')
  parser.add_option_group(trip_options)

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
      default='trip', help='SLURM name prefix [Default: %default]')
  fold_options.add_option('--max_proc', dest='max_proc',
      default=None, type='int',
      help='Maximum concurrent processes [Default: %default]')
  fold_options.add_option('-q', dest='queue',
      default='geforce',
      help='SLURM queue on which to run the jobs [Default: %default]')
  fold_options.add_option('-r', dest='restart',
      default=False, action='store_true',
      help='Restart a partially completed job [Default: %default]')
  parser.add_option_group(fold_options)

  (options, args) = parser.parse_args()

  if len(args) != 4:
    print(len(args))
    print(args)
    parser.error('Must provide parameters file, cross-fold directory, TRIP promoter sequences, and TRIP insertion sites')
  else:
    params_file = args[0]
    exp_dir = args[1]
    promoters_file = args[2]
    insertions_file = args[3]

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
  # TRIP prediction jobs

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

      cmd_fold = '%s time borzoi_trip.py %s %s %s %s' % (cmd_base, params_file, model_file, promoters_file, insertions_file)

      # TRIP job
      job_out_dir = it_out_dir
      if not options.restart or not os.path.isfile('%s/preds.h5'%job_out_dir):
        cmd_job = cmd_fold
        cmd_job += ' %s' % options_string(options, trip_options, job_out_dir)
        j = slurm.Job(cmd_job, name,
            '%s.out'%job_out_dir, '%s.err'%job_out_dir,
            queue=options.queue, gpu=1,
            mem=60000, time='7-0:0:0')
        jobs.append(j)
        
  slurm.multi_run(jobs, max_proc=options.max_proc, verbose=True,
                  launch_sleep=10, update_sleep=60)

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
