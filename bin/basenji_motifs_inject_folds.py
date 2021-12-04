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
from optparse import OptionParser, OptionGroup

import glob
import json
import os
import sys

import h5py
import numpy as np
import pandas as pd

import slurm
import util

'''
basenji_predict_bed.py

Predict sequences from a BED file.
'''

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <model_file> <bed_file>'
  parser = OptionParser(usage)

  # inject options
  inject_options = OptionGroup(parser, 'motif inject options')
  inject_options.add_option('--db', dest='database',
        default='cisbp', help='Motif database [Default: %default]')
  inject_options.add_option('-e', dest='pwm_exp',
      default=1, type='float',
      help='Exponentiate the position weight matrix values [Default: %default]')
  inject_options.add_option('-g', dest='genome',
        default='ce11', help='Genome [Default: %default]')
  inject_options.add_option('-o', dest='out_dir',
      default='inject_out',
      help='Output directory [Default: %default]')
  inject_options.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Ensemble forward and reverse complement predictions [Default: %default]')
  inject_options.add_option('-s', dest='offset',
      default=0, type='int',
      help='Position offset to inject motif [Default: %default]')
  inject_options.add_option('--shifts', dest='shifts',
      default='0',
      help='Ensemble prediction shifts [Default: %default]')
  inject_options.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')

  fold_options = OptionGroup(parser, 'cross-fold options')
  fold_options.add_option('--env', dest='conda_env',
      default='tf2.6',
      help='Anaconda environment [Default: %default]')
  fold_options.add_option('--name', dest='name',
      default='inject', help='SLURM name prefix [Default: %default]')
  fold_options.add_option('-q', dest='queue',
      default='gtx1080ti',
      help='SLURM queue on which to run the jobs [Default: %default]')
  parser.add_option_group(fold_options)

  (options, args) = parser.parse_args()

  if len(args) == 3:
    params_file = args[0]
    folds_dir = args[1]
    bed_file = args[2]

  else:
    parser.error('Must provide parameter and model folds directory and BED file')

  if not os.path.isdir(options.out_dir):
    os.mkdir(options.out_dir)

  ################################################################
  # inject

  jobs = []
  scores_files = []

  for fold_dir in glob.glob('%s/f*' % folds_dir):
    fold_name = fold_dir.split('/')[-1]
    job_name = '%s-%s' % (options.name, fold_name)

    # update output directory
    inject_dir = '%s/%s' % (fold_dir, options.out_dir)
  
    # check if done
    scores_file = '%s/scores.h5' % inject_dir
    scores_files.append(scores_file)
    if os.path.isfile(scores_file):
      print('%s already generated.' % scores_file)
    else:
      basenji_cmd = '. /home/drk/anaconda3/etc/profile.d/conda.sh;'
      basenji_cmd += ' conda activate %s;' % options.conda_env
      basenji_cmd += ' echo $HOSTNAME;'

      basenji_cmd += ' basenji_motifs_inject.py'      
      basenji_cmd += ' %s' % options_string(options, inject_options, inject_dir)
      basenji_cmd += ' %s' % params_file
      basenji_cmd += ' %s/train/model_best.h5' % fold_dir
      basenji_cmd += ' %s' % bed_file
      
      basenji_job = slurm.Job(basenji_cmd, job_name,
        out_file='%s.out'%inject_dir,
        err_file='%s.err'%inject_dir,
        cpu=2, gpu=1,
        queue=options.queue,
        mem=30000, time='3-0:00:00')
      jobs.append(basenji_job)
        
  slurm.multi_run(jobs, verbose=True)

  ################################################################
  # ensemble
  
  ensemble_dir = '%s/ensemble' % folds_dir
  if not os.path.isdir(ensemble_dir):
    os.mkdir(ensemble_dir)

  inject_dir = '%s/%s' % (ensemble_dir, options.out_dir)
  if not os.path.isdir(inject_dir):
    os.mkdir(inject_dir)
    
  print('Generating ensemble scores.')
  ensemble_scores_h5(inject_dir, scores_files)


def ensemble_scores_h5(ensemble_dir, scores_files):
  # open ensemble
  ensemble_h5_file = '%s/scores.h5' % ensemble_dir
  if os.path.isfile(ensemble_h5_file):
    os.remove(ensemble_h5_file)
  ensemble_h5 = h5py.File(ensemble_h5_file, 'w')

  # transfer base
  base_keys = ['motif','tf']
  sad_stats = []
  scores0_h5 = h5py.File(scores_files[0], 'r')
  for key in scores0_h5.keys():
    if key in base_keys:
      ensemble_h5.create_dataset(key, data=scores0_h5[key])
    else:
      sad_stats.append(key)
      sad_shape = scores0_h5[key].shape
  scores0_h5.close()

  # average sum stats
  num_folds = len(scores_files)
  for sad_stat in sad_stats:
    # initialize ensemble array
    sad_values = np.zeros(shape=sad_shape, dtype='float32')

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
