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
import os
import shutil
import sys

import numpy as np
import pandas as pd

import slurm

"""
basenji_test_folds.py

Train Basenji model replicates using given parameters and data.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <exp_dir> <params_file> <data_dir> <bed_file>'
  parser = OptionParser(usage)

  # sat options
  sat_options = OptionGroup(parser, 'basenji_sat_bed.py options')
  sat_options.add_option('-d', dest='mut_down',
      default=0, type='int',
      help='Nucleotides downstream of center sequence to mutate [Default: %default]')
  sat_options.add_option('-f', dest='genome_fasta',
      default=None,
      help='Genome FASTA for sequences [Default: %default]')
  sat_options.add_option('-l', dest='mut_len',
      default=0, type='int',
      help='Length of center sequence to mutate [Default: %default]')
  sat_options.add_option('-o', dest='out_dir',
      default='sat_mut', help='Output directory [Default: %default]')
  sat_options.add_option('--plots', dest='plots',
      default=False, action='store_true',
      help='Make heatmap plots [Default: %default]')
  sat_options.add_option('-p', dest='processes',
      default=None, type='int',
      help='Number of processes, passed by multi script')
  sat_options.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Ensemble forward and reverse complement predictions [Default: %default]')
  sat_options.add_option('--shifts', dest='shifts',
      default='0',
      help='Ensemble prediction shifts [Default: %default]')
  sat_options.add_option('--stats', dest='sad_stats',
      default='sum',
      help='Comma-separated list of stats to save. [Default: %default]')
  sat_options.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')
  sat_options.add_option('-u', dest='mut_up',
      default=0, type='int',
      help='Nucleotides upstream of center sequence to mutate [Default: %default]')
  parser.add_option_group(sat_options)

  fold_options = OptionGroup(parser, 'cross-fold options')
  fold_options.add_option('-a', '--alt', dest='alternative',
      default='two-sided', help='Statistical test alternative [Default: %default]')
  fold_options.add_option('-c', dest='crosses',
      default=1, type='int',
      help='Number of cross-fold rounds [Default:%default]')
  fold_options.add_option('-e', dest='conda_env',
      default='tf2.6',
      help='Anaconda environment [Default: %default]')
  fold_options.add_option('--name', dest='name',
      default='sat', help='SLURM name prefix [Default: %default]')
  fold_options.add_option('-q', dest='queue',
      default='geforce',
      help='SLURM queue on which to run the jobs [Default: %default]')
  parser.add_option_group(fold_options)

  (options, args) = parser.parse_args()

  if len(args) != 4:
    parser.error('Must provide parameters file and data directory')
  else:
    exp_dir = args[0]
    params_file = args[1]
    data_dir = args[2]
    bed_file = args[3]

   # read data parameters
  data_stats_file = '%s/statistics.json' % data_dir
  with open(data_stats_file) as data_stats_open:
    data_stats = json.load(data_stats_open)

  # count folds
  num_folds = len([dkey for dkey in data_stats if dkey.startswith('fold')])

  ################################################################
  # sat
  ################################################################
  jobs = []

  for ci in range(options.crosses):
    for fi in range(num_folds):
      it_dir = '%s/f%d_c%d' % (exp_dir, fi, ci)

      # update output directory
      sat_dir = '%s/%s' % (it_dir, options.out_dir)

      # check if done
      scores_file = '%s/scores.h5' % sat_dir
      if os.path.isfile(scores_file):
        print('%s already generated.' % scores_file)
      else:
        basenji_cmd = '. /home/drk/anaconda3/etc/profile.d/conda.sh;'
        basenji_cmd += ' conda activate %s;' % options.conda_env
        basenji_cmd += ' echo $HOSTNAME;'

        basenji_cmd += ' basenji_sat_bed.py'
        basenji_cmd += ' %s' % options_string(options, sat_options, sat_dir)
        basenji_cmd += ' %s' % params_file
        basenji_cmd += ' %s/train/model_best.h5' % it_dir
        basenji_cmd += ' %s' % bed_file
        
        name = '%s-f%dc%d' % (options.name, fi, ci)
        basenji_job = slurm.Job(basenji_cmd,
                        name=name,
                        out_file='%s.out'%sat_dir,
                        err_file='%s.err'%sat_dir,
                        cpu=2,
                        gpu=1,
                        queue=options.queue,
                        mem=30000,
                        time='7-0:00:00')
        jobs.append(basenji_job)
        
  slurm.multi_run(jobs, verbose=True)


def options_string(options, train_options, rep_dir):
  options_str = ''

  for opt in train_options.option_list:
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
