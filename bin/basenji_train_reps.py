#!/usr/bin/env python
# Copyright 2017 Calico LLC

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
import os
import pdb
import pickle
import shutil
import subprocess
import sys

import slurm

"""
basenji_train_reps.py

Train Basenji model replicates using given parameters and data.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <data_dir>'
  parser = OptionParser(usage)

  # train
  train_options = OptionGroup(parser, 'basenji_train.py options')
  train_options.add_option('-o', dest='out_dir',
      default='train_out',
      help='Output directory for test statistics [Default: %default]')
  train_options.add_option('--restore', dest='restore',
      help='Restore model and continue training [Default: %default]')
  train_options.add_option('--trunk', dest='trunk',
      default=False, action='store_true',
      help='Restore only model trunk [Default: %default]')
  train_options.add_option('--tfr_train', dest='tfr_train_pattern',
      default='train-*.tfr',
      help='Training TFRecord pattern string appended to data_dir [Default: %default]')
  train_options.add_option('--tfr_eval', dest='tfr_eval_pattern',
      default='valid-*.tfr',
      help='Evaluation TFRecord pattern string appended to data_dir [Default: %default]')
  parser.add_option_group(train_options)

  # multi
  rep_options = OptionGroup(parser, 'replication options')
  rep_options.add_option('--name', dest='name',
      default='reps', help='SLURM name prefix [Default: %default]')
  rep_options.add_option('-p', dest='processes',
      default=None, type='int',
      help='Number of processes, passed by multi script')
  rep_options.add_option('-q', dest='queue',
      default='gtx1080ti',
      help='SLURM queue on which to run the jobs [Default: %default]')
  parser.add_option_group(rep_options)

  (options, args) = parser.parse_args()

  if len(args) != 2:
    parser.error('Must provide parameters and data directory.')
  else:
    params_file = os.path.abspath(args[0])
    data_dir = os.path.abspath(args[1])

  """
  #######################################################
  # prep work

  if os.path.isdir(options.out_dir):
    print('Output directory %s exists. Please remove.' % options.out_dir)
    exit(1)
  os.mkdir(options.out_dir)

  #######################################################
  # train

  jobs = []
  for pi in range(options.processes):
    rep_dir = '%s/rep%d' % (options.out_dir, pi)
    os.mkdir(rep_dir)

    cmd = '. /home/drk/anaconda3/etc/profile.d/conda.sh;'
    cmd += ' conda activate tf1.15-gpu;'
    cmd += ' echo $HOSTNAME;'

    cmd += ' basenji_train.py' 
    cmd += ' %s' % options_string(options, train_options, rep_dir)
    cmd += ' %s %s' % (params_file, data_dir)

    name = '%s-train%d' % (options.name, pi)
    sbf = os.path.abspath('%s/train.sb' % rep_dir)
    outf = os.path.abspath('%s/train.out' % rep_dir)
    errf = os.path.abspath('%s/train.err' % rep_dir)

    j = slurm.Job(cmd, name,
        outf, errf, sbf,
        queue=options.queue, gpu=1,
        mem=23000, time='28-0:0:0')
    jobs.append(j)

  slurm.multi_run(jobs, max_proc=options.processes, verbose=True,
                  launch_sleep=10, update_sleep=60)
  """

  #######################################################
  # test

  jobs = []
  for pi in range(options.processes):
    rep_dir = '%s/rep%d' % (options.out_dir, pi)
    test_dir = '%s/test_out' % rep_dir

    cmd = '. /home/drk/anaconda3/etc/profile.d/conda.sh;'
    cmd += ' conda activate tf1.15-gpu;'
    cmd += ' echo $HOSTNAME;'

    cmd += ' /home/drk/code/basenji2/bin/basenji_test.py' 
    cmd += ' --rc --shifts "1,0,-1"'
    cmd += ' -o %s' % test_dir
    cmd += ' %s %s/model_best.h5 %s' % (params_file, rep_dir, data_dir)

    name = '%s-test%d' % (options.name, pi)
    sbf = os.path.abspath('%s/test.sb' % rep_dir)
    outf = os.path.abspath('%s/test.out' % rep_dir)
    errf = os.path.abspath('%s/test.err' % rep_dir)

    j = slurm.Job(cmd, name,
        outf, errf, sbf,
        queue=options.queue, gpu=1,
        mem=23000, time='4:0:0')
    jobs.append(j)

  slurm.multi_run(jobs, max_proc=options.processes, verbose=True,
                  launch_sleep=10, update_sleep=60)


def options_string(options, train_options, out_dir):
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

      elif opt.dest == 'out_dir':
        opt_value = out_dir
      
      options_str += ' %s %s' % (opt_str, opt_value)

  return options_str


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
