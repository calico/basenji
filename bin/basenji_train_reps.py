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
import json
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
  usage = 'usage: %prog [options] <params_file> <data1_dir> ...'
  parser = OptionParser(usage)

  # train
  train_options = OptionGroup(parser, 'basenji_train.py options')
  train_options.add_option('-k', dest='keras_fit',
      default=False, action='store_true',
      help='Train with Keras fit method [Default: %default]')
  train_options.add_option('-o', dest='out_dir',
      default='train_out',
      help='Output directory for test statistics [Default: %default]')
  train_options.add_option('--restore', dest='restore',
      help='Restore model and continue training [Default: %default]')
  train_options.add_option('--trunk', dest='trunk',
      default=False, action='store_true',
      help='Restore only model trunk [Default: %default]')
  train_options.add_option('--tfr_train', dest='tfr_train_pattern',
      default=None,
      help='Training TFR pattern string appended to data_dir/tfrecords for subsetting [Default: %default]')
  train_options.add_option('--tfr_eval', dest='tfr_eval_pattern',
      default=None,
      help='Evaluation TFR pattern string appended to data_dir/tfrecords for subsetting [Default: %default]')
  parser.add_option_group(train_options)

  # test
  test_options = OptionGroup(parser, 'basenji_test.py options')
  test_options.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Average forward and reverse complement predictions [Default: %default]')
  test_options.add_option('--shifts', dest='shifts',
      default='0', type='str',
      help='Ensemble prediction shifts [Default: %default]')
  parser.add_option_group(test_options)

  # multi
  rep_options = OptionGroup(parser, 'replication options')
  rep_options.add_option('-e', dest='conda_env',
      default='tf2-gpu',
      help='Anaconda environment [Default: %default]')
  rep_options.add_option('--name', dest='name',
      default='reps', help='SLURM name prefix [Default: %default]')
  rep_options.add_option('-p', dest='processes',
      default=None, type='int',
      help='Number of processes, passed by multi script')
  rep_options.add_option('-q', dest='queue',
      default='gtx1080ti',
      help='SLURM queue on which to run the jobs [Default: %default]')
  rep_options.add_option('-r', dest='restart',
      default=False, action='store_true')
  parser.add_option_group(rep_options)

  (options, args) = parser.parse_args()

  if len(args) < 2:
    parser.error('Must provide parameters and data directory.')
  else:
    params_file = os.path.abspath(args[0])
    data_dirs = [os.path.abspath(arg) for arg in args[1:]]

  num_data = len(data_dirs)

  # read model parameters
  with open(params_file) as params_open:
    params = json.load(params_open)
  params_train = params['train']

  #######################################################
  # prep work
  
  if not options.restart and os.path.isdir(options.out_dir):
    print('Output directory %s exists. Please remove.' % options.out_dir)
    exit(1)
  if not os.path.isdir(options.out_dir):
    os.mkdir(options.out_dir)

  #######################################################
  # train

  jobs = []
  for pi in range(options.processes):
    rep_dir = '%s/%d' % (options.out_dir, pi)
    if options.restart and os.path.isdir(rep_dir):
      print('%s found and skipped.' % rep_dir)
    else:
      os.mkdir(rep_dir)

      cmd = '. /home/drk/anaconda3/etc/profile.d/conda.sh;'
      cmd += ' conda activate %s;' % options.conda_env
      cmd += ' echo $HOSTNAME;'

      cmd += ' basenji_train.py' 
      cmd += ' %s' % options_string(options, train_options, '%s/train'%rep_dir)
      cmd += ' %s %s' % (params_file, ' '.join(data_dirs))

      name = '%s-train%d' % (options.name, pi)
      sbf = os.path.abspath('%s/train.sb' % rep_dir)
      outf = os.path.abspath('%s/train.out' % rep_dir)
      errf = os.path.abspath('%s/train.err' % rep_dir)

      j = slurm.Job(cmd, name,
                    outf, errf, sbf,
                    queue=options.queue,
                    gpu=params_train.get('num_gpu',1),
                    mem=30000, time='28-0:0:0')
      jobs.append(j)

  slurm.multi_run(jobs, max_proc=options.processes, verbose=True,
                  launch_sleep=10, update_sleep=60)

  #######################################################
  # test train

  jobs = []
  for pi in range(options.processes):
    rep_dir = '%s/%d' % (options.out_dir, pi)

    for di in range(num_data):
      if num_data == 1:
        out_dir = '%s/test_train' % rep_dir
        model_file = '%s/train/model_check.h5' % rep_dir
      else:
        out_dir = '%s/test%d_train' % (rep_dir, di)
        model_file = '%s/train/model%d_check.h5' % (rep_dir, di)
    
      # check if done
      acc_file = '%s/acc.txt' % out_dir
      if options.restart and os.path.isfile(acc_file):
        print('%s already generated.' % acc_file)
      else:
        # basenji test
        cmd = '. /home/drk/anaconda3/etc/profile.d/conda.sh;'
        cmd += ' conda activate %s;' % options.conda_env
        cmd += ' basenji_test.py'
        cmd += ' --head %d' % di
        cmd += ' -o %s' % out_dir
        if options.rc:
          cmd += ' --rc'
        if options.shifts:
          cmd += ' --shifts %s' % options.shifts
        cmd += ' --split train'
        cmd += ' %s' % params_file
        cmd += ' %s' % model_file
        cmd += ' %s' % data_dirs[di]

        name = '%s-testtr%d' % (options.name, pi)
        j = slurm.Job(cmd,
                        name=name,
                        out_file='%s.out'%out_dir,
                        err_file='%s.err'%out_dir,
                        queue=options.queue,
                        cpu=1, gpu=1,
                        mem=23000,
                        time='4:00:00')
        jobs.append(j)

  #######################################################
  # test best

  for pi in range(options.processes):
    rep_dir = '%s/%d' % (options.out_dir, pi)

    for di in range(num_data):
      if num_data == 1:
        out_dir = '%s/test' % rep_dir
        model_file = '%s/train/model_best.h5' % rep_dir
      else:
        out_dir = '%s/test%d' % (rep_dir, di)
        model_file = '%s/train/model%d_best.h5' % (rep_dir, di)
    
      # check if done
      acc_file = '%s/acc.txt' % out_dir
      if options.restart and os.path.isfile(acc_file):
        print('%s already generated.' % acc_file)
      else:
        # basenji test
        cmd = '. /home/drk/anaconda3/etc/profile.d/conda.sh;'
        cmd += ' conda activate %s;' % options.conda_env
        cmd += ' basenji_test.py'
        cmd += ' --head %d' % di
        cmd += ' -o %s' % out_dir
        if options.rc:
          cmd += ' --rc'
        if options.shifts:
          cmd += ' --shifts %s' % options.shifts
        cmd += ' %s' % params_file
        cmd += ' %s' % model_file
        cmd += ' %s' % data_dirs[di]

        name = '%s-test%d' % (options.name, pi)
        j = slurm.Job(cmd,
                        name=name,
                        out_file='%s.out'%out_dir,
                        err_file='%s.err'%out_dir,
                        queue=options.queue,
                        cpu=1, gpu=1,
                        mem=23000,
                        time='4:00:00')
        jobs.append(j)

  #######################################################
  # test best specificity

  for pi in range(options.processes):
    rep_dir = '%s/%d' % (options.out_dir, pi)
    
    for di in range(num_data):
      if num_data == 1:
        out_dir = '%s/test_spec' % rep_dir
        model_file = '%s/train/model_best.h5' % rep_dir
      else:
        out_dir = '%s/test%d_spec' % (rep_dir, di)
        model_file = '%s/train/model%d_best.h5' % (rep_dir, di)
    
      # check if done
      acc_file = '%s/acc.txt' % out_dir
      if options.restart and os.path.isfile(acc_file):
        print('%s already generated.' % acc_file)
      else:
        # basenji test
        cmd = '. /home/drk/anaconda3/etc/profile.d/conda.sh;'
        cmd += ' conda activate %s;' % options.conda_env
        cmd += ' basenji_test_specificity.py'
        cmd += ' --head %d' % di
        cmd += ' -o %s' % out_dir
        if options.rc:
          cmd += ' --rc'
        if options.shifts:
          cmd += ' --shifts %s' % options.shifts
        cmd += ' %s' % params_file
        cmd += ' %s' % model_file
        cmd += ' %s' % data_dirs[di]

        name = '%s-spec%d' % (options.name, pi)
        j = slurm.Job(cmd,
                        name=name,
                        out_file='%s.out'%out_dir,
                        err_file='%s.err'%out_dir,
                        queue=options.queue,
                        cpu=1, gpu=1,
                        mem=90000,
                        time='8:00:00')
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
