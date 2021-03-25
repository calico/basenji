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
import copy
import glob
import json
import os
import pdb
import pickle
import shutil
import subprocess
import sys

from natsort import natsorted

import slurm

"""
basenji_train_folds.py

Train Basenji model replicates on cross folds using given parameters and data.
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
      help='Restore model and continue training, from existing fold train dir [Default: %default]')
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
  rep_options.add_option('-c', dest='crosses',
      default=1, type='int',
      help='Number of cross-fold rounds [Default:%default]')
  rep_options.add_option('-e', dest='conda_env',
      default='tf2-gpu',
      help='Anaconda environment [Default: %default]')
  rep_options.add_option('-f', dest='fold_subset',
      default=None, type='int',
      help='Run a subset of folds [Default:%default]')
  rep_options.add_option('--name', dest='name',
      default='fold', help='SLURM name prefix [Default: %default]')
  rep_options.add_option('-p', dest='processes',
      default=None, type='int',
      help='Number of processes, passed by multi script')
  rep_options.add_option('-q', dest='queue',
      default='gtx1080ti',
      help='SLURM queue on which to run the jobs [Default: %default]')
  rep_options.add_option('-r', dest='restart',
      default=False, action='store_true')
  rep_options.add_option('--spec_off', dest='spec_off',
      default=False, action='store_true')
  rep_options.add_option('--test_off', dest='test_off',
      default=False, action='store_true')
  rep_options.add_option('--test_train_off', dest='test_train_off',
      default=False, action='store_true')
  parser.add_option_group(rep_options)

  (options, args) = parser.parse_args()

  if len(args) < 2:
    parser.error('Must provide parameters and data directory.')
  else:
    params_file = os.path.abspath(args[0])
    data_dirs = [os.path.abspath(arg) for arg in args[1:]]

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

  # read data parameters
  num_data = len(data_dirs)
  data_stats_file = '%s/statistics.json' % data_dirs[0]
  with open(data_stats_file) as data_stats_open:
    data_stats = json.load(data_stats_open)

  # count folds
  num_folds = len([dkey for dkey in data_stats if dkey.startswith('fold')])

  # subset folds
  if options.fold_subset is not None:
    num_folds = min(options.fold_subset, num_folds)

  #######################################################
  # train

  jobs = []

  for ci in range(options.crosses):
    for fi in range(num_folds):
      rep_dir = '%s/f%d_c%d' % (options.out_dir, fi, ci)
      if options.restart and os.path.isdir(rep_dir):
        print('%s found and skipped.' % rep_dir)
      else:
        # make rep dir
        os.mkdir(rep_dir)

        # make rep data
        rep_data_dirs = []
        for di in range(num_data):
          rep_data_dirs.append('%s/data%d' % (rep_dir, di))
          make_rep_data(data_dirs[di], rep_data_dirs[-1], fi, ci)

        # train command
        cmd = '. /home/drk/anaconda3/etc/profile.d/conda.sh;'
        cmd += ' conda activate %s;' % options.conda_env
        cmd += ' echo $HOSTNAME;'

        cmd += ' basenji_train.py' 
        cmd += ' %s' % options_string(options, train_options, rep_dir)
        cmd += ' %s %s' % (params_file, ' '.join(rep_data_dirs))

        name = '%s-train-f%dc%d' % (options.name, fi, ci)
        sbf = os.path.abspath('%s/train.sb' % rep_dir)
        outf = os.path.abspath('%s/train.out' % rep_dir)
        errf = os.path.abspath('%s/train.err' % rep_dir)

        j = slurm.Job(cmd, name,
                      outf, errf, sbf,
                      queue=options.queue,
                      cpu=4,
                      gpu=params_train.get('num_gpu',1),
                      mem=30000, time='28-0:0:0')
        jobs.append(j)

  slurm.multi_run(jobs, max_proc=options.processes, verbose=True,
                  launch_sleep=10, update_sleep=60)
  

  #######################################################
  # test train

  jobs = []

  if not options.test_train_off:
    for ci in range(options.crosses):
      for fi in range(num_folds):
        it_dir = '%s/f%d_c%d' % (options.out_dir, fi, ci)

        for di in range(num_data):
          if num_data == 1:
            out_dir = '%s/test_train' % it_dir
            model_file = '%s/train/model_check.h5' % it_dir
          else:
            out_dir = '%s/test%d_train' % (it_dir, di)
            model_file = '%s/train/model%d_check.h5' % (it_dir, di)
        
          # check if done
          acc_file = '%s/acc.txt' % out_dir
          if os.path.isfile(acc_file):
            print('%s already generated.' % acc_file)
          else:
            # basenji test
            basenji_cmd = '. /home/drk/anaconda3/etc/profile.d/conda.sh;'
            basenji_cmd += ' conda activate %s;' % options.conda_env
            basenji_cmd += ' basenji_test.py'
            basenji_cmd += ' --head %d' % di
            basenji_cmd += ' -o %s' % out_dir
            if options.rc:
              basenji_cmd += ' --rc'
            if options.shifts:
              basenji_cmd += ' --shifts %s' % options.shifts
            basenji_cmd += ' --split train'
            basenji_cmd += ' %s' % params_file
            basenji_cmd += ' %s' % model_file
            basenji_cmd += ' %s/data%d' % (it_dir, di)

            name = '%s-testtr-f%dc%d' % (options.name, fi, ci)
            basenji_job = slurm.Job(basenji_cmd,
                            name=name,
                            out_file='%s.out'%out_dir,
                            err_file='%s.err'%out_dir,
                            queue=options.queue,
                            cpu=1, gpu=1,
                            mem=23000,
                            time='8:00:00')
            jobs.append(basenji_job)


  #######################################################
  # test best

  if not options.test_off:
    for ci in range(options.crosses):
      for fi in range(num_folds):
        it_dir = '%s/f%d_c%d' % (options.out_dir, fi, ci)

        for di in range(num_data):
          if num_data == 1:
            out_dir = '%s/test' % it_dir
            model_file = '%s/train/model_best.h5' % it_dir
          else:
            out_dir = '%s/test%d' % (it_dir, di)
            model_file = '%s/train/model%d_best.h5' % (it_dir, di)

          # check if done
          acc_file = '%s/acc.txt' % out_dir
          if os.path.isfile(acc_file):
            print('%s already generated.' % acc_file)
          else:
            # basenji test
            basenji_cmd = '. /home/drk/anaconda3/etc/profile.d/conda.sh;'
            basenji_cmd += ' conda activate %s;' % options.conda_env
            basenji_cmd += ' basenji_test.py'
            basenji_cmd += ' --head %d' % di
            basenji_cmd += ' -o %s' % out_dir
            if options.rc:
              basenji_cmd += ' --rc'
            if options.shifts:
              basenji_cmd += ' --shifts %s' % options.shifts
            basenji_cmd += ' %s' % params_file
            basenji_cmd += ' %s' % model_file
            basenji_cmd += ' %s/data%d' % (it_dir, di)

            name = '%s-test-f%dc%d' % (options.name, fi, ci)
            basenji_job = slurm.Job(basenji_cmd,
                            name=name,
                            out_file='%s.out'%out_dir,
                            err_file='%s.err'%out_dir,
                            queue=options.queue,
                            cpu=1, gpu=1,
                            mem=23000,
                            time='4:00:00')
            jobs.append(basenji_job)

  #######################################################
  # test best specificity
  
  if not options.spec_off:
    for ci in range(options.crosses):
      for fi in range(num_folds):
        it_dir = '%s/f%d_c%d' % (options.out_dir, fi, ci)

        for di in range(num_data):
          if num_data == 1:
            out_dir = '%s/test_spec' % it_dir
            model_file = '%s/train/model_best.h5' % it_dir
          else:
            out_dir = '%s/test%d_spec' % (it_dir, di)
            model_file = '%s/train/model%d_best.h5' % (it_dir, di)

          # check if done
          acc_file = '%s/acc.txt' % out_dir
          if os.path.isfile(acc_file):
            print('%s already generated.' % acc_file)
          else:
            # basenji test
            basenji_cmd = '. /home/drk/anaconda3/etc/profile.d/conda.sh;'
            basenji_cmd += ' conda activate %s;' % options.conda_env
            basenji_cmd += ' basenji_test_specificity.py'
            basenji_cmd += ' --head %d' % di
            basenji_cmd += ' -o %s' % out_dir
            if options.rc:
              basenji_cmd += ' --rc'
            if options.shifts:
              basenji_cmd += ' --shifts %s' % options.shifts
            basenji_cmd += ' %s' % params_file
            basenji_cmd += ' %s' % model_file
            basenji_cmd += ' %s/data%d' % (it_dir, di)

            name = '%s-spec-f%dc%d' % (options.name, fi, ci)
            basenji_job = slurm.Job(basenji_cmd,
                            name=name,
                            out_file='%s.out'%out_dir,
                            err_file='%s.err'%out_dir,
                            queue=options.queue,
                            cpu=1, gpu=1,
                            mem=90000,
                            time='6:00:00')
            jobs.append(basenji_job)
        
  slurm.multi_run(jobs, max_proc=options.processes, verbose=True,
                  launch_sleep=10, update_sleep=60)

def make_rep_data(data_dir, rep_data_dir, fi, ci):
  # read data parameters
  data_stats_file = '%s/statistics.json' % data_dir
  with open(data_stats_file) as data_stats_open:
    data_stats = json.load(data_stats_open)

  # sequences per fold
  fold_seqs = []
  dfi = 0
  while 'fold%d_seqs'%dfi in data_stats:
    fold_seqs.append(data_stats['fold%d_seqs'%dfi])
    del data_stats['fold%d_seqs'%dfi]
    dfi += 1
  num_folds = dfi

  # split folds into train/valid/test
  test_fold = fi
  valid_fold = (fi+1+ci) % num_folds
  train_folds = [fold for fold in range(num_folds) if fold not in [valid_fold,test_fold]]

  # make data directory
  os.mkdir(rep_data_dir)

  # dump data stats
  data_stats['test_seqs'] = fold_seqs[test_fold]
  data_stats['valid_seqs'] = fold_seqs[valid_fold]
  data_stats['train_seqs'] = sum([fold_seqs[tf] for tf in train_folds])
  with open('%s/statistics.json'%rep_data_dir, 'w') as data_stats_open:
    json.dump(data_stats, data_stats_open, indent=4)

  # copy targets
  shutil.copy('%s/targets.txt'%data_dir, '%s/targets.txt'%rep_data_dir)

  # sym link tfrecords
  rep_tfr_dir = '%s/tfrecords' % rep_data_dir
  os.mkdir(rep_tfr_dir)

  # test tfrecords
  ti = 0
  test_tfrs = natsorted(glob.glob('%s/tfrecords/fold%d-*.tfr' % (data_dir, test_fold)))
  for test_tfr in test_tfrs:
    test_tfr = os.path.abspath(test_tfr)
    test_rep_tfr = '%s/test-%d.tfr' % (rep_tfr_dir, ti)
    os.symlink(test_tfr, test_rep_tfr)
    ti += 1

  # valid tfrecords
  ti = 0
  valid_tfrs = natsorted(glob.glob('%s/tfrecords/fold%d-*.tfr' % (data_dir, valid_fold)))
  for valid_tfr in valid_tfrs:
    valid_tfr = os.path.abspath(valid_tfr)
    valid_rep_tfr = '%s/valid-%d.tfr' % (rep_tfr_dir, ti)
    os.symlink(valid_tfr, valid_rep_tfr)
    ti += 1

  # train tfrecords
  ti = 0
  train_tfrs = []
  for tfi in train_folds:
    train_tfrs += natsorted(glob.glob('%s/tfrecords/fold%d-*.tfr' % (data_dir, tfi)))
  for train_tfr in train_tfrs:
    train_tfr = os.path.abspath(train_tfr)
    train_rep_tfr = '%s/train-%d.tfr' % (rep_tfr_dir, ti)
    os.symlink(train_tfr, train_rep_tfr)
    ti += 1


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
      opt_value = '%s/train' % rep_dir

    # find matching restore
    elif opt.dest == 'restore':
      fold_dir_mid = rep_dir.split('/')[-1]
      if options.trunk:
        opt_value = '%s/%s/train/model_trunk.h5' % (opt_value, fold_dir_mid)
      else:
        opt_value = '%s/%s/train/model_best.h5' % (opt_value, fold_dir_mid)

    options_str += ' %s %s' % (opt_str, opt_value)

  return options_str


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
