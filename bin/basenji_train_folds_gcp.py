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
from natsort import natsorted
import os
import pdb
import pickle
import shutil
import subprocess
import sys
import time

try:
  import util
except ModuleNotFoundError:
  pass

"""
basenji_train_folds_gcp.py

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
  rep_options.add_option('--checkpoint', dest='checkpoint',
      default=False, action='store_true',
      help='Restart training from checkpoint [Default: %default]')
  rep_options.add_option('-f', dest='num_folds',
      default=None, type='int',
      help='Number of data folds [Default:%default]')
  rep_options.add_option('-r', '--restart', dest='restart',
      default=False, action='store_true')
  rep_options.add_option('--spec_off', dest='spec_off',
      default=False, action='store_true')
  rep_options.add_option('--test_off', dest='test_off',
      default=False, action='store_true')
  rep_options.add_option('--test_train_off', dest='test_train_off',
      default=False, action='store_true')
  parser.add_option_group(rep_options)

  # gcp
  gcp_options = OptionGroup(parser, 'GCP options')
  gcp_options.add_option('-d', '--disk', dest='disk_snap',
      default='tf26-snap')
  gcp_options.add_option('-i', '--init', dest='initialize',
      default=False, action='store_true',
      help='Prepare VMs but do not run commands [Default: %default]')
  gcp_options.add_option('-v', '--vm', dest='vm_base',
      default='tf26')
  gcp_options.add_option('-w', dest='worker',
      default=None)
  gcp_options.add_option('-z', '--zone', dest='zone',
      default='us-central1-a')
  parser.add_option_group(gcp_options)

  (options, args) = parser.parse_args()

  if len(args) < 2:
    parser.error('Must provide parameters and data directory.')
  else:
    params_file = args[0]
    data_dirs = args[1:]

  # read model parameters
  with open(params_file) as params_open:
    params = json.load(params_open)
  params_train = params['train']
  num_gpu = params_train.get('num_gpu', 1)

  #######################################################
  # prep work
  
  os.makedirs(options.out_dir, exist_ok=True)

  #######################################################
  # initialize VM cluster

  python_path = '/opt/conda/bin/python'
  basenji_path = '/home/drk/code/basenji/bin/'

  jobs = []

  if options.worker is None:
    for ci in range(options.crosses):
      for fi in range(options.num_folds):
        rep_label = 'f%d-c%d' % (fi, ci)
        vm_name = '%s-%s' % (options.vm_base, rep_label)

        # query VM status
        gcp_desc = 'gcloud compute instances describe %s --zone=%s' % (vm_name, options.zone)

        try:
          desc_text = subprocess.check_output(gcp_desc, shell=True).decode('UTF-8')

          if desc_text.find('status: TERMINATED') != -1:
            # start VM
            gcp_start = 'gcloud compute instances start %s' % vm_name
            gcp_start += ' --zone=%s' % options.zone
            subprocess.call(gcp_start, shell=True)

        except subprocess.CalledProcessError:

          # create VM
          gcp_create = 'gcloud compute --project=seqnn-170614 instances create %s' % vm_name
          gcp_create += ' --subnet=default --maintenance-policy=TERMINATE --service-account=1090276179925-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/devstorage.read_write,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append --boot-disk-size=1024gb --boot-disk-type=pd-balanced --no-shielded-secure-boot --shielded-vtpm --shielded-integrity-monitoring --reservation-affinity=any --metadata="install-nvidia-driver=True"'
          gcp_create += ' --machine-type=a2-highgpu-%dg' % num_gpu
          gcp_create += ' --accelerator=type=nvidia-tesla-a100,count=%d' % num_gpu
          gcp_create += ' --zone=%s' % options.zone
          gcp_create += ' --source-snapshot=%s' % options.disk_snap
          gcp_create += ' --boot-disk-device-name=%s' % vm_name
          # print(gcp_create)
          subprocess.call(gcp_create, shell=True)

        # scp/ssh needs time
        time.sleep(15)

        # copy params
        gcp_params = 'gcloud compute scp %s %s:./' % (params_file, vm_name)
        subprocess.call(gcp_params, shell=True)

        # launch worker
        cmd_fold = '%s %s/basenji_train_folds_gcp.py' % (python_path, basenji_path)
        cmd_fold += ' %s' % options_string(parser, options)
        cmd_fold += ' -w %s' % rep_label
        cmd_fold += ' %s %s' % (params_file, ' '.join(data_dirs))
        gcp_fold = 'gcloud compute ssh %s --command "%s"' % (vm_name, cmd_fold)
        jobs.append(gcp_fold)

    if not options.initialize:
      import util
      util.exec_par(jobs, verbose=True)

  #######################################################
  # prep directory

  if options.worker is not None:
    # read data parameters
    num_data = len(data_dirs)
    # data_stats_file = '%s/statistics.json' % data_dirs[0]
    # with open(data_stats_file) as data_stats_open:
    #   data_stats = json.load(data_stats_open)

    for ci in range(options.crosses):
      for fi in range(options.num_folds):
        rep_label = 'f%d_c%d' % (fi, ci)
        rep_dir = '%s/%s' % (options.out_dir, rep_label)

        # make rep dir
        os.makedirs(rep_dir, exist_ok=True)

        # make rep data
        rep_data_dirs = []
        for di in range(num_data):
          rep_data_dirs.append('%s/data%d' % (rep_dir, di))
          make_rep_data(data_dirs[di], rep_data_dirs[-1], fi, ci)


  #######################################################
  # train

  for ci in range(options.crosses):
    for fi in range(options.num_folds):
      rep_label = 'f%d-c%d' % (fi, ci)
      rep_dir = '%s/%s' % (options.out_dir, rep_label)
      vm_name = '%s-%s' % (options.vm_base, rep_label)

      if options.worker == rep_label:
        cmd_train = ' %s %s/basenji_train.py' % (python_path, basenji_path)
        cmd_train += ' %s' % options_train_string(options, train_options, rep_dir)
        cmd_train += ' %s %s' % (params_file, ' '.join(rep_data_dirs))
        cmd_train += ' | tee %s/train.out' % rep_dir
        cmd_train += ' 2> %s/train.err' % rep_dir
        subprocess.call(cmd_train, shell=True)
        # gcp_train = 'gcloud compute ssh %s --command "%s"' % (vm_fold, cmd_train)
        # subprocess.call(gcp_train, shell=True)


  #######################################################
  # test train
  """
  jobs = []
  if not options.test_train_off:
    for ci in range(options.crosses):
      for fi in range(options.num_folds):
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
  """
  #######################################################
  # test best
  
  if not options.test_off:
    for ci in range(options.crosses):
      for fi in range(options.num_folds):
        rep_label = 'f%d_c%d' % (fi, ci)
        rep_dir = '%s/%s' % (options.out_dir, rep_label)
        vm_name = '%s-%s' % (options.vm_base, rep_label)

        if options.worker == rep_label:

          for di in range(num_data):
            if num_data == 1:
              out_dir = '%s/test' % rep_dir
              model_file = '%s/train/model_best.h5' % rep_dir
            else:
              out_dir = '%s/test%d' % (rep_dir, di)
              model_file = '%s/train/model%d_best.h5' % (rep_dir, di)

            # basenji test
            cmd_test = ' %s %s/basenji_test.py' % (python_path, basenji_path)
            cmd_test += ' --head %d' % di
            cmd_test += ' -o %s' % out_dir
            if options.rc:
              cmd_test += ' --rc'
            if options.shifts:
              cmd_test += ' --shifts %s' % options.shifts
            cmd_test += ' %s' % params_file
            cmd_test += ' %s' % model_file
            cmd_test += ' %s/data%d' % (rep_dir, di)
            cmd_test += ' | tee %s.out' % out_dir
            cmd_test += ' 2> %s.err' % out_dir

            subprocess.call(cmd_test, shell=True)
 
  #######################################################
  # test best specificity
  """
  if not options.spec_off:
    for ci in range(options.crosses):
      for fi in range(options.num_folds):
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
  """

  #######################################################
  # conclusion

  if options.worker is None and not options.initialize:
    for ci in range(options.crosses):
      for fi in range(options.num_folds):
        rep_label = 'f%d_c%d' % (fi, ci)
        rep_dir = '%s/%s' % (options.out_dir, rep_label)
        vm_name = '%s-%s' % (options.vm_base, rep_label)

        # guarantee local dir
        os.makedirs(rep_dir, exist_ok=True)

        # scp results
        gcp_copy = 'gcloud compute scp --recurse %s:%s/t* %s/' % (vm_name, rep_dir, rep_dir)
        subprocess.call(gcp_copy, shell=True)

        # stop VM
        gcp_stop = 'gcloud compute instances stop %s' % vm_name
        subprocess.call(gcp_stop, shell=True)


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

  # clear existing directory
  if os.path.isdir(rep_data_dir):
    shutil.rmtree(rep_data_dir)

  # make data directory
  os.makedirs(rep_data_dir, exist_ok=True)

  # dump data stats
  data_stats['test_seqs'] = fold_seqs[test_fold]
  data_stats['valid_seqs'] = fold_seqs[valid_fold]
  data_stats['train_seqs'] = sum([fold_seqs[tf] for tf in train_folds])
  with open('%s/statistics.json'%rep_data_dir, 'w') as data_stats_open:
    json.dump(data_stats, data_stats_open, indent=4)

  # set sequence tvt
  try:
    seqs_bed_out = open('%s/sequences.bed'%rep_data_dir, 'w')
    for line in open('%s/sequences.bed'%data_dir):
      a = line.split()
      sfi = int(a[-1].replace('fold',''))
      if sfi == test_fold:
        a[-1] = 'test'
      elif sfi == valid_fold:
        a[-1] = 'valid'
      else:
        a[-1] = 'train'
      print('\t'.join(a), file=seqs_bed_out)
    seqs_bed_out.close()
  except (ValueError, FileNotFoundError):
    pass

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


def options_string(parser, options):
  options_str = ''

  for opt_group in parser.option_groups:

    for opt in opt_group.option_list:
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

      options_str += ' %s %s' % (opt_str, opt_value)

  return options_str


def options_train_string(options, train_options, rep_dir):
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
