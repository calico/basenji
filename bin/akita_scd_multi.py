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

from optparse import OptionParser
import glob
import os
import pickle
import shutil
import subprocess
import sys

import h5py
import numpy as np

import slurm

"""
akita_scd_multi.py

Compute SNP contact difference scores for variants in a VCF file,
using multiple processes.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <model_file> <vcf_file>'
  parser = OptionParser(usage)

  # scd
  parser.add_option('-f', dest='genome_fasta',
      default='%s/data/hg19.fa' % os.environ['BASENJIDIR'],
      help='Genome FASTA for sequences [Default: %default]')
  parser.add_option('-m', dest='plot_map',
      default=False, action='store_true',
      help='Plot contact map for each allele [Default: %default]')
  parser.add_option('-o',dest='out_dir',
      default='scd',
      help='Output directory for tables and plots [Default: %default]')
  parser.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Average forward and reverse complement predictions [Default: %default]')
  parser.add_option('--shifts', dest='shifts',
      default='0', type='str',
      help='Ensemble prediction shifts [Default: %default]')
  parser.add_option('--stats', dest='scd_stats',
      default='SCD',
      help='Comma-separated list of stats to save. [Default: %default]')
  parser.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')

  # multi
  parser.add_option('--cpu', dest='cpu',
      default=False, action='store_true',
      help='Run without a GPU [Default: %default]')
  parser.add_option('--name', dest='name',
      default='scd', help='SLURM name prefix [Default: %default]')
  parser.add_option('--max_proc', dest='max_proc',
      default=None, type='int',
      help='Maximum concurrent processes [Default: %default]')
  parser.add_option('-p', dest='processes',
      default=None, type='int',
      help='Number of processes, passed by multi script')
  parser.add_option('-q', dest='queue',
      default='gtx1080ti',
      help='SLURM queue on which to run the jobs [Default: %default]')
  parser.add_option('-r', dest='restart',
      default=False, action='store_true',
      help='Restart a partially completed job [Default: %default]')
  (options, args) = parser.parse_args()

  if len(args) != 3:
    parser.error('Must provide parameters and model files and VCF file')
  else:
    params_file = args[0]
    model_file = args[1]
    vcf_file = args[2]

  #######################################################
  # prep work

  # output directory
  if not options.restart:
    if os.path.isdir(options.out_dir):
      print('Please remove %s' % options.out_dir, file=sys.stderr)
      exit(1)
    os.mkdir(options.out_dir)

  # pickle options
  options_pkl_file = '%s/options.pkl' % options.out_dir
  options_pkl = open(options_pkl_file, 'wb')
  pickle.dump(options, options_pkl)
  options_pkl.close()

  #######################################################
  # launch worker threads
  jobs = []
  for pi in range(options.processes):
    if not options.restart or not job_completed(options, pi):
      if options.cpu:
        cmd = ''
      else:
        cmd = '. /home/drk/anaconda3/etc/profile.d/conda.sh;'
        cmd += 'conda activate tf1.15-gpu;'

      cmd += ' /home/drk/code/basenji2/bin/akita_scd.py %s %s %d' % (
          options_pkl_file, ' '.join(args), pi)

      name = '%s_p%d' % (options.name, pi)
      outf = '%s/job%d.out' % (options.out_dir, pi)
      errf = '%s/job%d.err' % (options.out_dir, pi)

      num_gpu = 1*(not options.cpu)

      j = slurm.Job(cmd, name,
          outf, errf,
          queue=options.queue, gpu=num_gpu,
          mem=15000, time='14-0:0:0')
      jobs.append(j)

  slurm.multi_run(jobs, max_proc=options.max_proc, verbose=True,
                  launch_sleep=10, update_sleep=60)

  #######################################################
  # collect output

  collect_h5('scd.h5', options.out_dir, options.processes)

  # for pi in range(options.processes):
  #     shutil.rmtree('%s/job%d' % (options.out_dir,pi))


def collect_table(file_name, out_dir, num_procs):
  os.rename('%s/job0/%s' % (out_dir, file_name), '%s/%s' % (out_dir, file_name))
  for pi in range(1, num_procs):
    subprocess.call(
        'tail -n +2 %s/job%d/%s >> %s/%s' % (out_dir, pi, file_name, out_dir,
                                             file_name),
        shell=True)


def collect_h5(file_name, out_dir, num_procs):
  # count variants
  num_variants = 0
  for pi in range(num_procs):
    # open job
    job_h5_file = '%s/job%d/%s' % (out_dir, pi, file_name)
    job_h5_open = h5py.File(job_h5_file, 'r')
    num_variants += len(job_h5_open['snp'])
    job_h5_open.close()

  # initialize final h5
  final_h5_file = '%s/%s' % (out_dir, file_name)
  final_h5_open = h5py.File(final_h5_file, 'w')

  # keep dict for string values
  final_strings = {}

  job0_h5_file = '%s/job0/%s' % (out_dir, file_name)
  job0_h5_open = h5py.File(job0_h5_file, 'r')
  for key in job0_h5_open.keys():
    if key in ['target_ids', 'target_labels']:
      # copy
      final_h5_open.create_dataset(key, data=job0_h5_open[key])

    elif job0_h5_open[key].dtype.char == 'S':
        final_strings[key] = []

    elif job0_h5_open[key].ndim == 1:
      final_h5_open.create_dataset(key, shape=(num_variants,), dtype=job0_h5_open[key].dtype)

    else:
      num_targets = job0_h5_open[key].shape[1]
      final_h5_open.create_dataset(key, shape=(num_variants, num_targets), dtype=job0_h5_open[key].dtype)

  job0_h5_open.close()

  # set values
  vi = 0
  for pi in range(num_procs):
    # open job
    job_h5_file = '%s/job%d/%s' % (out_dir, pi, file_name)
    job_h5_open = h5py.File(job_h5_file, 'r')

    # append to final
    for key in job_h5_open.keys():
      if key in ['target_ids', 'target_labels']:
        # once is enough
        pass

      else:
        if job_h5_open[key].dtype.char == 'S':
          final_strings[key] += list(job_h5_open[key])
        else:
          job_variants = job_h5_open[key].shape[0]
          final_h5_open[key][vi:vi+job_variants] = job_h5_open[key]

    vi += job_variants
    job_h5_open.close()

  # create final string datasets
  for key in final_strings:
    final_h5_open.create_dataset(key,
      data=np.array(final_strings[key], dtype='S'))

  final_h5_open.close()


def job_completed(options, pi):
  """Check whether a specific job has generated its
     output file."""
  out_file = '%s/job%d/scd.h5' % (options.out_dir, pi)
  return os.path.isfile(out_file) or os.path.isdir(out_file)


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
