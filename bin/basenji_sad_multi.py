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
import gc
import glob
import os
import pickle
import shutil
import subprocess
import sys

import numpy as np
import zarr

import slurm

"""
basenji_sad_multi.py

Compute SNP expression difference scores for variants in a VCF file,
using multiple processes.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <model_file> <vcf_file>'
  parser = OptionParser(usage)
  parser.add_option('-b',dest='batch_size',
      default=256, type='int',
      help='Batch size [Default: %default]')
  parser.add_option('-c', dest='csv',
      default=False, action='store_true',
      help='Print table as CSV [Default: %default]')
  parser.add_option('-f', dest='genome_fasta',
      default='%s/assembly/hg19.fa' % os.environ['HG19'],
      help='Genome FASTA for sequences [Default: %default]')
  parser.add_option('-g', dest='genome_file',
      default='%s/assembly/human.hg19.genome' % os.environ['HG19'],
      help='Chromosome lengths file [Default: %default]')
  parser.add_option('-l', dest='seq_len',
      default=131072, type='int',
      help='Sequence length provided to the model [Default: %default]')
  parser.add_option('--local',dest='local',
      default=1024, type='int',
      help='Local SAD score [Default: %default]')
  parser.add_option('-n', dest='norm_file',
      default=None,
      help='Normalize SAD scores')
  parser.add_option('-o',dest='out_dir',
      default='sad',
      help='Output directory for tables and plots [Default: %default]')
  parser.add_option('-p', dest='processes',
      default=None, type='int',
      help='Number of processes, passed by multi script')
  parser.add_option('--pseudo', dest='log_pseudo',
      default=1, type='float',
      help='Log2 pseudocount [Default: %default]')
  parser.add_option('-q', dest='queue',
      default='k80',
      help='SLURM queue on which to run the jobs [Default: %default]')
  parser.add_option('-r', dest='restart',
      default=False, action='store_true',
      help='Restart a partially completed job [Default: %default]')
  parser.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Average forward and reverse complement predictions [Default: %default]')
  parser.add_option('--shifts', dest='shifts',
      default='0', type='str',
      help='Ensemble prediction shifts [Default: %default]')
  parser.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')
  parser.add_option('--ti', dest='track_indexes',
      default=None, type='str',
      help='Comma-separated list of target indexes to output BigWig tracks')
  parser.add_option('-u', dest='penultimate',
      default=False, action='store_true',
      help='Compute SED in the penultimate layer [Default: %default]')
  parser.add_option('-z', dest='zarr',
      default=False, action='store_true',
      help='Output max SAR to sad.zarr [Default: %default]')
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
    if not options.restart or not job_completed(options.out_dir, pi, options.zarr, options.csv):
      cmd = 'source activate py3_gpu; basenji_sad.py %s %s %d' % (
          options_pkl_file, ' '.join(args), pi)
      name = 'sad_p%d' % pi
      outf = '%s/job%d.out' % (options.out_dir, pi)
      errf = '%s/job%d.err' % (options.out_dir, pi)
      j = slurm.Job(cmd, name,
          outf, errf,
          queue=options.queue, gpu=1,
          mem=15000, time='7-0:0:0')
      jobs.append(j)

  slurm.multi_run(jobs, max_proc=options.processes, verbose=True, sleep_time=60)

  #######################################################
  # collect output

  if options.zarr:
    collect_zarr('sad_table.zarr', options.out_dir, options.processes)

  else:
    collect_table('sad_table.txt', options.out_dir, options.processes)

  # for pi in range(options.processes):
  #     shutil.rmtree('%s/job%d' % (options.out_dir,pi))


def collect_table(file_name, out_dir, num_procs):
  os.rename('%s/job0/%s' % (out_dir, file_name), '%s/%s' % (out_dir, file_name))
  for pi in range(1, num_procs):
    subprocess.call(
        'tail -n +2 %s/job%d/%s >> %s/%s' % (out_dir, pi, file_name, out_dir,
                                             file_name),
        shell=True)


def collect_zarr(file_name, out_dir, num_procs):
  final_zarr_file = '%s/%s' % (out_dir, file_name)

  # seed w/ job0
  job_zarr_file = '%s/job0/%s' % (out_dir, file_name)
  shutil.copytree(job_zarr_file, final_zarr_file)

  # open final
  final_zarr_open = zarr.open_group(final_zarr_file)

  for pi in range(1, num_procs):
    # open job
    job_zarr_file = '%s/job%d/%s' % (out_dir, pi, file_name)
    job_zarr_open = zarr.open_group(job_zarr_file, 'r')

    # append to final
    for key in final_zarr_open.keys():
      final_zarr_open[key].append(job_zarr_open[key])


def job_completed(out_dir, pi, opt_zarr, opt_csv):
  """Check whether a specific job has generated its
     output file."""
  if opt_zarr:
    out_file = '%s/job%d/sad_table.zarr' % (out_dir,pi)
  elif opt_csv:
    out_file = '%s/job%d/sad_table.csv' % (out_dir,pi)
  else:
    out_file = '%s/job%d/sad_table.txt' % (out_dir,pi)
  return os.path.isfile(out_file) or os.path.isdir(out_file)


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
