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
sonnet_bench_gtex.py

Compute SNP expression difference scores for variants in VCF files of
fine-mapped GTEx variants to benchmark as features in a classification
task.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <model_file>'
  parser = OptionParser(usage)

  # sad
  parser.add_option('-b', dest='batch_size',
      default=4, type='int',
      help='Batch size [Default: %default]')
  parser.add_option('-c', dest='slice_center',
      default=None, type='int',
      help='Slice center positions [Default: %default]')
  parser.add_option('-f', dest='genome_fasta',
      default='%s/data/hg38.fa' % os.environ['BASENJIDIR'],
      help='Genome FASTA for sequences [Default: %default]')
  parser.add_option('-n', dest='norm_file',
      default=None,
      help='Normalize SAD scores')
  parser.add_option('-o',dest='out_dir',
      default='sad_gtex',
      help='Output directory for tables and plots [Default: %default]')
  parser.add_option('--pseudo', dest='log_pseudo',
      default=1, type='float',
      help='Log2 pseudocount [Default: %default]')
  parser.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Average forward and reverse complement predictions [Default: %default]')
  parser.add_option('--shifts', dest='shifts',
      default='0', type='str',
      help='Ensemble prediction shifts [Default: %default]')
  parser.add_option('--species', dest='species',
      default='human')
  parser.add_option('--stats', dest='sad_stats',
      default='SAD',
      help='Comma-separated list of stats to save. [Default: %default]')
  parser.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')

  # classify
  parser.add_option('--msl', dest='msl',
      default=1, type='int',
      help='Random forest min_samples_leaf [Default: %default]')

  # multi
  parser.add_option('-e', dest='conda_env',
      default='tf2.4',
      help='Anaconda environment [Default: %default]')
  parser.add_option('-g', dest='gtex_vcf_dir',
      default='/home/drk/seqnn/data/gtex_fine/susie_pip90')
  parser.add_option('--name', dest='name',
      default='gtex', help='SLURM name prefix [Default: %default]')
  parser.add_option('--max_proc', dest='max_proc',
      default=None, type='int',
      help='Maximum concurrent processes [Default: %default]')
  parser.add_option('-p', dest='processes',
      default=None, type='int',
      help='Number of processes, passed by multi script. \
            (Unused, but needs to appear as dummy.)')
  parser.add_option('-q', dest='queue',
      default='gtx1080ti',
      help='SLURM queue on which to run the jobs [Default: %default]')
  parser.add_option('-r', dest='restart',
      default=False, action='store_true',
      help='Restart a partially completed job [Default: %default]')
  (options, args) = parser.parse_args()

  if len(args) != 1:
    parser.error('Must provide model file')
  else:
    model_file = args[0]

  #######################################################
  # prep work

  # output directory
  if not options.restart:
    if os.path.isdir(options.out_dir):
      print('Please remove %s' % options.out_dir, file=sys.stderr)
      exit(1)
    os.mkdir(options.out_dir)

  sad_stats_list = options.sad_stats.split(',')

  # pickle options
  options_pkl_file = '%s/options.pkl' % options.out_dir
  options_pkl = open(options_pkl_file, 'wb')
  pickle.dump(options, options_pkl)
  options_pkl.close()

  #######################################################
  # predict

  cmd_base = '. /home/drk/anaconda3/etc/profile.d/conda.sh;'
  cmd_base += ' conda activate %s;' % options.conda_env
  cmd_base += ' sonnet_sad.py %s %s' % (options_pkl_file, model_file)

  jobs = []
  for gtex_pos_vcf in glob.glob('%s/*_pos.vcf' % options.gtex_vcf_dir):
    # positive job
    job_base = os.path.splitext(os.path.split(gtex_pos_vcf)[1])[0]
    out_dir = '%s/%s' % (options.out_dir, job_base)
    if not options.restart or not os.path.isfile('%s/sad.h5'%out_dir):
      cmd = '%s -o %s %s' % (cmd_base, out_dir, gtex_pos_vcf)
      name = '%s_%s' % (options.name, job_base)
      j = slurm.Job(cmd, name,
          '%s.out'%out_dir, '%s.err'%out_dir,
          queue=options.queue, gpu=1,
          mem=22000, time='1-0:0:0')
      jobs.append(j)

    # negative job
    gtex_neg_vcf = gtex_pos_vcf.replace('_pos.','_neg.')
    job_base = os.path.splitext(os.path.split(gtex_neg_vcf)[1])[0]
    out_dir = '%s/%s' % (options.out_dir, job_base)
    if not options.restart or not os.path.isfile('%s/sad.h5'%out_dir):
      cmd = '%s -o %s %s' % (cmd_base, out_dir, gtex_neg_vcf)
      name = '%s_%s' % (options.name, job_base)
      j = slurm.Job(cmd, name,
          '%s.out'%out_dir, '%s.err'%out_dir,
          queue=options.queue, gpu=1,
          mem=22000, time='1-0:0:0')
      jobs.append(j)

  slurm.multi_run(jobs, max_proc=options.max_proc, verbose=True,
                  launch_sleep=10, update_sleep=60)


  #######################################################
  # classify

  cmd_base = 'basenji_bench_classify.py --iscale 0.1 -i 100 -p 2 -r 44 -s'

  cmd_base += ' --msl %d' % options.msl

  jobs = []
  for gtex_pos_vcf in glob.glob('%s/*_pos.vcf' % options.gtex_vcf_dir):
    tissue = os.path.splitext(os.path.split(gtex_pos_vcf)[1])[0][:-4]
    sad_pos = '%s/%s_pos/sad.h5' % (options.out_dir, tissue)
    sad_neg = '%s/%s_neg/sad.h5' % (options.out_dir, tissue)

    for sad_stat in sad_stats_list:
      out_dir = '%s/%s_class-%s' % (options.out_dir, tissue, sad_stat.lower())
      if not options.restart or not os.path.isfile('%s/stats.txt' % out_dir):
        cmd = '%s -o %s --stat %s %s %s' % (cmd_base, out_dir, sad_stat, sad_pos, sad_neg)
        j = slurm.Job(cmd, tissue,
            '%s.out'%out_dir, '%s.err'%out_dir,
            queue='standard', cpu=2,
            mem=22000, time='20:0:0')
        jobs.append(j)

  slurm.multi_run(jobs, verbose=True)


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
