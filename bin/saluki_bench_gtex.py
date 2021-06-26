#!/usr/bin/env python
# Copyright 2020 Calico LLC

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
import pandas as pd

import slurm

"""
saluki_bench_gtex.py

Compute SNP expression difference scores for variants in VCF files of
fine-mapped GTEx variants to benchmark as features in a classification
task.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <models_dir>'
  parser = OptionParser(usage)

  # ssd
  parser.add_option('-f', dest='genome_fasta',
      default='%s/data/hg38.fa' % os.environ['BASENJIDIR'],
      help='Genome FASTA for sequences [Default: %default]')
  parser.add_option('-g', dest='genes_gtf',
    default='/home/drk/rnaml/data/genes/gencode36_saluki.gtf',
    help='Genes GTF [Default: %default]')
  parser.add_option('-o',dest='out_dir',
      default='ssd_gtex',
      help='Output directory for tables and plots [Default: %default]')
  parser.add_option('--shifts', dest='shifts',
      default='0', type='str',
      help='Ensemble prediction shifts [Default: %default]')

  # multi
  parser.add_option('-e', dest='conda_env',
      default='tf2.4-rna',
      help='Anaconda environment [Default: %default]')
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
  parser.add_option('-v', dest='gtex_vcf_dir',
      default='/home/drk/rnaml/data/gtex_fine/susie_pip90')
  (options, args) = parser.parse_args()

  if len(args) != 1:
    parser.error('Must provide cross-validation model directory')
  else:
    models_dir = args[0]

  #######################################################
  # prep work

  # output directory
  if not options.restart:
    if os.path.isdir(options.out_dir):
      print('Please remove %s' % options.out_dir, file=sys.stderr)
      exit(1)
    os.mkdir(options.out_dir)

  num_folds = len(glob.glob('%s/f*/model_best.h5' % models_dir))
  if not options.restart:
    for fi in range(num_folds):
      os.mkdir('%s/f%d' % (options.out_dir, fi))

  # pickle options
  options_pkl_file = '%s/options.pkl' % options.out_dir
  options_pkl = open(options_pkl_file, 'wb')
  pickle.dump(options, options_pkl)
  options_pkl.close()

  #######################################################
  # predict

  params_file = '%s/params.json' % models_dir

  cmd_base = '. /home/drk/anaconda3/etc/profile.d/conda.sh;'
  cmd_base += ' conda activate %s;' % options.conda_env
  cmd_base += ' saluki_ssd.py %s %s' % (options_pkl_file, params_file)

  jobs = []
  for gtex_pos_vcf in glob.glob('%s/*_pos.vcf' % options.gtex_vcf_dir):
    for fi in range(num_folds):
      model_file = '%s/f%d/model_best.h5' % (models_dir, fi)

      # positive job
      job_base = os.path.splitext(os.path.split(gtex_pos_vcf)[1])[0]
      out_dir = '%s/f%d/%s' % (options.out_dir, fi, job_base)
      if not options.restart or not os.path.isfile('%s/ssd.tsv'%out_dir):
        cmd = '%s -o %s %s %s' % (cmd_base, out_dir, model_file, gtex_pos_vcf)
        name = '%s_%s' % (options.name, job_base)
        j = slurm.Job(cmd, name,
            '%s.out'%out_dir, '%s.err'%out_dir,
            queue=options.queue, gpu=1,
            mem=22000, time='1-0:0:0')
        jobs.append(j)

      # negative job
      gtex_neg_vcf = gtex_pos_vcf.replace('_pos.','_neg.')
      job_base = os.path.splitext(os.path.split(gtex_neg_vcf)[1])[0]
      out_dir = '%s/f%d/%s' % (options.out_dir, fi, job_base)
      if not options.restart or not os.path.isfile('%s/ssd.tsv'%out_dir):
        cmd = '%s -o %s %s %s' % (cmd_base, out_dir, model_file, gtex_neg_vcf)
        name = '%s_%s' % (options.name, job_base)
        j = slurm.Job(cmd, name,
            '%s.out'%out_dir, '%s.err'%out_dir,
            queue=options.queue, gpu=1,
            mem=22000, time='1-0:0:0')
        jobs.append(j)

  slurm.multi_run(jobs, max_proc=options.max_proc, verbose=True,
                  launch_sleep=10, update_sleep=60)

  #######################################################
  # ensemble

  if not os.path.isdir('%s/ensemble' % (options.out_dir)):
    os.mkdir('%s/ensemble' % (options.out_dir))

  for gtex_pos_vcf in glob.glob('%s/*_pos.vcf' % options.gtex_vcf_dir):
    tissue = os.path.splitext(os.path.split(gtex_pos_vcf)[1])[0][:-4]
    ensemble_ssd(options.out_dir, tissue, 'pos', num_folds)
    ensemble_ssd(options.out_dir, tissue, 'neg', num_folds)


  #######################################################
  # classify

  cmd_base = 'saluki_bench_classify.py -a -i 100 -p 2 -r 44 -s'

  jobs = []
  for gtex_pos_vcf in glob.glob('%s/*_pos.vcf' % options.gtex_vcf_dir):
    tissue = os.path.splitext(os.path.split(gtex_pos_vcf)[1])[0][:-4]
    ssd_pos = '%s/ensemble/%s_pos/ssd.tsv' % (options.out_dir, tissue)
    ssd_neg = '%s/ensemble/%s_neg/ssd.tsv' % (options.out_dir, tissue)
    out_dir = '%s/ensemble/%s_class' % (options.out_dir, tissue)
    if not options.restart or not os.path.isfile('%s/stats.txt' % out_dir):
      cmd = '%s -o %s %s %s' % (cmd_base, out_dir, ssd_pos, ssd_neg)
      j = slurm.Job(cmd, tissue,
          '%s.out'%out_dir, '%s.err'%out_dir,
          queue='standard', cpu=2,
          mem=22000, time='1-0:0:0')
      jobs.append(j)

  slurm.multi_run(jobs, verbose=True)


def ensemble_ssd(out_dir, tissue, posneg, num_folds):
  # read fold0
  ssd_file = '%s/f0/%s_%s/ssd.tsv' % (out_dir, tissue, posneg)
  ensemble_df = pd.read_csv(ssd_file, sep='\t')

  # add next folds
  for fi in range(1, num_folds):
    ssd_file = '%s/f%d/%s_%s/ssd.tsv' % (out_dir, fi, tissue, posneg)
    fold_df = pd.read_csv(ssd_file, sep='\t')
    ensemble_df.SSD += fold_df.SSD

  # take mean
  ensemble_df.SSD /= num_folds

  # write
  ssd_ens_dir = '%s/ensemble/%s_%s' % (out_dir, tissue, posneg)
  if not os.path.isdir(ssd_ens_dir):
    os.mkdir(ssd_ens_dir)
  ssd_ens_file = '%s/ensemble/%s_%s/ssd.tsv' % (out_dir, tissue, posneg)
  ensemble_df.to_csv(ssd_ens_file, sep='\t', index=False)

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
