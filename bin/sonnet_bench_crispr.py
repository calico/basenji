#!/usr/bin/env python
# Copyright 2022 Calico LLC

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
import pickle
import pdb
import os
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm

import slurm

from basenji.gene import gtf_kv
from borzoi_satg_gene_multi import collect_h5

'''
sonnet_bench_crispr.py
'''

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params> <model>'
  parser = OptionParser(usage)

  # satg
  satg_options = OptionGroup(parser, 'sonnet_satg_bed.py options')
  satg_options.add_option('-c', dest='slice_center',
      default=False, action='store_true',
      help='Slice center position(s) for gradient [Default: %default]')
  satg_options.add_option('-d', dest='mut_down',
      default=0, type='int',
      help='Nucleotides downstream of center sequence to mutate [Default: %default]')
  satg_options.add_option('-f', dest='genome_fasta',
      default='%s/data/hg38.fa' % os.environ['BASENJIDIR'],
      help='Genome FASTA for sequences [Default: %default]')
  satg_options.add_option('-l', dest='mut_len',
      default=0, type='int',
      help='Length of center sequence to mutate [Default: %default]')
  satg_options.add_option('-o', dest='out_dir',
      default='crispr_out', help='Output directory [Default: %default]')
  satg_options.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Ensemble forward and reverse complement predictions [Default: %default]')
  satg_options.add_option('--shifts', dest='shifts',
      default='0', type='str',
      help='Ensemble prediction shifts [Default: %default]')
  satg_options.add_option('--species', dest='species',
      default='human')
  satg_options.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')
  satg_options.add_option('-u', dest='mut_up',
      default=0, type='int',
      help='Nucleotides upstream of center sequence to mutate [Default: %default]')
  parser.add_option_group(satg_options)

  # multi
  bench_options = OptionGroup(parser, 'benchmark options')
  bench_options.add_option('-b', dest='bench_dir',
      default='/home/drk/seqnn/data/crispr/flowfish')
  bench_options.add_option('-e', dest='conda_env',
      default='tf28',
      help='Anaconda environment [Default: %default]')
  bench_options.add_option('--max_proc', dest='max_proc',
      default=None, type='int',
      help='Maximum concurrent processes [Default: %default]')
  bench_options.add_option('-p', dest='processes',
      default=1, type='int',
      help='Number of processes to parallelize satg. One will run locally [Default: %default]')
  bench_options.add_option('-q', dest='queue',
      default='standard',
      help='SLURM queue on which to run the jobs [Default: %default]')
  parser.add_option_group(bench_options)

  (options,args) = parser.parse_args()

  if len(args) != 1:
    parser.error('Must provide model')
  else:
    model_file = args[0]

  # crispr files
  crispr_bed_file = '%s/crispr_tss.bed' % options.bench_dir
  crispr_table_tsv = '%s/crispr_table.tsv' % options.bench_dir

  # output directory
  os.makedirs(options.out_dir, exist_ok=True)

  #######################################################
  # satg

  satg_h5f = '%s/scores.h5' % options.out_dir
  if complete_h5(satg_h5f):
    print('Satg scores computed')
  else:
    if options.processes == 1:
      satg_cmd = 'sonnet_satg_bed.py %s %s' % (model_file, crispr_bed_file)
      satg_cmd += options_string(options, satg_options)
      subprocess.call(satg_cmd, shell=True)

    else:
      # pickle options
      options_pkl_file = '%s/options.pkl' % options.out_dir
      options_pkl = open(options_pkl_file, 'wb')
      pickle.dump(options, options_pkl)
      options_pkl.close()

      # choose slurm options
      if options.queue == 'standard':
        num_gpu = 0
        num_cpu = 8
      else:
        num_gpu = 1
        num_cpu = 4

      jobs = []

      for pi in range(options.processes):
        satg_pi_h5f = '%s/job%d/scores.h5' % (options.out_dir, pi)
        if not complete_h5(satg_pi_h5f):
          satg_cmd = '. /home/drk/anaconda3/etc/profile.d/conda.sh;'
          satg_cmd += ' conda activate %s;' % options.conda_env

          satg_cmd += ' sonnet_satg_bed.py %s %s %s %d' % (
              options_pkl_file, ' '.join(args), crispr_bed_file, pi)
          name = 'crispr_p%d' % pi
          outf = '%s/job%d.out' % (options.out_dir, pi)
          errf = '%s/job%d.err' % (options.out_dir, pi)
          j = slurm.Job(satg_cmd, name,
              outf, errf,
              queue=options.queue,
              cpu=num_cpu, gpu=num_gpu,
              mem=45000, time='9-0:0:0')
          jobs.append(j)

      slurm.multi_run(jobs, max_proc=options.max_proc, verbose=True,
                      launch_sleep=10, update_sleep=60)

      # collect output
      collect_h5(options.out_dir, options.processes, 'grads')

  #######################################################
  # metrics

  # read sequences and reference scores
  with h5py.File(satg_h5f, 'r') as satg_h5:
    seq_starts = satg_h5['start'][:]
    num_seqs, seq_len, _ = satg_h5['grads'].shape
    grads_ref = np.zeros((num_seqs,seq_len), dtype='float16')
    for si in range(num_seqs):
      grads_si = satg_h5['grads'][si]
      seq_1hot = satg_h5['seqs'][si]
      n_mask = (seq_1hot.sum(axis=-1) == 0)
      seq_1hot[n_mask] = np.array([1,0,0,0])
      grads_ref[si] = np.abs(grads_si[seq_1hot])

  # hash gene names to indexes
  tss_df = pd.read_csv(crispr_bed_file, sep='\t', names=['chr','start','end','gene'])
  gene_i = {}
  for ti, tss in tss_df.iterrows():
    gene_i.setdefault(tss.gene,[]).append(ti)

  # score sites
  crispr_df = pd.read_csv(crispr_table_tsv, sep='\t')
  crispr_df['score'] = score_sites(crispr_df, gene_i, grads_ref, seq_starts)
  # np.save('%s/site_scores.npy' % options.out_dir, crispr_df['score'])
  crispr_df.to_csv('%s/sites_table.tsv' % options.out_dir, sep='\t')

  # compute stats
  accuracy_stats(crispr_df, options.out_dir)


def accuracy_stats(crispr_df, out_dir):
  # borzoi
  # distance_boundaries = [0, 15000, 60000, 130000, 262144]

  # original enformer
  # distance_boundaries = [0, 3000, 12000, 35000, 100000]

  # better enformer
  distance_boundaries = [0, 5000, 20000, 40000, 98000]

  acc_out = open('%s/acc.txt' % out_dir, 'w')
  cols = ['distance', 'sites', 'regulated', 'enformer_auroc', 'enformer_auprc', 'tss_auroc', 'tss_auprc']
  print('\t'.join(cols), file=acc_out)

  for di in range(len(distance_boundaries)-1):
    distance_mask = (crispr_df.tss_distance > distance_boundaries[di])
    distance_mask &= (crispr_df.tss_distance <= distance_boundaries[di+1])
    dcrispr_df = crispr_df[distance_mask]

    nan_mask = np.isnan(dcrispr_df.score)
    dcrispr_df = dcrispr_df[~nan_mask]

    num_sites = dcrispr_df.shape[0]
    num_pos = dcrispr_df.regulate.sum()

    auroc_borzoi = roc_auc_score(dcrispr_df.regulate, dcrispr_df.score)
    auprc_borzoi = average_precision_score(dcrispr_df.regulate, dcrispr_df.score)

    auroc_tss = roc_auc_score(dcrispr_df.regulate, 1/dcrispr_df.tss_distance)
    auprc_tss = average_precision_score(dcrispr_df.regulate, 1/dcrispr_df.tss_distance)

    cols = [str(distance_boundaries[di+1]), str(num_sites), str(num_pos)]
    cols += ['%.4f' % auroc_borzoi, '%.4f' % auprc_borzoi]
    cols += ['%.4f' % auroc_tss, '%.4f' % auprc_tss]
    print('\t'.join(cols), file=acc_out)

  acc_out.close()


def complete_h5(h5_file):
  if os.path.isfile(h5_file):
    try:
      h5_open = h5py.File(h5_file, 'r')
      grads = h5_open['grads'][:]
      h5_open.close()
      if (grads == 0).all():
        return False
      else:
        return True
    except:
      return False
  else:
    return False


def options_string(options, group_options):
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

    options_str += ' %s %s' % (opt_str, opt_value)

  return options_str


def score_sites(crispr_df, gene_i, grads_ref, seq_starts, sigma=200, normalize=True):
  # flowfish likes an even smaller sigma, e.g. 100

  enh_ref = np.zeros(crispr_df.shape[0])
  seq_len = grads_ref.shape[1]

  # define smoothing helpers
  window = np.arange(-4*sigma, 4*sigma+1)
  weights = norm.pdf(window, scale=sigma)
  
  for ei, enh in crispr_df.iterrows():
    if enh.gene not in gene_i:
      print('WARNING: %s skipped' % enh.gene)
      enh_ref[ei] = np.nan
    else:
      for gi in gene_i[enh.gene]:
        gene_start = seq_starts[gi]

        """ original
        # extension boundaries
        emid = (enh.start + enh.end) // 2
        enh_start = emid - elen_ext//2
        enh_end = enh_start + elen_ext

        # map to gene sequence
        eg_start = enh_start - gene_start
        eg_end = enh_end - gene_start
        
        if eg_end < 0:
          # enhancer completely left of sequence
          pass
        elif eg_start > seq_len:
          # enhancer completely right of sequence
          pass
        else:
          # clip edges
          eg_start = max(eg_start, 0)
          eg_end = min(eg_end, seq_len)
          
          # take mean
          escore = grads_ref[gi,eg_start:eg_end].mean(dtype='float32')
          if normalize:
            escore /= grads_ref[gi].mean(dtype='float32')
        """

        emid = (enh.start + enh.end) // 2
        eg_mid = emid - gene_start
        
        if eg_mid >= 0 and eg_mid < seq_len:
          # determine smoothing window
          eg_window = window + eg_mid
          wstart, wend = 0, seq_len
          if eg_window[0] < 0:
            wstart = -eg_window[0]
          if eg_window[-1] > seq_len:
            wend = seq_len - eg_window[-1] - 1
          eg_window = eg_window[wstart:wend]

          # pull scores
          grads_ref_gi = grads_ref[gi].astype('float32')
          weights_trunc = weights[wstart:wend]
          escore = np.sum(weights_trunc*grads_ref_gi[eg_window]) / np.sum(weights_trunc)
          if normalize:
            escore /= grads_ref_gi.mean()

          # collect
          enh_ref[ei] += escore
              
  return enh_ref


def trim_dot(gene_id):
  dot_i = gene_id.rfind('.')
  if dot_i == -1:
    return gene_id
  else:
    return gene_id[:dot_i]

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
