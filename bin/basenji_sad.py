#!/usr/bin/env python
# Copyright 2017 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
from __future__ import print_function

from optparse import OptionParser
import json
import pdb
import pickle
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd
import pysam
from scipy.sparse import dok_matrix
from scipy.special import rel_entr
import tensorflow as tf
from tqdm import tqdm

from basenji import seqnn
from basenji import stream
from basenji import vcf as bvcf
from borzoi_sed import targets_prep_strand

'''
basenji_sad.py

Compute SNP Activity Difference (SAD) scores for SNPs in a VCF file.
'''

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <model_file> <vcf_file>'
  parser = OptionParser(usage)
  parser.add_option('-f', dest='genome_fasta',
      default='%s/data/hg38.fa' % os.environ['BASENJIDIR'],
      help='Genome FASTA for sequences [Default: %default]')
  parser.add_option('-o',dest='out_dir',
      default='sad',
      help='Output directory for tables and plots [Default: %default]')
  parser.add_option('-p', dest='processes',
      default=None, type='int',
      help='Number of processes, passed by multi script')
  parser.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Average forward and reverse complement predictions [Default: %default]')
  parser.add_option('--shifts', dest='shifts',
      default='0', type='str',
      help='Ensemble prediction shifts [Default: %default]')
  parser.add_option('--stats', dest='sad_stats',
      default='SAD',
      help='Comma-separated list of stats to save. [Default: %default]')
  parser.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')
  (options, args) = parser.parse_args()

  if len(args) == 3:
    # single worker
    params_file = args[0]
    model_file = args[1]
    vcf_file = args[2]

  elif len(args) == 4:
    # multi separate
    options_pkl_file = args[0]
    params_file = args[1]
    model_file = args[2]
    vcf_file = args[3]

    # save out dir
    out_dir = options.out_dir

    # load options
    options_pkl = open(options_pkl_file, 'rb')
    options = pickle.load(options_pkl)
    options_pkl.close()

    # update output directory
    options.out_dir = out_dir

  elif len(args) == 5:
    # multi worker
    options_pkl_file = args[0]
    params_file = args[1]
    model_file = args[2]
    vcf_file = args[3]
    worker_index = int(args[4])

    # load options
    options_pkl = open(options_pkl_file, 'rb')
    options = pickle.load(options_pkl)
    options_pkl.close()

    # update output directory
    options.out_dir = '%s/job%d' % (options.out_dir, worker_index)

  else:
    parser.error('Must provide parameters and model files and QTL VCF file')

  if not os.path.isdir(options.out_dir):
    os.mkdir(options.out_dir)

  options.shifts = [int(shift) for shift in options.shifts.split(',')]
  options.sad_stats = options.sad_stats.split(',')

  #################################################################
  # read parameters and targets

  # read model parameters
  with open(params_file) as params_open:
    params = json.load(params_open)
  params_model = params['model']
  params_train = params['train']

  if options.targets_file is None:
    target_slice = None
    sum_strand = False
  else:
    targets_df = pd.read_csv(options.targets_file, sep='\t', index_col=0)
    target_slice = targets_df.index

    if 'strand_pair' in targets_df.columns:
      sum_strand = True

      # prep strand
      targets_strand_df = targets_prep_strand(targets_df)

      # set strand pairs
      params_model['strand_pair'] = [np.array(targets_df.strand_pair)]

      # construct strand sum transform
      strand_transform = dok_matrix((targets_df.shape[0], targets_strand_df.shape[0]))
      sti = 0
      for ti, target in targets_df.iterrows():
          strand_transform[ti,sti] = True
          if target.strand_pair == target.name:
              sti += 1
          else:
              if target.identifier[-1] == '-':
                  sti += 1
      strand_transform = strand_transform.tocsr()

    else:
      targets_strand_df = targets_df
      sum_strand = False

  #################################################################
  # setup model

  # can we sum on GPU?
  length_stats = set(['SAX','SAXR','SAR','ALT','REF'])
  sum_length = length_stats.isdisjoint(set(options.sad_stats))
  # sum_length = False

  seqnn_model = seqnn.SeqNN(params_model)
  seqnn_model.restore(model_file)
  seqnn_model.build_slice(target_slice)
  if sum_length:
    seqnn_model.build_sad()
  seqnn_model.build_ensemble(options.rc, options.shifts)

  targets_length = seqnn_model.target_lengths[0]
  num_targets = seqnn_model.num_targets()
  if options.targets_file is None:
    target_ids = ['t%d' % ti for ti in range(num_targets)]
    target_labels = ['']*len(target_ids)
    targets_strand_df = pd.DataFrame({
      'identifier':target_ids,
      'description':target_labels})

  #################################################################
  # load SNPs

  # filter for worker SNPs
  if options.processes is not None:
    # determine boundaries
    num_snps = bvcf.vcf_count(vcf_file)
    worker_bounds = np.linspace(0, num_snps, options.processes+1, dtype='int')

    # read SNPs form VCF
    snps = bvcf.vcf_snps(vcf_file, start_i=worker_bounds[worker_index],
      end_i=worker_bounds[worker_index+1])

  else:
    # read SNPs form VCF
    snps = bvcf.vcf_snps(vcf_file)

  # open genome FASTA
  genome_open = pysam.Fastafile(options.genome_fasta)

  #################################################################
  # predict SNP scores, write output

  # setup output
  sad_out = initialize_output_h5(options.out_dir, options.sad_stats,
                                 snps, targets_length, targets_strand_df)

  for si, snp in tqdm(enumerate(snps), total=len(snps)):
    # get SNP sequences
    snp_1hot_list = bvcf.snp_seq1(snp, params_model['seq_length'], genome_open)
    snps_1hot = np.array(snp_1hot_list)

    # get predictions
    snp_preds = seqnn_model(np.array(snps_1hot))
    ref_preds, alt_preds = snp_preds[0], snp_preds[1]

    # sum strand pairs
    if sum_strand:
      ref_preds = ref_preds * strand_transform
      alt_preds = alt_preds * strand_transform
      # ref_preds = sum_stranded(ref_preds, strand_transform)
      # alt_preds = sum_stranded(alt_preds, strand_transform)

    # process SNP
    if sum_length:
      write_snp(ref_preds, alt_preds, sad_out, si,
                options.sad_stats)
    else:
      write_snp_len(ref_preds, alt_preds, sad_out, si,
                    options.sad_stats)

  # close genome
  genome_open.close()

  ###################################################
  # compute SAD distributions across variants

  write_pct(sad_out, options.sad_stats)
  sad_out.close()


def initialize_output_h5(out_dir, sad_stats, snps, targets_length, targets_df):
  """Initialize an output HDF5 file for SAD stats."""

  num_targets = targets_df.shape[0]
  num_snps = len(snps)

  sad_out = h5py.File('%s/sad.h5' % out_dir, 'w')

  # write SNPs
  snp_ids = np.array([snp.rsid for snp in snps], 'S')
  sad_out.create_dataset('snp', data=snp_ids)

  # write SNP chr
  snp_chr = np.array([snp.chr for snp in snps], 'S')
  sad_out.create_dataset('chr', data=snp_chr)

  # write SNP pos
  snp_pos = np.array([snp.pos for snp in snps], dtype='uint32')
  sad_out.create_dataset('pos', data=snp_pos)

  # check flips
  snp_flips = [snp.flipped for snp in snps]

  # write SNP reference allele
  snp_refs = []
  snp_alts = []
  for snp in snps:
    if snp.flipped:
      snp_refs.append(snp.alt_alleles[0])
      snp_alts.append(snp.ref_allele)
    else:
      snp_refs.append(snp.ref_allele)
      snp_alts.append(snp.alt_alleles[0])
  snp_refs = np.array(snp_refs, 'S')
  snp_alts = np.array(snp_alts, 'S')
  sad_out.create_dataset('ref_allele', data=snp_refs)
  sad_out.create_dataset('alt_allele', data=snp_alts)

  # write targets
  sad_out.create_dataset('target_ids', data=np.array(targets_df.identifier, 'S'))
  sad_out.create_dataset('target_labels', data=np.array(targets_df.description, 'S'))

  # initialize SAD stats
  for sad_stat in sad_stats:
    if sad_stat in ['REF','ALT']:
      sad_out.create_dataset(sad_stat,
        shape=(num_snps, targets_length, num_targets),
        dtype='float16')
    else:      
      sad_out.create_dataset(sad_stat,
        shape=(num_snps, num_targets),
        dtype='float16')

  return sad_out


# def sum_stranded(preds, targets_df):
#     un_mask = (targets_df['strand'] == '.')
#     pos_mask = (targets_df['strand'] == '+')
#     neg_mask = (targets_df['strand'] == '-')

#     unpos_weights = np.zeros(targets_df.shape[0])
#     unpos_weights[un_mask] = 0.5
#     unpos_weights[pos_mask] = 1.0

#     unneg_weights = np.zeros(targets_df.shape[0])
#     unneg_weights[un_mask] = 0.5
#     unneg_weights[neg_mask] = 1.0

#     preds_sum = np.dot(preds, unpos_weights) + np.dot(preds, unneg_weights)
#     return preds_sum

def sum_stranded(preds, strand_transform):
  preds_sum = preds * strand_transform
  pdb.set_trace()
  return preds_sum

def write_pct(sad_out, sad_stats):
  """Compute percentile values for each target and write to HDF5."""

  # define percentiles
  d_fine = 0.001
  d_coarse = 0.01
  percentiles_neg = np.arange(d_fine, 0.1, d_fine)
  percentiles_base = np.arange(0.1, 0.9, d_coarse)
  percentiles_pos = np.arange(0.9, 1, d_fine)

  percentiles = np.concatenate([percentiles_neg, percentiles_base, percentiles_pos])
  sad_out.create_dataset('percentiles', data=percentiles)
  pct_len = len(percentiles)

  for sad_stat in sad_stats:
    if sad_stat not in ['REF','ALT']:
      sad_stat_pct = '%s_pct' % sad_stat

      # compute
      sad_pct = np.percentile(sad_out[sad_stat], 100*percentiles, axis=0).T
      sad_pct = sad_pct.astype('float16')

      # save
      sad_out.create_dataset(sad_stat_pct, data=sad_pct, dtype='float16')

    
def write_snp(ref_preds_sum, alt_preds_sum, sad_out, si, sad_stats):
  """Write SNP predictions to HDF, assuming the length dimension has
      been collapsed."""

  # compare reference to alternative via mean subtraction
  if 'SAD' in sad_stats:
    sad = alt_preds_sum - ref_preds_sum
    sad_out['SAD'][si,:] = sad.astype('float16')

  # compare reference to alternative via mean log division
  # if 'SADR' in sad_stats:
  #   sar = np.log2(alt_preds_sum + log_pseudo) \
  #                  - np.log2(ref_preds_sum + log_pseudo)
  #   sad_out['SADR'][si,:] = sar.astype('float16')


def write_snp_len(ref_preds, alt_preds, sad_out, si, sad_stats):
  """Write SNP predictions to HDF, assuming the length dimension has
      been maintained."""

  ref_preds = ref_preds.astype('float64')
  alt_preds = alt_preds.astype('float64')
  seq_length, num_targets = ref_preds.shape

  # compute pseudocounts
  pseudocounts = np.percentile(ref_preds, 25, axis=0)

  # sum across length
  ref_preds_sum = ref_preds.sum(axis=0)
  alt_preds_sum = alt_preds.sum(axis=0)

  # difference
  altref_diff = alt_preds - ref_preds

  # compare reference to alternative via mean subtraction
  if 'SAD' in sad_stats:
    sad = alt_preds_sum - ref_preds_sum
    sad_out['SAD'][si] = sad.astype('float16')

  # compare reference to alternative via max subtraction
  if 'SAX' in sad_stats:
    # sad_vec = (alt_preds - ref_preds)
    # max_i = np.argmax(np.abs(sad_vec), axis=0)
    # sax = sad_vec[max_i, np.arange(num_targets)]

    sad_max = np.max(altref_diff, axis=0)
    sad_min = np.min(altref_diff, axis=0)
    sax = np.where(sad_max>-sad_min, sad_max, sad_min)
    pdb.set_trace()
    sad_out['SAX'][si] = sax.astype('float16')

  # compare reference to alternative via mean log division
  if 'SADR' in sad_stats:
    sar = np.log2(alt_preds_sum + pseudocounts) \
                   - np.log2(ref_preds_sum + pseudocounts)
    sad_out['SADR'][si] = sar.astype('float16')

  # compare reference to alternative via max subtraction
  if 'SAXR' in sad_stats:
    sar_vec = np.log2(alt_preds + pseudocounts) \
                - np.log2(ref_preds + pseudocounts)
    max_i = np.argmax(np.abs(sar_vec), axis=0)
    saxr = sar_vec[max_i, np.arange(num_targets)]
    sad_out['SAXR'][si] = saxr.astype('float16')

  # compare geometric means
  if 'SAR' in sad_stats:
    sar_vec = np.log2(alt_preds + pseudocounts) \
                - np.log2(ref_preds + pseudocounts)
    geo_sad = sar_vec.sum(axis=0)
    sad_out['SAR'][si] = geo_sad.astype('float16')

  # L1 norm of difference vector
  if 'D1' in sad_stats:
    sad_vec = np.abs(altref_diff)
    sad_d1 = sad_vec.sum(axis=0)
    sad_out['D1'][si] = sad_d1.astype('float16')

  # L2 norm of difference vector
  if 'D2' in sad_stats:
    sad_vec = np.power(altref_diff, 2)
    sad_d2 = sad_vec.sum(axis=0)
    sad_out['D2'][si] = sad_d2.astype('float16')

  if 'JS' in sad_stats:
    # normalized scores
    ref_preds_norm = ref_preds + pseudocounts
    ref_preds_norm /= ref_preds_norm.sum(axis=0)
    alt_preds_norm = alt_preds + pseudocounts
    alt_preds_norm /= alt_preds_norm.sum(axis=0)

    # compare normalized JS
    ref_alt_entr = rel_entr(ref_preds_norm, alt_preds_norm).sum(axis=0)
    alt_ref_entr = rel_entr(alt_preds_norm, ref_preds_norm).sum(axis=0)
    js_dist = (ref_alt_entr + alt_ref_entr) / 2
    sad_out['JS'][si] = js_dist.astype('float16')

  # predictions
  if 'REF' in sad_stats:
    sad_out['REF'][si] = ref_preds.astype('float16')
  if 'ALT' in sad_stats:
    sad_out['ALT'][si] = alt_preds.astype('float16')


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
