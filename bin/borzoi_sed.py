#!/usr/bin/env python
# Copyright 2022 Calico LLC
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
from collections import OrderedDict
import json
import pickle
import os
import pdb
import sys
import time
from tqdm import tqdm

import h5py
import numpy as np
import pandas as pd
import pybedtools
import pysam
from scipy.special import rel_entr
import tensorflow as tf

from basenji import gene as bgene
from basenji import seqnn
from basenji import stream
from basenji import vcf as bvcf
from basenji_sad import untransform_preds, untransform_preds1
'''
borzoi_sed.py

Compute SNP Expression Difference (SED) scores for SNPs in a VCF file,
relative to gene exons in a GTF file.
'''

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <model_file> <vcf_file>'
  parser = OptionParser(usage)
  parser.add_option('-b', dest='bedgraph',
      default=False, action='store_true',
      help='Write ref/alt predictions as bedgraph [Default: %default]')
  parser.add_option('-f', dest='genome_fasta',
      default='%s/data/hg38.fa' % os.environ['BASENJIDIR'],
      help='Genome FASTA for sequences [Default: %default]')
  parser.add_option('-g', dest='genes_gtf',
      default='%s/genes/gencode41/gencode41_basic_nort.gtf' % os.environ['HG38'],
      help='GTF for gene definition [Default %default]')
  parser.add_option('-o',dest='out_dir',
      default='sed',
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
  parser.add_option('--span', dest='span',
      default=False, action='store_true',
      help='Aggregate entire gene span [Default: %default]')
  parser.add_option('--stats', dest='sed_stats',
      default='SED',
      help='Comma-separated list of stats to save. [Default: %default]')
  parser.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')
  parser.add_option('-u', dest='untransform_old',
      default=False, action='store_true')
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
    parser.error('Must provide parameters/model, VCF, and genes GTF')

  if not os.path.isdir(options.out_dir):
    os.mkdir(options.out_dir)

  options.shifts = [int(shift) for shift in options.shifts.split(',')]
  options.sed_stats = options.sed_stats.split(',')

  #################################################################
  # read parameters and targets

  # read model parameters
  with open(params_file) as params_open:
    params = json.load(params_open)
  params_model = params['model']
  params_train = params['train']
  seq_len = params_model['seq_length']

  if options.targets_file is None:
    parser.error('Must provide targets table to properly handle strands.')
  else:
    targets_df = pd.read_csv(options.targets_file, sep='\t', index_col=0)

  # prep strand
  targets_strand_df = targets_prep_strand(targets_df)

  # set strand pairs (using new indexing)
  orig_new_index = dict(zip(targets_df.index, np.arange(targets_df.shape[0])))
  targets_strand_pair = np.array([orig_new_index[ti] for ti in targets_df.strand_pair])
  params_model['strand_pair'] = [targets_strand_pair]

  #################################################################
  # setup model

  seqnn_model = seqnn.SeqNN(params_model)
  seqnn_model.restore(model_file)
  seqnn_model.build_slice(targets_df.index)
  seqnn_model.build_ensemble(options.rc, options.shifts)

  model_stride = seqnn_model.model_strides[0]
  out_seq_len = seqnn_model.target_lengths[0]*model_stride

  #################################################################
  # read SNPs / genes

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

  # read genes
  transcriptome = bgene.Transcriptome(options.genes_gtf)
  gene_strand = {}
  for gene_id, gene in transcriptome.genes.items():
    gene_strand[gene_id] = gene.strand

  # map SNP sequences to gene positions
  snpseq_gene_slice = map_snpseq_genes(snps, out_seq_len, transcriptome, model_stride, options.span)

  # remove SNPs w/o genes
  num_snps_pre = len(snps)
  snp_gene_mask = np.array([len(sgs) > 0 for sgs in snpseq_gene_slice])
  snps = [snps[si] for si in range(num_snps_pre) if snp_gene_mask[si]]
  snpseq_gene_slice = [snpseq_gene_slice[si] for si in range(num_snps_pre) if snp_gene_mask[si]]
  num_snps = len(snps)

  #################################################################
  # setup output

  sed_out = initialize_output_h5(options.out_dir, options.sed_stats, snps,
                                 snpseq_gene_slice, targets_strand_df)

  #################################################################
  # predict SNP scores, write output

  # create SNP seq generator
  genome_open = pysam.Fastafile(options.genome_fasta)

  # SNP/gene index
  xi = 0

  # for each SNP sequence
  for si, snp in tqdm(enumerate(snps), total=len(snps)):
    # get SNP sequences
    snp_1hot_list = bvcf.snp_seq1(snp, seq_len, genome_open)
    snps_1hot = np.array(snp_1hot_list)

    # get predictions
    if params_train['batch_size'] == 1:
      ref_preds = seqnn_model(snps_1hot[:1])[0]
      alt_preds = seqnn_model(snps_1hot[1:])[0]
    else:
      snp_preds = seqnn_model(snps_1hot)
      ref_preds, alt_preds = snp_preds[0], snp_preds[1]

     # untransform predictions
    if options.targets_file is not None:
      if options.untransform_old:
        ref_preds = untransform_preds1(ref_preds, targets_df)
        alt_preds = untransform_preds1(alt_preds, targets_df)
      else:
        ref_preds = untransform_preds(ref_preds, targets_df)
        alt_preds = untransform_preds(alt_preds, targets_df)

    if options.bedgraph:
      write_bedgraph_snp(snps[si], ref_preds, alt_preds, options.out_dir, model_stride)

    # for each overlapping gene
    for gene_id, gene_slice in snpseq_gene_slice[si].items():
      if len(gene_slice) > len(set(gene_slice)):
        print('WARNING: %d %s has overlapping bins' % (si,gene_id))

      # slice gene positions
      ref_preds_gene = ref_preds[gene_slice]
      alt_preds_gene = alt_preds[gene_slice]

      # slice relevant strand targets
      if gene_strand[gene_id] == '+':
        gene_strand_mask = (targets_df.strand != '-')
      else:
        gene_strand_mask = (targets_df.strand != '+')
      ref_preds_gene = ref_preds_gene[...,gene_strand_mask]
      alt_preds_gene = alt_preds_gene[...,gene_strand_mask]

      # compute pseudocounts
      ref_preds_strand = ref_preds[...,gene_strand_mask]
      pseudocounts = np.percentile(ref_preds_strand, 25, axis=0)

      # write scores to HDF
      write_snp(ref_preds_gene, alt_preds_gene, sed_out, xi,
        options.sed_stats, pseudocounts)

      xi += 1

  # close genome
  genome_open.close()

  ###################################################
  # compute SAD distributions across variants

  # write_pct(sed_out, options.sed_stats)
  sed_out.close()


def clip_float(x, dtype=np.float16):
  return np.clip(x, np.finfo(dtype).min, np.finfo(dtype).max)


def initialize_output_h5(out_dir: str, sed_stats, snps, snpseq_gene_slice, targets_df):
  """Initialize an output HDF5 file for SAD stats.
  
  Args:
      out_dir (str): Output directory.
      sed_stats (list): List of SAD stats to compute.
      snps ([bvcf.SNP]): SNP list.
      snpseq_gene_slice ([dict]): List of dicts mapping gene_ids
        to their exon-overlapping positions for each sequence.
      targets_df (pandas.DataFrame): Targets table.
  """
  sed_out = h5py.File('%s/sed.h5' % out_dir, 'w')

  # collect identifier tuples
  snp_indexes = []
  gene_ids = []
  snp_ids = []
  for si, gene_slice in enumerate(snpseq_gene_slice):
    snp_genes = list(gene_slice.keys())
    gene_ids += snp_genes
    snp_indexes += [si]*len(snp_genes)
  num_scores = len(snp_indexes)

  # write SNP indexes
  snp_indexes = np.array(snp_indexes)
  sed_out.create_dataset('si', data=snp_indexes)

  # write genes
  gene_ids = np.array(gene_ids, 'S')
  sed_out.create_dataset('gene', data=gene_ids)

  # write SNPs
  snp_ids = np.array([snp.rsid for snp in snps], 'S')
  sed_out.create_dataset('snp', data=snp_ids)

  # write SNP chr
  snp_chr = np.array([snp.chr for snp in snps], 'S')
  sed_out.create_dataset('chr', data=snp_chr)

  # write SNP pos
  snp_pos = np.array([snp.pos for snp in snps], dtype='uint32')
  sed_out.create_dataset('pos', data=snp_pos)

  # write SNP reference allele
  snp_refs = []
  snp_alts = []
  for snp in snps:
    if snp.flipped:
      print('SNP %s is flipped. How did that happen?' % snp.rsid)
      snp_refs.append(snp.alt_alleles[0])
      snp_alts.append(snp.ref_allele)
    else:
      snp_refs.append(snp.ref_allele)
      snp_alts.append(snp.alt_alleles[0])
  snp_refs = np.array(snp_refs, 'S')
  snp_alts = np.array(snp_alts, 'S')
  sed_out.create_dataset('ref_allele', data=snp_refs)
  sed_out.create_dataset('alt_allele', data=snp_alts)

  # write targets
  sed_out.create_dataset('target_ids', data=np.array(targets_df.identifier, 'S'))
  sed_out.create_dataset('target_labels', data=np.array(targets_df.description, 'S'))

  # initialize SED stats
  num_targets = targets_df.shape[0]
  for sed_stat in sed_stats:
    sed_out.create_dataset(sed_stat,
      shape=(num_scores, num_targets),
      dtype='float16')

  return sed_out


def make_snpseq_bedt(snps, seq_len: int):
  """Make a BedTool object for all SNP sequences, where seq_len considers cropping."""
  num_snps = len(snps)
  left_len = seq_len // 2
  right_len = seq_len // 2
 
  snpseq_bed_lines = []
  for si in range(num_snps):
    # bound sequence start at 0 (true sequence will be N padded)
    snpseq_start = max(0, snps[si].pos - left_len)
    snpseq_end = snps[si].pos + right_len
    # correct end for alternative indels
    snpseq_end += max(0, len(snps[si].ref_allele) - snps[si].longest_alt())
    snpseq_bed_lines.append('%s %d %d %d' % (snps[si].chr, snpseq_start, snpseq_end, si))

  snpseq_bedt = pybedtools.BedTool('\n'.join(snpseq_bed_lines), from_string=True)
  return snpseq_bedt


def map_snpseq_genes(
    snps,
    seq_len: int,
    transcriptome,
    model_stride: int,
    span: bool,
    majority_overlap: bool=True,
    intron1: bool=False):
  """Intersect SNP sequences with gene exons, constructing a list
     mapping sequence indexes to dictionaries of gene_ids to their
     exon-overlapping positions in the sequence.
     
     Args:
        snps ([bvcf.SNP]): SNP list.
        seq_len (int): Sequence length, after model cropping.
        transcriptome (bgene.Transcriptome): Transcriptome.
        model_stride (int): Model stride.
        span (bool): If True, use gene span instead of exons.
        majority_overlap (bool): If True, only consider bins for which
          the majority of the space overlaps an exon.
        intron1 (bool): If True, include intron bins adjacent to junctions.
     """

  # make gene BEDtool
  if span:
    genes_bedt = transcriptome.bedtool_span()
  else:
    genes_bedt = transcriptome.bedtool_exon()

  # make SNP sequence BEDtool
  snpseq_bedt = make_snpseq_bedt(snps, seq_len)

  # map SNPs to genes
  snpseq_gene_slice = []
  for snp in snps:
    snpseq_gene_slice.append(OrderedDict())

  for overlap in genes_bedt.intersect(snpseq_bedt, wo=True):
    gene_id = overlap[3]
    gene_start = int(overlap[1])
    gene_end = int(overlap[2])
    seq_start = int(overlap[7])
    seq_end = int(overlap[8])
    si = int(overlap[9])

    # adjust for left overhang padded
    seq_len_chop = seq_end - seq_start
    seq_start -= (seq_len - seq_len_chop)

    # clip left boundaries
    gene_seq_start = max(0, gene_start - seq_start)
    gene_seq_end = max(0, gene_end - seq_start)

    if majority_overlap:
      # requires >50% overlap
      bin_start = int(np.round(gene_seq_start / model_stride))
      bin_end = int(np.round(gene_seq_end / model_stride))
    else:
      # any overlap
      bin_start = int(np.floor(gene_seq_start / model_stride))
      bin_end = int(np.ceil(gene_seq_end / model_stride))

    if intron1:
      bin_start -= 1
      bin_end += 1

    # clip boundaries
    bin_max = int(seq_len/model_stride)
    bin_start = min(bin_start, bin_max)
    bin_end = min(bin_end, bin_max)
    bin_start = max(0, bin_start)
    bin_end = max(0, bin_end)

    if bin_end - bin_start > 0:
      # save gene bin positions
      snpseq_gene_slice[si].setdefault(gene_id,[]).extend(range(bin_start, bin_end))

  # handle possible overlaps
  for si in range(len(snps)):
    for gene_id, gene_slice in snpseq_gene_slice[si].items():
      snpseq_gene_slice[si][gene_id] = np.unique(gene_slice)

  return snpseq_gene_slice


def targets_prep_strand(targets_df):
  """Adjust targets table for merged stranded datasets."""
  # attach strand
  targets_strand = []
  for _, target in targets_df.iterrows():
    if target.strand_pair == target.name:
      targets_strand.append('.')
    else:
      targets_strand.append(target.identifier[-1])
  targets_df['strand'] = targets_strand

  # collapse stranded
  strand_mask = (targets_df.strand != '-')
  targets_strand_df = targets_df[strand_mask]

  return targets_strand_df


def write_pct(sed_out, sed_stats):
  """Compute percentile values for each target and write to HDF5."""
  # define percentiles
  d_fine = 0.001
  d_coarse = 0.01
  percentiles_neg = np.arange(d_fine, 0.1, d_fine)
  percentiles_base = np.arange(0.1, 0.9, d_coarse)
  percentiles_pos = np.arange(0.9, 1, d_fine)

  percentiles = np.concatenate([percentiles_neg, percentiles_base, percentiles_pos])
  sed_out.create_dataset('percentiles', data=percentiles)
  pct_len = len(percentiles)

  for sad_stat in sed_stats:
    if sad_stat not in ['REF','ALT']:
      sad_stat_pct = '%s_pct' % sad_stat

      # compute
      sad_pct = np.percentile(sed_out[sad_stat], 100*percentiles, axis=0).T
      sad_pct = sad_pct.astype('float16')

      # save
      sed_out.create_dataset(sad_stat_pct, data=sad_pct, dtype='float16')


def write_bedgraph_snp(snp, ref_preds, alt_preds, out_dir: str, model_stride: int):
  """Write full predictions around SNP as BedGraph.
  
  Args:
    snp (bvcf.SNP): SNP.
    ref_preds (np.ndarray): Reference predictions.
    alt_preds (np.ndarray): Alternate predictions.
    out_dir (str): Output directory.
    model_stride (int): Model stride.
  """
  target_length, num_targets = ref_preds.shape

  # mean across targets
  ref_preds = ref_preds.mean(axis=-1, dtype='float32')
  alt_preds = alt_preds.mean(axis=-1, dtype='float32')
  diff_preds = alt_preds - ref_preds

  # initialize raw predictions/targets
  ref_out = open('%s/%s_ref.bedgraph' % (out_dir, snp.rsid), 'w')
  alt_out = open('%s/%s_alt.bedgraph' % (out_dir, snp.rsid), 'w')
  diff_out = open('%s/%s_diff.bedgraph' % (out_dir, snp.rsid), 'w')

  # specify positions
  seq_len = target_length * model_stride
  left_len = seq_len // 2 - 1
  right_len = seq_len // 2
  seq_start = snp.pos - left_len - 1
  seq_end = snp.pos + right_len + max(0,
                                      len(snp.ref_allele) - snp.longest_alt())

  # write values
  bin_start = seq_start
  for bi in range(target_length):
    bin_end = bin_start + model_stride
    cols = [snp.chr, str(bin_start), str(bin_end), str(ref_preds[bi])]
    print('\t'.join(cols), file=ref_out)
    cols = [snp.chr, str(bin_start), str(bin_end), str(alt_preds[bi])]
    print('\t'.join(cols), file=alt_out)
    cols = [snp.chr, str(bin_start), str(bin_end), str(diff_preds[bi])]
    print('\t'.join(cols), file=diff_out)
    bin_start = bin_end

  ref_out.close()
  alt_out.close()
  diff_out.close()


def write_snp(ref_preds, alt_preds, sed_out, xi: int, sed_stats, pseudocounts):
  """Write SNP predictions to HDF, assuming the length dimension has
      been maintained.
      
    Args:
      ref_preds (np.ndarray): Reference predictions, (gene length x tasks)
      alt_preds (np.ndarray): Alternate predictions, (gene length x tasks)
      sed_out (h5py.File): HDF5 output file.
      xi (int): SNP index.
      sed_stats (list): SED statistics to compute.
      pseudocounts (np.ndarray): Target pseudocounts for safe logs.
    """

  # ref/alt_preds is L x T
  seq_len, num_targets = ref_preds.shape

  # sum across bins
  ref_preds_sum = ref_preds.sum(axis=0)
  alt_preds_sum = alt_preds.sum(axis=0)

  # difference of sums
  if 'SED' in sed_stats:
    sed = alt_preds_sum - ref_preds_sum
    sed_out['SED'][xi] = clip_float(sed).astype('float16')
  if 'logSED' in sed_stats:
    log_sed = np.log2(alt_preds_sum + 1) \
          - np.log2(ref_preds_sum + 1)
    sed_out['logSED'][xi] = log_sed.astype('float16')

  # difference L1 norm
  if 'D1' in sed_stats:
    diff_abs = np.abs(ref_preds - alt_preds)
    diff_norm1 = diff_abs.sum(axis=0)
    sed_out['D1'][xi] = clip_float(diff_norm1).astype('float16')
    
  # difference L2 norm
  if 'D2' in sed_stats:
    diff2 = np.power(ref_preds - alt_preds, 2)
    diff_norm2 = np.sqrt(diff2.sum(axis=0))
    sed_out['D2'][xi] = clip_float(diff_norm2).astype('float16')

  # normalized scores
  ref_preds_norm = ref_preds + pseudocounts
  ref_preds_norm /= ref_preds_norm.sum(axis=0)
  alt_preds_norm = alt_preds + pseudocounts
  alt_preds_norm /= alt_preds_norm.sum(axis=0)

  # compare normalized squared difference
  if 'nD2' in sed_stats:
    ndiff2 = np.power(ref_preds_norm - alt_preds_norm, 2)
    ndiff_norm2 = np.sqrt(ndiff2.sum(axis=0))
    sed_out['nD2'][xi] = ndiff_norm2.astype('float16')

  # compare normalized abs max
  if 'nDi' in sed_stats:
    ndiff_abs = np.abs(ref_preds_norm - alt_preds_norm)
    ndiff_normi = ndiff_abs.max(axis=0)
    sed_out['nDi'][xi] = ndiff_normi.astype('float16')

  # compare normalized JS
  if 'JS' in sed_stats:
    ref_alt_entr = rel_entr(ref_preds_norm, alt_preds_norm).sum(axis=0)
    alt_ref_entr = rel_entr(alt_preds_norm, ref_preds_norm).sum(axis=0)
    js_dist = (ref_alt_entr + alt_ref_entr) / 2
    sed_out['JS'][xi] = js_dist.astype('float16')

  # predictions
  if 'REF' in sed_stats:
    sed_out['REF'][xi] = ref_preds_sum.astype('float16')
  if 'ALT' in sed_stats:
    sed_out['ALT'][xi] = alt_preds_sum.astype('float16')


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
