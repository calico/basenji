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

import h5py
import numpy as np
import pandas as pd
import pybedtools
import pysam
from scipy.special import rel_entr
import tensorflow as tf

from basenji import dna_io
from basenji import gene as bgene
from basenji import seqnn
from basenji import stream
from basenji import vcf as bvcf

'''
borzoi_sed_ipaqtl_cov.py

Compute SNP COVerage Ratio (COVR) scores for SNPs in a VCF file,
relative to intronic polyadenylation sites in an annotation file.
'''

def eprint(*args, **kwargs):
  print(*args, file=sys.stderr, **kwargs)

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <model_file> <vcf_file>'
  parser = OptionParser(usage)
  parser.add_option('-f', dest='genome_fasta',
      default='%s/data/hg38.fa' % os.environ['BASENJIDIR'],
      help='Genome FASTA for sequences [Default: %default]')
  parser.add_option('-g', dest='genes_gtf',
      default='%s/genes/gencode41/gencode41_basic_nort.gtf' % os.environ['HG38'],
      help='GTF for gene definition [Default %default]')
  parser.add_option('--apafile', dest='apa_file',
      default='polyadb_human_v3.csv.gz')
  parser.add_option('-o',dest='out_dir',
      default='sed',
      help='Output directory for tables and plots [Default: %default]')
  parser.add_option('-p', dest='processes',
      default=None, type='int',
      help='Number of processes, passed by multi script')
  parser.add_option('--pseudo', dest='cov_pseudo',
      default=50, type='float',
      help='Coverage pseudocount [Default: %default]')
  parser.add_option('--cov', dest='cov_min',
      default=100, type='float',
      help='Min coverage [Default: %default]')
  parser.add_option('--paext', dest='pas_ext',
      default=50, type='float',
      help='Extension in bp past gene span annotation [Default: %default]')
  parser.add_option('--rc', dest='rc',
      default=True, action='store_true',
      help='Average forward and reverse complement predictions [Default: %default]')
  parser.add_option('--shifts', dest='shifts',
      default='0', type='str',
      help='Ensemble prediction shifts [Default: %default]')
  parser.add_option('--stats', dest='sed_stats',
      default='COVR,SCOVR',
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
  
  #######################################################
  # make APA BED (PolyADB)
  
  apa_df = pd.read_csv(options.apa_file, sep='\t', compression='gzip')
  
  # filter for intronic or 3' UTR polyA sites only
  apa_df = apa_df.query("site_type == '3\\' most exon' or site_type == 'Intron'").copy().reset_index(drop=True)
  apa_df = apa_df.sort_values(by=['gene', 'site_num'], ascending=True).copy().reset_index(drop=True)
  
  print("n intron sites = " + str(len(apa_df.query("site_type == 'Intron'"))), flush=True)
  print("n utr3 sites   = " + str(len(apa_df.query("site_type == '3\\' most exon'"))), flush=True)
  
  apa_df['start_hg38'] = apa_df['position_hg38']
  apa_df['end_hg38'] = apa_df['position_hg38'] + 1
  
  apa_df = apa_df.rename(columns={'chrom' : 'Chromosome', 'start_hg38' : 'Start', 'end_hg38' : 'End', 'position_hg38' : 'cut_mode', 'strand' : 'pas_strand'})
  
  apa_df = apa_df[['Chromosome', 'Start', 'End', 'pas_id', 'pas_strand', 'gene', 'site_num']]
  
  # map SNP sequences to gene / polyA signal positions
  snpseq_gene_slice, snpseq_apa_slice = map_snpseq_apa(snps, out_seq_len, transcriptome, apa_df, model_stride, options.sed_stats, options.pas_ext)

  # remove SNPs w/o genes
  num_snps_pre = len(snps)
  snp_gene_mask = np.array([len(sgs) > 0 for sgs in snpseq_gene_slice])
  snps = [snps[si] for si in range(num_snps_pre) if snp_gene_mask[si]]
  snpseq_gene_slice = [snpseq_gene_slice[si] for si in range(num_snps_pre) if snp_gene_mask[si]]
  snpseq_apa_slice = [snpseq_apa_slice[si] for si in range(num_snps_pre) if snp_gene_mask[si]]
  num_snps = len(snps)

  # create SNP seq generator
  genome_open = pysam.Fastafile(options.genome_fasta)

  def snp_gen():
    for snp in snps:
      # get SNP sequences
      snp_1hot_list = bvcf.snp_seq1(snp, seq_len, genome_open)
      for snp_1hot in snp_1hot_list:
        yield snp_1hot

  #################################################################
  # setup output

  sed_out = initialize_output_h5(options.out_dir, options.sed_stats, snps, snpseq_gene_slice, targets_strand_df, out_name='sed')
  sed_out_apa = initialize_output_h5(options.out_dir, ['REF','ALT'], snps, snpseq_apa_slice, targets_strand_df, out_name='sed_pas')

  #################################################################
  # predict SNP scores, write output

  # initialize predictions stream
  preds_stream = stream.PredStreamGen(seqnn_model, snp_gen(), params_train['batch_size'])

  # predictions index
  pi = 0

  # SNP/gene index
  xi_gene = 0
  
  # SNP/pas index
  xi_pas = 0

  # for each SNP sequence
  for si in range(num_snps):
    # get predictions
    ref_preds = preds_stream[pi]
    pi += 1
    alt_preds = preds_stream[pi]
    pi += 1
    
    # undo scale
    ref_preds /= np.expand_dims(targets_df.scale, axis=0)
    alt_preds /= np.expand_dims(targets_df.scale, axis=0)

    # undo sqrt
    ref_preds = ref_preds**(4/3)
    alt_preds = alt_preds**(4/3)

    # for each overlapping gene
    for gene_id, gene_slice_dup in snpseq_gene_slice[si]['bins'].items():
      
      # remove duplicate bin coordinates (artifact of PASs that are within <32bp)
      gene_slice = []
      gene_slice_dict = {}
      for gslpas in gene_slice_dup:
        gslpas_key = str(gslpas[0]) + "_" + str(gslpas[1])
        
        if gslpas_key not in gene_slice_dict :
          gene_slice_dict[gslpas_key] = True
          gene_slice.append(gslpas)
      
      # slice gene positions
      ref_preds_gene = np.concatenate([np.sum(ref_preds[gene_slice_start:gene_slice_end, :], axis=0)[None, :] for [gene_slice_start, gene_slice_end] in gene_slice], axis=0)
      alt_preds_gene = np.concatenate([np.sum(alt_preds[gene_slice_start:gene_slice_end, :], axis=0)[None, :] for [gene_slice_start, gene_slice_end] in gene_slice], axis=0)
      
      if gene_strand[gene_id] == '+':
        gene_strand_mask = (targets_df.strand != '-')
      else:
        gene_strand_mask = (targets_df.strand != '+')
      
      ref_preds_gene = ref_preds_gene[...,gene_strand_mask]
      alt_preds_gene = alt_preds_gene[...,gene_strand_mask]

      # write scores to HDF
      write_snp(ref_preds_gene, alt_preds_gene, sed_out, xi_gene, options.sed_stats, options.cov_pseudo, options.cov_min)

      xi_gene += 1
    
    # for each overlapping PAS
    for pas_id, pas_slice in snpseq_apa_slice[si]['bins'].items():
      if len(pas_slice) > len(set(pas_slice)):
        print('WARNING: %d %s has overlapping bins' % (si,pas_id))
        eprint('WARNING: %d %s has overlapping bins' % (si,pas_id))
        
      # slice pas positions
      ref_preds_pas = ref_preds[pas_slice]
      alt_preds_pas = alt_preds[pas_slice]
      
      # slice relevant strand targets
      if '+' in pas_id:
        pas_strand_mask = (targets_df.strand != '-')
      else:
        pas_strand_mask = (targets_df.strand != '+')
      
      ref_preds_pas = ref_preds_pas[...,pas_strand_mask]
      alt_preds_pas = alt_preds_pas[...,pas_strand_mask]

      # write scores to HDF
      write_snp(ref_preds_pas, alt_preds_pas, sed_out_apa, xi_pas, ['REF','ALT'], options.cov_pseudo, options.cov_min)

      xi_pas += 1

  # close genome
  genome_open.close()

  ###################################################
  # compute SAD distributions across variants

  # write_pct(sed_out, options.sed_stats)
  sed_out.close()
  sed_out_apa.close()

def map_snpseq_apa(snps, seq_len, transcriptome, apa_df, model_stride, sed_stats, pas_ext):
  """Intersect SNP sequences with genes and polyA sites, constructing a list
     mapping sequence indexes to dictionaries of gene_ids or pas_ids."""

  # make gene BEDtool
  genes_bedt = transcriptome.bedtool_span()
  
  # make SNP sequence BEDtool
  snpseq_bedt = make_snpseq_bedt(snps, seq_len)

  # map SNPs to genes and polyA sites
  snpseq_gene_slice = []
  snpseq_apa_slice = []
  for snp in snps:
    snpseq_gene_slice.append({ 'bins' : OrderedDict(), 'distances' : OrderedDict() })
    snpseq_apa_slice.append({ 'bins' : OrderedDict(), 'distances' : OrderedDict() })

  for i1, overlap in enumerate(genes_bedt.intersect(snpseq_bedt, wo=True)):
    gene_id = overlap[3]
    gene_chrom = overlap[0]
    gene_start = int(overlap[1])
    gene_end = int(overlap[2])
    seq_start = int(overlap[7])
    seq_end = int(overlap[8])
    si = int(overlap[9])
    
    snp_pos = snps[si].pos
    
    # get apa dataframe
    gene_apa_df = apa_df.query("Chromosome == '" + gene_chrom + "' and ((End > " + str(gene_start-pas_ext) + " and End <= " + str(gene_end+pas_ext) + ") or (Start < " + str(gene_end+pas_ext) + " and Start >= " + str(gene_start-pas_ext) + "))").sort_values(by=['gene', 'site_num'], ascending=True)
    
    # make sure 80% of all polyA signals are contained within the sequence input window
    if len(gene_apa_df) <= 0 or np.mean((gene_apa_df['Start'] >= seq_start).values) < 0.8 or np.mean((gene_apa_df['End'] < seq_end).values) < 0.8:
      continue
    
    # adjust for left overhang
    seq_len_chop = seq_end - seq_start
    seq_start -= (seq_len - seq_len_chop)
    
    for _, apa_row in gene_apa_df.iterrows():
      pas_id = apa_row['pas_id']
      pas_start = apa_row['Start']
      pas_end = apa_row['End']
      pas_strand = apa_row['pas_strand']
      
      pas_distance = int(np.abs(pas_start - snp_pos))
      
      if 'PROP3' in sed_stats and pas_id in snpseq_apa_slice[si]['bins']:
        continue
      elif pas_id + '_up' in snpseq_apa_slice[si]['bins'] or pas_id + '_dn' in snpseq_apa_slice[si]['bins']:
        continue
      
      # clip left boundaries
      pas_seq_start = max(0, pas_start - seq_start)
      pas_seq_end = max(0, pas_end - seq_start)

      # accumulate list of pas-snp distances
      snpseq_gene_slice[si]['distances'].setdefault(gene_id,[]).append(pas_distance)
      if 'PROP3' in sed_stats:
        snpseq_apa_slice[si]['distances'].setdefault(pas_id,[]).append(pas_distance)
      elif 'COVR3' in sed_stats or 'COVR3WIDE' in sed_stats:
        snpseq_apa_slice[si]['distances'].setdefault(pas_id + '_up',[]).append(pas_distance)
      else :
        snpseq_apa_slice[si]['distances'].setdefault(pas_id + '_up',[]).append(pas_distance)
        snpseq_apa_slice[si]['distances'].setdefault(pas_id + '_dn',[]).append(pas_distance)
      
      if 'PROP3' in sed_stats:
        # coverage (overlapping PAS)
        bin_end = int(np.round(pas_seq_start / model_stride)) + 3
        bin_start = bin_end - 5

        # clip right boundaries
        bin_max = int(seq_len/model_stride)
        bin_start = max(min(bin_start, bin_max), 0)
        bin_end = max(min(bin_end, bin_max), 0)

        if bin_end - bin_start > 0:
          # save gene bin positions
          snpseq_gene_slice[si]['bins'].setdefault(gene_id,[]).append([bin_start, bin_end])
          snpseq_apa_slice[si]['bins'].setdefault(pas_id,[]).extend(range(bin_start, bin_end))
      
      elif 'COVR3' in sed_stats:
        # upstream coverage (before PAS)
        bin_start = None
        bin_end = None
        if pas_strand == '+' :
          bin_end = int(np.round(pas_seq_start / model_stride)) + 1
          bin_start = bin_end - 4 - 1
        else :
          bin_start = int(np.round(pas_seq_end / model_stride))
          bin_end = bin_start + 4 + 1

        # clip right boundaries
        bin_max = int(seq_len/model_stride)
        bin_start = max(min(bin_start, bin_max), 0)
        bin_end = max(min(bin_end, bin_max), 0)

        if bin_end - bin_start > 0:
          # save gene bin positions
          snpseq_gene_slice[si]['bins'].setdefault(gene_id,[]).append([bin_start, bin_end])
          snpseq_apa_slice[si]['bins'].setdefault(pas_id + '_up',[]).extend(range(bin_start, bin_end))
      
      elif 'COVR3WIDE' in sed_stats:
        # upstream coverage (before PAS); wider
        bin_start = None
        bin_end = None
        if pas_strand == '+' :
          bin_end = int(np.round(pas_seq_start / model_stride)) + 1
          bin_start = bin_end - 9 - 1
        else :
          bin_start = int(np.round(pas_seq_end / model_stride))
          bin_end = bin_start + 9 + 1

        # clip right boundaries
        bin_max = int(seq_len/model_stride)
        bin_start = max(min(bin_start, bin_max), 0)
        bin_end = max(min(bin_end, bin_max), 0)

        if bin_end - bin_start > 0:
          # save gene bin positions
          snpseq_gene_slice[si]['bins'].setdefault(gene_id,[]).append([bin_start, bin_end])
          snpseq_apa_slice[si]['bins'].setdefault(pas_id + '_up',[]).extend(range(bin_start, bin_end))
        
      else: # default (COVR)
        # upstream coverage (before PAS)
        bin_start = None
        bin_end = None
        if pas_strand == '+' :
          bin_end = int(np.round(pas_seq_start / model_stride)) + 1
          bin_start = bin_end - 3 - 1
        else :
          bin_start = int(np.round(pas_seq_end / model_stride))
          bin_end = bin_start + 3 + 1

        # clip right boundaries
        bin_max = int(seq_len/model_stride)
        bin_start = max(min(bin_start, bin_max), 0)
        bin_end = max(min(bin_end, bin_max), 0)

        if bin_end - bin_start > 0:
          # save gene bin positions
          snpseq_gene_slice[si]['bins'].setdefault(gene_id,[]).append([bin_start, bin_end])
          snpseq_apa_slice[si]['bins'].setdefault(pas_id + '_up',[]).extend(range(bin_start, bin_end))
        
        # downstream coverage (after PAS)
        bin_start = None
        bin_end = None
        if pas_strand == '+' :
          bin_start = int(np.round(pas_seq_end / model_stride))
          bin_end = bin_start + 3 + 1
        else :
          bin_end = int(np.round(pas_seq_start / model_stride)) + 1
          bin_start = bin_end - 3 - 1

        # clip right boundaries
        bin_max = int(seq_len/model_stride)
        bin_start = max(min(bin_start, bin_max), 0)
        bin_end = max(min(bin_end, bin_max), 0)

        if bin_end - bin_start > 0:
          # save gene bin positions
          snpseq_gene_slice[si]['bins'].setdefault(gene_id,[]).append([bin_start, bin_end])
          snpseq_apa_slice[si]['bins'].setdefault(pas_id + '_dn',[]).extend(range(bin_start, bin_end))

  return snpseq_gene_slice, snpseq_apa_slice

def initialize_output_h5(out_dir, sed_stats, snps, snpseq_gene_slice, targets_df, out_name='sed'):
  """Initialize an output HDF5 file for SAD stats."""

  sed_out = h5py.File('%s/%s.h5' % (out_dir, out_name), 'w')

  # collect identifier tuples
  snp_indexes = []
  gene_ids = []
  distances = []
  ns = []
  snp_ids = []
  snp_a1 = []
  snp_a2 = []
  snp_flips = []
  for si, gene_slice in enumerate(snpseq_gene_slice):
    snp_genes = list(gene_slice['bins'].keys())
    gene_ids += snp_genes
    distances += [int(np.min(gene_slice['distances'][snp_gene])) for snp_gene in snp_genes]
    ns += [len(gene_slice['distances'][snp_gene]) for snp_gene in snp_genes]
    snp_indexes += [si]*len(snp_genes)
  num_scores = len(snp_indexes)

  # write SNP indexes
  snp_indexes = np.array(snp_indexes)
  sed_out.create_dataset('si', data=snp_indexes)

  # write genes
  gene_ids = np.array(gene_ids, 'S')
  sed_out.create_dataset('gene', data=gene_ids)

  # write distances
  distances = np.array(distances, 'int32')
  sed_out.create_dataset('distance', data=distances)

  # write number of sites
  ns = np.array(ns, 'int32')
  sed_out.create_dataset('n', data=ns)

  # write SNPs
  snp_ids = np.array([snp.rsid for snp in snps], 'S')
  sed_out.create_dataset('snp', data=snp_ids)

  # write SNP chr
  snp_chr = np.array([snp.chr for snp in snps], 'S')
  sed_out.create_dataset('chr', data=snp_chr)

  # write SNP pos
  snp_pos = np.array([snp.pos for snp in snps], dtype='uint32')
  sed_out.create_dataset('pos', data=snp_pos)

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


def make_snpseq_bedt(snps, seq_len):
  """Make a BedTool object for all SNP sequences."""
  num_snps = len(snps)
  left_len = seq_len // 2
  right_len = seq_len // 2
 
  snpseq_bed_lines = []
  for si in range(num_snps):
    snpseq_start = max(0, snps[si].pos - left_len)
    snpseq_end = snps[si].pos + right_len
    snpseq_end += max(0, len(snps[si].ref_allele) - snps[si].longest_alt())
    snpseq_bed_lines.append('%s %d %d %d' % (snps[si].chr, snpseq_start, snpseq_end, si))

  snpseq_bedt = pybedtools.BedTool('\n'.join(snpseq_bed_lines), from_string=True)
  return snpseq_bedt


def targets_prep_strand(targets_df):
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


def write_snp(ref_preds, alt_preds, sed_out, xi, sed_stats, cov_pseudo, cov_min):
  """Write SNP predictions to HDF, assuming the length dimension has
      been maintained."""

  # ref/alt_preds is L x T
  ref_preds = ref_preds.astype('float64')
  alt_preds = alt_preds.astype('float64')
  seq_len, num_targets = ref_preds.shape

  # sum across bins
  ref_preds_sum = ref_preds.sum(axis=0)
  alt_preds_sum = alt_preds.sum(axis=0)

  # compare reference to alternative via mean downstream/upstream coverage ratios
  if 'COVR' in sed_stats:
    
    cov_vec = (alt_preds + cov_pseudo) / (ref_preds + cov_pseudo)
    
    if np.sum(np.mean(ref_preds, axis=1) > cov_min) >= 1 :
        cov_vec = cov_vec[np.mean(ref_preds, axis=1) > cov_min, :]
    
    cov_vec = np.concatenate([
      np.ones((1, cov_vec.shape[1])),
      cov_vec,
      np.ones((1, cov_vec.shape[1])),
    ], axis=0)
    
    max_scores = np.zeros(cov_vec.shape[1])
    for j in range(1, cov_vec.shape[0]) :
      avg_up = np.mean(cov_vec[:j, :], axis=0)
      avg_dn = np.mean(cov_vec[j:, :], axis=0)
        
      scores = np.abs(np.log2(avg_dn / avg_up))
        
      if np.mean(scores) > np.mean(max_scores) :
        max_scores = scores
      
    sed_out['COVR'][xi] = max_scores.astype('float16')
  
  # compare reference to alternative via mean downstream/upstream coverage ratios (signed)
  if 'SCOVR' in sed_stats:
    
    cov_vec = (alt_preds + cov_pseudo) / (ref_preds + cov_pseudo)
    
    if np.sum(np.mean(ref_preds, axis=1) > cov_min) >= 1 :
        cov_vec = cov_vec[np.mean(ref_preds, axis=1) > cov_min, :]
    
    cov_vec = np.concatenate([
      np.ones((1, cov_vec.shape[1])),
      cov_vec,
      np.ones((1, cov_vec.shape[1])),
    ], axis=0)
    
    max_scores = np.zeros(cov_vec.shape[1])
    max_scores_s = np.zeros(cov_vec.shape[1])
    for j in range(1, cov_vec.shape[0]) :
      avg_up = np.mean(cov_vec[:j, :], axis=0)
      avg_dn = np.mean(cov_vec[j:, :], axis=0)
        
      scores = np.abs(np.log2(avg_dn / avg_up))
      scores_s = np.log2(avg_dn / avg_up)
        
      if np.mean(scores) > np.mean(max_scores) :
        max_scores = scores
        max_scores_s = scores_s
      
    sed_out['SCOVR'][xi] = max_scores_s.astype('float16')

  # compare reference to alternative via mean downstream/upstream proportion ratios (PAS-seq)
  if 'PROP3' in sed_stats or 'COVR3' in sed_stats or 'COVR3WIDE' in sed_stats:
    
    prop_vec_ref = (ref_preds + cov_pseudo) / np.sum(ref_preds + cov_pseudo, axis=0)[None, :]
    prop_vec_alt = (alt_preds + cov_pseudo) / np.sum(alt_preds + cov_pseudo, axis=0)[None, :]
    
    max_scores = np.zeros(prop_vec_ref.shape[1])
    for j in range(1, prop_vec_ref.shape[0]) :
      
      dist_usage_ref = np.sum(prop_vec_ref[j:, :], axis=0)
      dist_usage_alt = np.sum(prop_vec_alt[j:, :], axis=0)
      
      scores = np.abs(np.log2(dist_usage_alt / (1. - dist_usage_alt)) - np.log2(dist_usage_ref / (1. - dist_usage_ref)))
        
      if np.mean(scores) > np.mean(max_scores) :
        max_scores = scores
    
    if 'PROP3' in sed_stats:
      sed_out['PROP3'][xi] = max_scores.astype('float16')
    if 'COVR3' in sed_stats:
      sed_out['COVR3'][xi] = max_scores.astype('float16')
    if 'COVR3WIDE' in sed_stats:
      sed_out['COVR3WIDE'][xi] = max_scores.astype('float16')

  # compare reference to alternative via mean downstream/upstream proportion ratios (PAS-seq; signed)
  if 'SPROP3' in sed_stats or 'SCOVR3' in sed_stats or 'SCOVR3WIDE' in sed_stats:
    
    prop_vec_ref = (ref_preds + cov_pseudo) / np.sum(ref_preds + cov_pseudo, axis=0)[None, :]
    prop_vec_alt = (alt_preds + cov_pseudo) / np.sum(alt_preds + cov_pseudo, axis=0)[None, :]
    
    max_scores = np.zeros(prop_vec_ref.shape[1])
    max_scores_s = np.zeros(prop_vec_ref.shape[1])
    for j in range(1, prop_vec_ref.shape[0]) :
      
      dist_usage_ref = np.sum(prop_vec_ref[j:, :], axis=0)
      dist_usage_alt = np.sum(prop_vec_alt[j:, :], axis=0)
      
      scores = np.abs(np.log2(dist_usage_alt / (1. - dist_usage_alt)) - np.log2(dist_usage_ref / (1. - dist_usage_ref)))
      scores_s = np.log2(dist_usage_alt / (1. - dist_usage_alt)) - np.log2(dist_usage_ref / (1. - dist_usage_ref))
        
      if np.mean(scores) > np.mean(max_scores) :
        max_scores = scores
        max_scores_s = scores_s
    
    if 'SPROP3' in sed_stats:
      sed_out['SPROP3'][xi] = max_scores_s.astype('float16')
    if 'SCOVR3' in sed_stats:
      sed_out['SCOVR3'][xi] = max_scores_s.astype('float16')
    if 'SCOVR3WIDE' in sed_stats:
      sed_out['SCOVR3WIDE'][xi] = max_scores_s.astype('float16')

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
