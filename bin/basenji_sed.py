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
import pickle
import os
import pdb
from queue import Queue
import sys
from threading import Thread
import time

import h5py
import numpy as np
import pandas as pd
import pysam
import tensorflow as tf
if tf.__version__[0] == '1':
  tf.compat.v1.enable_eager_execution()

from basenji import dna_io
from basenji import seqnn
from basenji import stream
from basenji import vcf as bvcf

'''
basenji_sed.py

Compute SNP Expression Difference (SED) scores for SNPs in a VCF file,
relative to gene TSS in a BED file.
'''

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <model_file> <vcf_file> <tss_bed_file>'
  parser = OptionParser(usage)
  parser.add_option('-f', dest='genome_fasta',
      default='%s/data/hg38.fa' % os.environ['BASENJIDIR'],
      help='Genome FASTA for sequences [Default: %default]')
  parser.add_option('-n', dest='norm_file',
      default=None,
      help='Normalize SED scores')
  parser.add_option('-o',dest='out_dir',
      default='sed',
      help='Output directory for tables and plots [Default: %default]')
  parser.add_option('-p', dest='processes',
      default=None, type='int',
      help='Number of processes, passed by multi script')
  parser.add_option('--pseudo', dest='log_pseudo',
      default=1, type='float',
      help='Log2 pseudocount [Default: %default]')
  parser.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Average forward and reverse complement predictions [Default: %default]')
  parser.add_option('--shifts', dest='shifts',
      default='0', type='str',
      help='Ensemble prediction shifts [Default: %default]')
  parser.add_option('--stats', dest='sed_stats',
      default='SED',
      help='Comma-separated list of stats to save. [Default: %default]')
  parser.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')
  parser.add_option('--threads', dest='threads',
      default=False, action='store_true',
      help='Run CPU math and output in a separate thread [Default: %default]')
  (options, args) = parser.parse_args()

  if len(args) == 4:
    # single worker
    params_file = args[0]
    model_file = args[1]
    vcf_file = args[2]
    tss_bed_file = args[3]

  elif len(args) == 5:
    # multi separate
    options_pkl_file = args[0]
    params_file = args[1]
    model_file = args[2]
    vcf_file = args[3]
    tss_bed_file = args[4]

    # save out dir
    out_dir = options.out_dir

    # load options
    options_pkl = open(options_pkl_file, 'rb')
    options = pickle.load(options_pkl)
    options_pkl.close()

    # update output directory
    options.out_dir = out_dir

  elif len(args) == 6:
    # multi worker
    options_pkl_file = args[0]
    params_file = args[1]
    model_file = args[2]
    vcf_file = args[3]
    tss_bed_file = args[4]
    worker_index = int(args[5])

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
  options.sed_stats = options.sed_stats.split(',')

  #################################################################
  # read parameters and targets

  # read model parameters
  with open(params_file) as params_open:
    params = json.load(params_open)
  params_model = params['model']
  params_train = params['train']

  if options.targets_file is None:
    target_slice = None
  else:
    targets_df = pd.read_csv(options.targets_file, sep='\t', index_col=0)
    target_ids = targets_df.identifier
    target_labels = targets_df.description
    target_slice = targets_df.index

  #################################################################
  # setup model

  seqnn_model = seqnn.SeqNN(params_model)
  seqnn_model.restore(model_file)
  seqnn_model.build_slice(target_slice)
  seqnn_model.build_ensemble(options.rc, options.shifts)

  targets_length = seqnn_model.target_lengths[0]
  num_targets = seqnn_model.num_targets()
  if options.targets_file is None:
    target_ids = ['t%d' % ti for ti in range(num_targets)]
    target_labels = ['']*len(target_ids)


  #################################################################
  # read SNPs / genes

  # read SNPs from VCF
  snps = bvcf.vcf_snps(vcf_file)
  num_snps = len(snps)

  # read TSS from BED
  tss_seqs = read_tss_bed(tss_bed_file, params_model['seq_length'])

  # filter for worker TSS
  if options.processes is not None:
    worker_bounds = np.linspace(0, len(tss_seqs), options.processes+1, dtype='int')
    wstart = worker_bounds[worker_index]
    wend = worker_bounds[worker_index+1]
    tss_seqs = tss_seqs[wstart:wend]

  # map TSS index to SNP indexes
  tss_snps = bvcf.intersect_seqs_snps(vcf_file, tss_seqs)

  # open genome FASTA
  fasta_open = pysam.Fastafile(options.genome_fasta)

  def seq_gen():
    for si, tss_seq in enumerate(tss_seqs):
      ref_1hot = tss_seq.make_1hot(fasta_open)
      yield ref_1hot

      for vi in tss_snps[si]:
        alt_1hot = make_1hot_alt(ref_1hot, tss_seq.start, snps[vi])
        yield alt_1hot


  #################################################################
  # setup output

  sed_out = initialize_output_h5(options.out_dir, options.sed_stats, tss_seqs, tss_snps,
                                 snps, target_ids, target_labels, targets_length)

  if options.threads:
    snp_threads = []
    snp_queue = Queue()
    for i in range(1):
      sw = SNPWorker(snp_queue, sed_out, options.sed_stats, options.log_pseudo)
      sw.start()
      snp_threads.append(sw)


  #################################################################
  # predict SNP scores, write output

  # initialize predictions stream
  preds_stream = stream.PredStreamGen(seqnn_model, seq_gen(), params_train['batch_size'])

  # predictions index
  pi = 0

  # TSS/SNP index
  xi = 0

  for si, tss_seq in enumerate(tss_seqs):
    # get reference predictions
    ref_preds = preds_stream[pi]
    pi += 1

    # for each variant
    for vi in tss_snps[si]:

      # get alternative predictions
      alt_preds = preds_stream[pi]
      pi += 1

      if options.threads:
        # queue SNP
        snp_queue.put((ref_preds, alt_preds, xi))
      else:
        write_snp(ref_preds, alt_preds, sed_out, xi,
                  options.sed_stats, options.log_pseudo)
      xi += 1

  if options.threads:
    # finish queue
    print('Waiting for threads to finish.', flush=True)
    snp_queue.join()

  # close genome
  fasta_open.close()

  ###################################################
  # compute SAD distributions across variants

  # write_pct(sed_out, options.sed_stats)
  sed_out.close()


def initialize_output_h5(out_dir, sed_stats, tss_seqs, tss_snps, snps, target_ids, target_labels, targets_length):
  """Initialize an output HDF5 file for SAD stats."""

  num_targets = len(target_ids)

  sed_out = h5py.File('%s/sed.h5' % out_dir, 'w')

  # collect identifier tuples
  tss_ids = []
  snp_ids = []
  for si, tss_seq in enumerate(tss_seqs):
    for vi in tss_snps[si]:
      tss_ids.append(tss_seq.identifier)
      snp_ids.append(snps[vi].rsid)
  num_scores = len(snp_ids)

  # write TSS
  tss_ids = np.array(tss_ids, 'S')
  sed_out.create_dataset('tss', data=tss_ids)

  # write SNPs
  snp_ids = np.array(snp_ids, 'S')
  sed_out.create_dataset('snp', data=snp_ids)

  # write SNP chr
  # snp_chr = np.array([snp.chr for snp in snps], 'S')
  # sed_out.create_dataset('chr', data=snp_chr)

  # write SNP pos
  # snp_pos = np.array([snp.pos for snp in snps], dtype='uint32')
  # sed_out.create_dataset('pos', data=snp_pos)

  # check flips
  # snp_flips = [snp.flipped for snp in snps]

  # write SNP reference allele
  # snp_refs = []
  # snp_alts = []
  # for snp in snps:
  #   if snp.flipped:
  #     snp_refs.append(snp.alt_alleles[0])
  #     snp_alts.append(snp.ref_allele)
  #   else:
  #     snp_refs.append(snp.ref_allele)
  #     snp_alts.append(snp.alt_alleles[0])
  # snp_refs = np.array(snp_refs, 'S')
  # snp_alts = np.array(snp_alts, 'S')
  # sed_out.create_dataset('ref_allele', data=snp_refs)
  # sed_out.create_dataset('alt_allele', data=snp_alts)

  # write targets
  sed_out.create_dataset('target_ids', data=np.array(target_ids, 'S'))
  sed_out.create_dataset('target_labels', data=np.array(target_labels, 'S'))

  # initialize SED stats
  for sed_stat in sed_stats:
    if sed_stat in ['REF','ALT']:
      sed_out.create_dataset(sed_stat,
        shape=(num_scores, targets_length, num_targets),
        dtype='float16')
    else:
      sed_out.create_dataset(sed_stat,
        shape=(num_scores, num_targets),
        dtype='float16')

  return sed_out


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


def write_snp(ref_preds, alt_preds, sed_out, xi, sed_stats, log_pseudo):
  """Write SNP predictions to HDF, assuming the length dimension has
      been maintained."""

  ref_preds = ref_preds.astype('float64')
  alt_preds = alt_preds.astype('float64')
  num_targets = ref_preds.shape[-1]

  # sum across length
  ref_preds_sum = ref_preds.sum(axis=0)
  alt_preds_sum = alt_preds.sum(axis=0)

  # compare reference to alternative via mean subtraction
  if 'SED' in sed_stats:
    sad = alt_preds_sum - ref_preds_sum
    sed_out['SED'][xi,:] = sad.astype('float16')

  # compare reference to alternative via mean log division
  if 'SEDR' in sed_stats:
    sar = np.log2(alt_preds_sum + log_pseudo) \
                   - np.log2(ref_preds_sum + log_pseudo)
    sed_out['SEDR'][xi,:] = sar.astype('float16')

  # compare geometric means
  if 'SER' in sed_stats:
    sar_vec = np.log2(alt_preds + log_pseudo) \
                - np.log2(ref_preds + log_pseudo)
    geo_sad = sar_vec.sum(axis=0)
    sed_out['SER'][xi,:] = geo_sad.astype('float16')

  # predictions
  if 'REF' in sed_stats:
    sed_out['REF'][xi,:] = ref_preds.astype('float16')
  if 'ALT' in sed_stats:
    sed_out['ALT'][xi,:] = alt_preds.astype('float16')


def read_tss_bed(tss_bed_file, seq_length):
  tss_list = []
  for line in open(tss_bed_file):
    a = line.split('\t')
    chrom = a[0]
    tstart = int(a[1])
    tend = int(a[2])
    identifier = a[3]
    strand = a[5]

    mid = (tstart + tend) // 2
    sstart = mid - seq_length//2
    send = sstart + seq_length

    tss_list.append(TssSeq(identifier, chrom, sstart, send, strand))

  return tss_list


def make_1hot_alt(ref_1hot, seq_start, snp):
  """Return alternative allele one hot coding."""

  # helper variables
  snp_seq_pos = snp.pos - 1 - seq_start
  alt_allele = snp.alt_alleles[0]
  ref_n = len(snp.ref_allele)
  alt_n = len(alt_allele)

  # verify reference alleles
  ref_snp1 = ref_1hot[snp_seq_pos:snp_seq_pos+ref_n]
  ref_snp = dna_io.hot1_dna(ref_snp1)
  if snp.ref_allele != ref_snp:
    print('ERROR: %s does not match reference %s' % (snp, ref_snp), file=sys.stderr)
    exit(1)

  # copy reference
  alt_1hot = np.copy(ref_1hot)

  if alt_n == ref_n:
    # SNP
    dna_io.hot1_set(alt_1hot, snp_seq_pos, alt_allele)

  elif ref_n > alt_n:
    # deletion
    delete_len = ref_n - alt_n
    if (snp.ref_allele[0] == alt_allele[0]):
      dna_io.hot1_delete(alt_1hot, snp_seq_pos+1, delete_len)
    else:
      print('WARNING: Delection first nt does not match: %s %s' % (snp.ref_allele, alt_allele), file=sys.stderr)    

  else:
    # insertion
    if (snp.ref_allele[0] == alt_allele[0]):
      dna_io.hot1_insert(alt_1hot, snp_seq_pos+1, alt_allele[1:])
    else:
      print('WARNING: Insertion first nt does not match: %s %s' % (snp.ref_allele, alt_allele), file=sys.stderr)    

  return alt_1hot


class TssSeq:
  def __init__(self, identifier, chrom, start, end, strand):
    self.identifier = identifier
    self.chrom = chrom
    self.start = start
    self.end = end
    self.strand = strand

  def add_snp(self, snp):
    self.snps.append(snp)

  def make_1hot(self, fasta_open):
    # read DNA
    if self.start < 0:
      seq_dna = 'N'*(-self.start)
      seq_dna += fasta_open.fetch(self.chrom, 0, self.end)
    else:
      seq_dna = fasta_open.fetch(self.chrom, self.start, self.end)

    # extend to full length
    if len(seq_dna) < self.end - self.start:
      seq_dna += 'N' * (self.end - self.start - len(seq_dna)) 

    # one hot encode, with N -> 0
    seq_1hot = dna_io.dna_1hot(seq_dna)

    return seq_1hot


class SNPWorker(Thread):
  """Compute summary statistics and write to HDF."""
  def __init__(self, snp_queue, sad_out, stats, log_pseudo=1):
    Thread.__init__(self)
    self.queue = snp_queue
    self.daemon = True
    self.sad_out = sad_out
    self.stats = stats
    self.log_pseudo = log_pseudo

  def run(self):
    while True:
      # unload predictions
      ref_preds, alt_preds, szi = self.queue.get()

      # write SNP
      write_snp(ref_preds, alt_preds, self.sad_out, szi, self.stats, self.log_pseudo)

      if szi % 32 == 0:
        gc.collect()

      # communicate finished task
      self.queue.task_done()

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
