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
import gc
import json
import pdb
import pickle
import os
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
from basenji import vcf as bvcf
from basenji import stream

from basenji_sad import SNPWorker, initialize_output_h5, write_pct, write_snp

'''
basenji_sad_ref.py

Compute SNP Activity Difference (SAD) scores for SNPs in a VCF file.
This versions saves computation by clustering nearby SNPs in order to
make a single reference prediction for several SNPs.
'''

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <model_file> <vcf_file>'
  parser = OptionParser(usage)
  parser.add_option('-c', dest='center_pct',
      default=0.25, type='float',
      help='Require clustered SNPs lie in center region [Default: %default]')
  parser.add_option('-f', dest='genome_fasta',
      default='%s/data/hg19.fa' % os.environ['BASENJIDIR'],
      help='Genome FASTA for sequences [Default: %default]')
  parser.add_option('--flip', dest='flip_ref',
      default=False, action='store_true',
      help='Flip reference/alternate alleles when simple [Default: %default]')
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
  parser.add_option('--ti', dest='track_indexes',
      default=None, type='str',
      help='Comma-separated list of target indexes to output BigWig tracks')
  parser.add_option('--threads', dest='threads',
      default=False, action='store_true',
      help='Run CPU math and output in a separate thread [Default: %default]')
  parser.add_option('-u', dest='penultimate',
      default=False, action='store_true',
      help='Compute SED in the penultimate layer [Default: %default]')
  (options, args) = parser.parse_args()

  if len(args) == 3:
    # single worker
    params_file = args[0]
    model_file = args[1]
    vcf_file = args[2]

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

  if options.track_indexes is None:
    options.track_indexes = []
  else:
    options.track_indexes = [int(ti) for ti in options.track_indexes.split(',')]
    if not os.path.isdir('%s/tracks' % options.out_dir):
      os.mkdir('%s/tracks' % options.out_dir)

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
  else:
    targets_df = pd.read_csv(options.targets_file, sep='\t', index_col=0)
    target_ids = targets_df.identifier
    target_labels = targets_df.description
    target_slice = targets_df.index

  if options.penultimate:
    parser.error('Not implemented for TF2')

  #################################################################
  # setup model

  seqnn_model = seqnn.SeqNN(params_model)
  seqnn_model.restore(model_file)
  seqnn_model.build_slice(target_slice)
  seqnn_model.build_ensemble(options.rc, options.shifts)

  num_targets = seqnn_model.num_targets()
  if options.targets_file is None:
    target_ids = ['t%d' % ti for ti in range(num_targets)]
    target_labels = ['']*len(target_ids)

  #################################################################
  # load SNPs

  # filter for worker SNPs
  if options.processes is not None:
    # determine boundaries
    num_snps = bvcf.vcf_count(vcf_file)
    worker_bounds = np.linspace(0, num_snps, options.processes+1, dtype='int')

    # read sorted SNPs from VCF
    snps = bvcf.vcf_snps(vcf_file, require_sorted=True, flip_ref=options.flip_ref,
                         validate_ref_fasta=options.genome_fasta,
                         start_i=worker_bounds[worker_index],
                         end_i=worker_bounds[worker_index+1])
  else:
    # read sorted SNPs from VCF
    snps = bvcf.vcf_snps(vcf_file, require_sorted=True, flip_ref=options.flip_ref,
                         validate_ref_fasta=options.genome_fasta)

  # cluster SNPs by position
  snp_clusters = cluster_snps(snps, params_model['seq_length'], options.center_pct)

  # delimit sequence boundaries
  [sc.delimit(params_model['seq_length']) for sc in snp_clusters]

  # open genome FASTA
  genome_open = pysam.Fastafile(options.genome_fasta)

  # make SNP sequence generator
  def snp_gen():
    for sc in snp_clusters:
      snp_1hot_list = sc.get_1hots(genome_open)
      for snp_1hot in snp_1hot_list:
        yield snp_1hot


  #################################################################
  # setup output

  snp_flips = np.array([snp.flipped for snp in snps], dtype='bool')

  sad_out = initialize_output_h5(options.out_dir, options.sad_stats,
                                 snps, target_ids, target_labels)

  if options.threads:
    snp_threads = []
    snp_queue = Queue()
    for i in range(1):
      sw = SNPWorker(snp_queue, sad_out, options.sad_stats, options.log_pseudo)
      sw.start()
      snp_threads.append(sw)


  #################################################################
  # predict SNP scores, write output

  # initialize predictions stream
  preds_stream = stream.PredStreamGen(seqnn_model, snp_gen(), params['train']['batch_size'])

  # predictions index
  pi = 0

  # SNP index
  si = 0

  for snp_cluster in snp_clusters:
    ref_preds = preds_stream[pi]
    pi += 1

    for snp in snp_cluster.snps:
      # print(snp, flush=True)

      alt_preds = preds_stream[pi]
      pi += 1

      if snp_flips[si]:
        ref_preds, alt_preds = alt_preds, ref_preds

      if options.threads:
        # queue SNP
          snp_queue.put((ref_preds, alt_preds, si))
      else:
        # process SNP
        write_snp(ref_preds, alt_preds, sad_out, si,
                  options.sad_stats, options.log_pseudo)

      # update SNP index
      si += 1

  # finish queue
  if options.threads:
    print('Waiting for threads to finish.', flush=True)
    snp_queue.join()

  # close genome
  genome_open.close()

  ###################################################
  # compute SAD distributions across variants

  write_pct(sad_out, options.sad_stats)
  sad_out.close()


def cluster_snps(snps, seq_len, center_pct):
  """Cluster a sorted list of SNPs into regions that will satisfy
     the required center_pct."""
  valid_snp_distance = int(seq_len*center_pct)

  snp_clusters = []
  cluster_chr = None

  for snp in snps:
    if snp.chr == cluster_chr and snp.pos < cluster_pos0 + valid_snp_distance:
      # append to latest cluster
      snp_clusters[-1].add_snp(snp)
    else:
      # initialize new cluster
      snp_clusters.append(SNPCluster())
      snp_clusters[-1].add_snp(snp)
      cluster_chr = snp.chr
      cluster_pos0 = snp.pos

  return snp_clusters


class SNPCluster:
  def __init__(self):
    self.snps = []
    self.chr = None
    self.start = None
    self.end = None

  def add_snp(self, snp):
    self.snps.append(snp)

  def delimit(self, seq_len):
    positions = [snp.pos for snp in self.snps]
    pos_min = np.min(positions)
    pos_max = np.max(positions)
    pos_mid = (pos_min + pos_max) // 2

    self.chr = self.snps[0].chr
    self.start = pos_mid - seq_len//2
    self.end = self.start + seq_len

    for snp in self.snps:
      snp.seq_pos = snp.pos - 1 - self.start

  def get_1hots(self, genome_open):
    seqs1_list = []

    # extract reference
    if self.start < 0:
      ref_seq = 'N'*(-self.start) + genome_open.fetch(self.chr, 0, self.end).upper()
    else:
      ref_seq = genome_open.fetch(self.chr, self.start, self.end).upper()

    # extend to full length
    if len(ref_seq) < self.end - self.start:
      ref_seq += 'N'*(self.end-self.start-len(ref_seq))

    # verify reference alleles
    for snp in self.snps:
      ref_n = len(snp.ref_allele)
      ref_snp = ref_seq[snp.seq_pos:snp.seq_pos+ref_n]
      if snp.ref_allele != ref_snp:
        print('ERROR: %s does not match reference %s' % (snp, ref_snp), file=sys.stderr)
        exit(1)

    # 1 hot code reference sequence
    ref_1hot = dna_io.dna_1hot(ref_seq)
    seqs1_list = [ref_1hot]

    # make alternative 1 hot coded sequences
    #  (assuming SNP is 1-based indexed)
    for snp in self.snps:
      alt_1hot = make_alt_1hot(ref_1hot, snp.seq_pos, snp.ref_allele, snp.alt_alleles[0])
      seqs1_list.append(alt_1hot)

    return seqs1_list


def make_alt_1hot(ref_1hot, snp_seq_pos, ref_allele, alt_allele):
  """Return alternative allele one hot coding."""
  ref_n = len(ref_allele)
  alt_n = len(alt_allele)

  # copy reference
  alt_1hot = np.copy(ref_1hot)

  if alt_n == ref_n:
    # SNP
    dna_io.hot1_set(alt_1hot, snp_seq_pos, alt_allele)

  elif ref_n > alt_n:
    # deletion
    delete_len = ref_n - alt_n
    if (ref_allele[0] == alt_allele[0]):
      dna_io.hot1_delete(alt_1hot, snp_seq_pos+1, delete_len)
    else:
      print('WARNING: Delection first nt does not match: %s %s' % (ref_allele, alt_allele), file=sys.stderr)    

  else:
    # insertion
    if (ref_allele[0] == alt_allele[0]):
      dna_io.hot1_insert(alt_1hot, snp_seq_pos+1, alt_allele[1:])
    else:
      print('WARNING: Insertion first nt does not match: %s %s' % (ref_allele, alt_allele), file=sys.stderr)    

  return alt_1hot


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
