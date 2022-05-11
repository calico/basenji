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
from basenji_sed import initialize_output_h5, make_1hot_alt, read_tss_bed, write_snp

'''
sonnet_sed.py

Compute SNP Expression Difference (SED) scores for SNPs in a VCF file,
relative to gene TSS in a BED file.
'''

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <model> <vcf_file> <tss_bed_file>'
  parser = OptionParser(usage)
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
  parser.add_option('--species', dest='species',
      default='human')
  parser.add_option('--stats', dest='sed_stats',
      default='SED',
      help='Comma-separated list of stats to save. [Default: %default]')
  parser.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')
  (options, args) = parser.parse_args()

  if len(args) == 3:
    # single worker
    model_file = args[0]
    vcf_file = args[1]
    tss_bed_file = args[2]

  elif len(args) == 4:
    # multi separate
    options_pkl_file = args[0]
    model_file = args[1]
    vcf_file = args[2]
    tss_bed_file = args[3]

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
    model_file = args[1]
    vcf_file = args[2]
    tss_bed_file = args[3]
    worker_index = int(args[4])

    # load options
    options_pkl = open(options_pkl_file, 'rb')
    options = pickle.load(options_pkl)
    options_pkl.close()

    # update output directory
    options.out_dir = '%s/job%d' % (options.out_dir, worker_index)

  else:
    parser.error('Must provide model, VCF, and gene TSS files')

  if not os.path.isdir(options.out_dir):
    os.mkdir(options.out_dir)

  options.shifts = [int(shift) for shift in options.shifts.split(',')]
  options.sed_stats = options.sed_stats.split(',')

  if options.targets_file is None:
    target_slice = None
  else:
    targets_df = pd.read_csv(options.targets_file, sep='\t', index_col=0)
    target_ids = targets_df.identifier
    target_labels = targets_df.description
    target_slice = targets_df.index

  #################################################################
  # setup model

  seqnn_model = tf.saved_model.load(model_file).model

  # query num model targets 
  seq_length = seqnn_model.predict_on_batch.input_signature[0].shape[1]
  null_1hot = np.zeros((1,seq_length,4))
  null_preds = seqnn_model.predict_on_batch(null_1hot)
  null_preds = null_preds[options.species].numpy()
  _, targets_length, num_targets = null_preds.shape

  if options.targets_file is None:
    target_ids = ['t%d' % ti for ti in range(num_targets)]
    target_labels = ['']*len(target_ids)

  #################################################################
  # read SNPs / genes

  # read SNPs from VCF
  snps = bvcf.vcf_snps(vcf_file)
  num_snps = len(snps)

  # read TSS from BED
  tss_seqs = read_tss_bed(tss_bed_file, seq_length)

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


  #################################################################
  # predict SNP scores, write output

  # initialize predictions stream
  preds_stream = stream.PredStreamSonnet(seqnn_model, seq_gen(),
    rc=options.rc, shifts=options.shifts, species=options.species, 
    slice_center=options.slice_center, batch_size=options.batch_size)

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

      write_snp(ref_preds, alt_preds, sed_out, xi,
                options.sed_stats, options.log_pseudo)
      xi += 1

  # close genome
  fasta_open.close()

  ###################################################
  # compute SAD distributions across variants

  # write_pct(sed_out, options.sed_stats)
  sed_out.close()


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
