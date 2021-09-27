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

from collections import OrderedDict
import json
import os
import pdb
import pickle
import sys

import h5py
import numpy as np
import pandas as pd
import pysam
import pyBigWig
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import fdrcorrection
import tensorflow as tf

if tf.__version__[0] == '1':
  tf.compat.v1.enable_eager_execution()

from basenji import bed
from basenji import dna_io
from basenji import seqnn
from basenji import stream

'''
basenji_predict_bed.py

Predict sequences from a BED file.
'''

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <model_file> <bed_file>'
  parser = OptionParser(usage)
  parser.add_option('--db', dest='database',
        default='cisbp', help='Motif database [Default: %default]')
  parser.add_option('-e', dest='pwm_exp',
      default=1, type='float',
      help='Exponentiate the position weight matrix values [Default: %default]')
  parser.add_option('-g', dest='genome',
        default='ce11', help='Genome [Default: %default]')
  parser.add_option('-o', dest='out_dir',
      default='inject_out',
      help='Output directory [Default: %default]')
  parser.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Ensemble forward and reverse complement predictions [Default: %default]')
  parser.add_option('-s', dest='offset',
      default=0, type='int',
      help='Position offset to inject motif [Default: %default]')
  parser.add_option('--shifts', dest='shifts',
      default='0',
      help='Ensemble prediction shifts [Default: %default]')
  parser.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')
  (options, args) = parser.parse_args()

  if len(args) == 3:
    params_file = args[0]
    model_file = args[1]
    bed_file = args[2]

  else:
    parser.error('Must provide parameter and model files and BED file')

  if not os.path.isdir(options.out_dir):
    os.mkdir(options.out_dir)

  options.shifts = [int(shift) for shift in options.shifts.split(',')]

  ################################################################
  # read motifs

  if options.genome == 'ce11':
    genome_fasta_file = '%s/assembly/ce11.fa' % os.environ['CE11']
    if options.database == 'cisbp':
      motif_pwm_file = '%s/cisbp2j/Caenorhabditis_elegans.meme' % os.environ['CE11']
      motif_bed_dir = '%s/cisbp2j/fimo_bed_out' % os.environ['CE11']
    elif options.database == 'jaspar':
      motif_pwm_file = '%s/jaspar/JASPAR2020_CORE_nematodes_non-redundant_pfms_meme.txt' % os.environ['CE11']
      motif_bed_dir = '%s/jaspar/fimo_bed_out' % os.environ['CE11']
    else:
      parser.error('Motif database %s not implemented' % options.database)
  else:
    parser.error('Genome %s not implemented' % options.genome)

  motif_tf = OrderedDict()
  motif_pwm = OrderedDict()

  for line in open(motif_pwm_file):
    a = line.split()
    if len(a) > 0:
      if a[0] == 'MOTIF':
        motif_id = a[1]
        motif_tf[motif_id] = cisbp_name(a[2])
        motif_pwm[motif_id] = []
      elif len(a) == 4:
        nt_col = [float(x) for x in line.split()]
        motif_pwm[motif_id].append(nt_col)

  for motif_id, pwm in motif_pwm.items():
    motif_pwm[motif_id] = np.array(pwm) ** options.pwm_exp
    motif_pwm[motif_id] /= motif_pwm[motif_id].sum(axis=-1, keepdims=True)

  num_motifs = len(motif_tf)

  #################################################################
  # read parameters and collet target information

  with open(params_file) as params_open:
    params = json.load(params_open)
  params_model = params['model']

  if options.targets_file is None:
    target_slice = None
  else:
    targets_df = pd.read_table(options.targets_file, index_col=0)
    target_slice = targets_df.index

  #################################################################
  # setup model

  # initialize model
  seqnn_model = seqnn.SeqNN(params_model)
  seqnn_model.restore(model_file)
  seqnn_model.build_slice(target_slice)
  seqnn_model.build_ensemble(options.rc, options.shifts)

  _, preds_length, num_targets = seqnn_model.model.output.shape    
  if type(preds_length) == tf.compat.v1.Dimension:
    preds_length = preds_length.value
    num_targets = preds_depth.value

  #################################################################
  # sequence dataset

  # construct model sequences
  model_seqs_dna, model_seqs_coords = bed.make_bed_seqs(
    bed_file, genome_fasta_file,
    params_model['seq_length'], stranded=True)
  num_seqs = len(model_seqs_dna)

  # define sequence generator
  def seqs_gen():
    for seq_dna in model_seqs_dna:
      seq_1hot = dna_io.dna_1hot(seq_dna)
      yield seq_1hot

      seq_center = seq_1hot.shape[0] // 2
      seq_center += options.offset
      
      for pwm in motif_pwm.values():
        motif_len = pwm.shape[0]
        motif_start = seq_center - motif_len//2

        seq_motif_1hot = seq_1hot.copy()

        for xi in range(motif_len):
          nti = np.random.choice(4, p=pwm[xi])
          seq_motif_1hot[motif_start+xi,:] = 0
          seq_motif_1hot[motif_start+xi,nti] = 1

        yield seq_motif_1hot

  #################################################################
  # predict

  preds_stream = stream.PredStreamGen(seqnn_model, seqs_gen(), params['train']['batch_size'])

  # predict index
  pi = 0

  # initialize score matrix
  seqs_motifs = np.zeros((num_seqs,num_motifs,num_targets), dtype='float16')

  for si in range(num_seqs):
    print(si, flush=True)

    preds_seq = preds_stream[pi].sum(axis=0)
    pi += 1

    for mi in range(num_motifs):
      preds_motif = preds_stream[pi].sum(axis=0)
      pi += 1
      seqs_motifs[si,mi] = (preds_motif - preds_seq).astype('float16')

  #################################################################
  # stats

  # subtract sequence means
  seqs_motifs -= seqs_motifs.mean(axis=1, keepdims=True)

  """
  # compute significant difference
  motifs_pvalues = ttest_1samp(seqs_motifs, 0, axis=0)[1]

  # FDR correct
  motifs_qvalues = fdrcorrection(motifs_pvalues.flatten())[1]
  motifs_qvalues = np.reshape(motifs_qvalues, (num_motifs, num_targets))
  motifs_qvalues = motifs_qvalues.min(axis=1)
  """
  # everything is significant

  #################################################################
  # output

  # initialize HDF5
  out_h5_file = '%s/scores.h5' % options.out_dir
  if os.path.isfile(out_h5_file):
    os.remove(out_h5_file)
  out_h5 = h5py.File(out_h5_file, 'w')

  # TEMP
  # out_h5.create_dataset('seqs_motifs', data=seqs_motifs, dtype='float16')

  motif_scores = seqs_motifs.mean(axis=0, dtype='float32').astype('float16')
  out_h5.create_dataset('scores', data=motif_scores, dtype='float16')

  motif_var = seqs_motifs.var(axis=0, dtype='float32').astype('float16')
  out_h5.create_dataset('var', data=motif_var, dtype='float16')

  # out_h5.create_dataset('qvalues', data=motifs_qvalues, dtype='float16')

  motif_ids = np.array(list(motif_tf.keys()), dtype='S')
  motif_tfs = np.array(list(motif_tf.values()), dtype='S')
  out_h5.create_dataset('motif', data=motif_ids)
  out_h5.create_dataset('tf', data=motif_tfs)

  # close output HDF5
  out_h5.close()

def cisbp_name(tf_name_long):
  if tf_name_long[0] == '(':
    close_paren = tf_name_long.find(')')
    tf_name = tf_name_long[1:close_paren]
  else:
    tf_name = tf_name_long
  return tf_name

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
