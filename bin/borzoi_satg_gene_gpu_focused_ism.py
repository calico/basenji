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
import os
import pdb
import pickle
from queue import Queue
import random
import sys
from threading import Thread
import time

import h5py
import numpy as np
import pandas as pd
import pysam
import tensorflow as tf

from basenji import dna_io
from basenji import gene as bgene
from basenji import seqnn
from borzoi_sed import targets_prep_strand

from scipy.ndimage import gaussian_filter1d

'''
borzoi_satg_gene_gpu_focused_ism.py

Perform an ISM analysis for genes specified in a GTF file, targeting high-saliency regions based on gradient scores.
'''

# tf code for computing ISM scores on GPU
@tf.function
def _score_func(model, seq_1hot, target_slice, pos_slice, pos_mask=None, pos_slice_denom=None, pos_mask_denom=True, track_scale=1., track_transform=1., clip_soft=None, pseudo_count=0., no_transform=False, aggregate_tracks=None, use_mean=False, use_ratio=False, use_logodds=False) :
      
      # predict
      preds = tf.gather(model(seq_1hot, training=False), target_slice, axis=-1, batch_dims=1)
      
      if not no_transform :
      
        # undo scale
        preds = preds / track_scale

        # undo soft_clip
        if clip_soft is not None :
          preds = tf.where(preds > clip_soft, (preds - clip_soft)**2 + clip_soft, preds)

        # undo sqrt
        preds = preds**(1. / track_transform)
      
      if aggregate_tracks is not None :
        preds = tf.reduce_mean(tf.reshape(preds, (preds.shape[0], preds.shape[1], preds.shape[2] // aggregate_tracks, aggregate_tracks)), axis=-1)
      
      # slice specified positions
      preds_slice = tf.gather(preds, pos_slice, axis=1, batch_dims=1)
      if pos_mask is not None :
        preds_slice = preds_slice * pos_mask
      
      # slice denominator positions
      if use_ratio and pos_slice_denom is not None:
        preds_slice_denom = tf.gather(preds, pos_slice_denom, axis=1, batch_dims=1)
        if pos_mask_denom is not None :
          preds_slice_denom = preds_slice_denom * pos_mask_denom
      
      # aggregate over positions
      if not use_mean :
        preds_agg = tf.reduce_sum(preds_slice, axis=1)
        if use_ratio and pos_slice_denom is not None:
          preds_agg_denom = tf.reduce_sum(preds_slice_denom, axis=1)
      else :
        if pos_mask is not None :
          preds_agg = tf.reduce_sum(preds_slice, axis=1) / tf.reduce_sum(pos_mask, axis=1)
        else :
          preds_agg = tf.reduce_mean(preds_slice, axis=1)
        
        if use_ratio and pos_slice_denom is not None:
          if pos_mask_denom is not None :
            preds_agg_denom = tf.reduce_sum(preds_slice_denom, axis=1) / tf.reduce_sum(pos_mask_denom, axis=1)
          else :
            preds_agg_denom = tf.reduce_mean(preds_slice_denom, axis=1)

      # compute final statistic
      if no_transform :
        score_ratios = preds_agg
      elif not use_ratio :
        score_ratios = tf.math.log(preds_agg + pseudo_count + 1e-6)
      else :
        if not use_logodds :
          score_ratios = tf.math.log((preds_agg + pseudo_count) / (preds_agg_denom + pseudo_count) + 1e-6)
        else :
          score_ratios = tf.math.log(((preds_agg + pseudo_count) / (preds_agg_denom + pseudo_count)) / (1. - ((preds_agg + pseudo_count) / (preds_agg_denom + pseudo_count))) + 1e-6)

      return score_ratios

def get_ism(seqnn_model, seq_1hot_wt, ism_start=0, ism_end=524288, head_i=None, target_slice=None, pos_slice=None, pos_mask=None, pos_slice_denom=None, pos_mask_denom=None, track_scale=1., track_transform=1., clip_soft=None, pseudo_count=0., no_transform=False, aggregate_tracks=None, use_mean=False, use_ratio=False, use_logodds=False, bases=[0, 1, 2, 3]) :
    
    # choose model
    if seqnn_model.ensemble is not None:
      model = seqnn_model.ensemble
    elif head_i is not None:
      model = seqnn_model.models[head_i]
    else:
      model = seqnn_model.model
    
    # verify tensor shape(s)
    seq_1hot_wt = seq_1hot_wt.astype('float32')
    target_slice = np.array(target_slice).astype('int32')
    pos_slice = np.array(pos_slice).astype('int32')
    
    # convert constants to tf tensors
    track_scale = tf.constant(track_scale, dtype=tf.float32)
    track_transform = tf.constant(track_transform, dtype=tf.float32)
    if clip_soft is not None :
        clip_soft = tf.constant(clip_soft, dtype=tf.float32)
    pseudo_count = tf.constant(pseudo_count, dtype=tf.float32)
    
    if pos_mask is not None :
      pos_mask = np.array(pos_mask).astype('float32')
    
    if use_ratio and pos_slice_denom is not None :
      pos_slice_denom = np.array(pos_slice_denom).astype('int32')
      
      if pos_mask_denom is not None :
        pos_mask_denom = np.array(pos_mask_denom).astype('float32')
    
    if len(seq_1hot_wt.shape) < 3:
      seq_1hot_wt = seq_1hot_wt[None, ...]
    
    if len(target_slice.shape) < 2:
      target_slice = target_slice[None, ...]
    
    if len(pos_slice.shape) < 2:
      pos_slice = pos_slice[None, ...]
    
    if pos_mask is not None and len(pos_mask.shape) < 2:
      pos_mask = pos_mask[None, ...]
    
    if use_ratio and pos_slice_denom is not None and len(pos_slice_denom.shape) < 2:
      pos_slice_denom = pos_slice_denom[None, ...]
      
      if pos_mask_denom is not None and len(pos_mask_denom.shape) < 2:
        pos_mask_denom = pos_mask_denom[None, ...]
    
    # convert to tf tensors
    seq_1hot_wt_tf = tf.convert_to_tensor(seq_1hot_wt, dtype=tf.float32)
    target_slice = tf.convert_to_tensor(target_slice, dtype=tf.int32)
    pos_slice = tf.convert_to_tensor(pos_slice, dtype=tf.int32)
    
    if pos_mask is not None :
        pos_mask = tf.convert_to_tensor(pos_mask, dtype=tf.float32)
    
    if use_ratio and pos_slice_denom is not None :
        pos_slice_denom = tf.convert_to_tensor(pos_slice_denom, dtype=tf.int32)
    
        if pos_mask_denom is not None :
            pos_mask_denom = tf.convert_to_tensor(pos_mask_denom, dtype=tf.float32)
    
    # allocate ism result tensor
    pred_ism = np.zeros((524288, 4, target_slice.shape[1] // (aggregate_tracks if aggregate_tracks is not None else 1)))
    
    # get wt pred
    score_wt = _score_func(model, seq_1hot_wt_tf, target_slice, pos_slice, pos_mask, pos_slice_denom, pos_mask_denom, track_scale, track_transform, clip_soft, pseudo_count, no_transform, aggregate_tracks, use_mean, use_ratio, use_logodds).numpy()
    
    for j in range(ism_start, ism_end) :
        for b in bases :
            if seq_1hot_wt[0, j, b] != 1. : 
                seq_1hot_mut = np.copy(seq_1hot_wt)
                seq_1hot_mut[0, j, :] = 0.
                seq_1hot_mut[0, j, b] = 1.
                
                # convert to tf tensor
                seq_1hot_mut_tf = tf.convert_to_tensor(seq_1hot_mut, dtype=tf.float32)

                # get mut pred
                score_mut = _score_func(model, seq_1hot_mut_tf, target_slice, pos_slice, pos_mask, pos_slice_denom, pos_mask_denom, track_scale, track_transform, clip_soft, pseudo_count, no_transform, aggregate_tracks, use_mean, use_ratio, use_logodds).numpy()
                
                pred_ism[j, b, :] = score_wt - score_mut

    pred_ism = np.tile(np.mean(pred_ism, axis=1, keepdims=True), (1, 4, 1)) * seq_1hot_wt[0, ..., None]
    
    return pred_ism


################################################################################
# main
# ###############################################################################
def main():
  usage = 'usage: %prog [options] <params> <model> <gene_gtf>'
  parser = OptionParser(usage)
  parser.add_option('--fa', dest='genome_fasta',
      default='%s/data/hg38.fa' % os.environ['BASENJIDIR'],
      help='Genome FASTA for sequences [Default: %default]')
  parser.add_option('-o', dest='out_dir',
      default='satg_out', help='Output directory [Default: %default]')
  parser.add_option('--rc', dest='rc',
      default=0, type='int',
      help='Ensemble forward and reverse complement predictions [Default: %default]')
  parser.add_option('-f', dest='folds',
      default='0', type='str',
      help='Model folds to use in ensemble [Default: %default]')
  parser.add_option('--shifts', dest='shifts',
      default='0', type='str',
      help='Ensemble prediction shifts [Default: %default]')
  parser.add_option('--span', dest='span',
      default=0, type='int',
      help='Aggregate entire gene span [Default: %default]')
  parser.add_option('--clip_soft', dest='clip_soft',
      default=None, type='float',
      help='Model clip_soft setting [Default: %default]')
  parser.add_option('--no_transform', dest='no_transform',
      default=0, type='int',
      help='Run gradients with no inverse transforms [Default: %default]')
  parser.add_option('--pseudo_qtl', dest='pseudo_qtl',
      default=None, type='float',
      help='Quantile of predicted scalars to choose as pseudo count [Default: %default]')
  parser.add_option('--aggregate_tracks', dest='aggregate_tracks',
      default=None, type='int',
      help='Run gradients with no inverse transforms [Default: %default]')
  parser.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')
  parser.add_option('--tissue_files', dest='tissue_files',
      default=None, type='str',
      help='Comma-separated list of files containing saliency scores (h5 format).')
  parser.add_option('--tissues', dest='tissues',
      default=None, type='str',
      help='Comma-separated list of tissue names.')
  parser.add_option('--tissue', dest='tissue',
      default=None, type='str',
      help='Tissue name to filter on in gene_file.')
  parser.add_option('--main_tissue_ix', dest='main_tissue_ix',
      default=0, type='int',
      help='Main tissue index.')
  parser.add_option('--ism_size', dest='ism_size',
      default=192, type='int',
      help='Length of sequence window to run ISM across.')
  parser.add_option('--gene_file', dest='gene_file',
      default=None, type='str',
      help='Csv-file of gene metadata.')
  parser.add_option('--max_n_genes', dest='max_n_genes',
      default=10, type='int',
      help='Maximum number of genes in the GTF to compute ISMs for [Default: %default]')
  parser.add_option('--gaussian_sigma', dest='gaussian_sigma',
      default=8, type='int',
      help='Sigma value for 1D gaussian smoothing filter [Default: %default]')
  parser.add_option('--min_padding', dest='min_padding',
      default=65536, type='int',
      help='Minimum crop to apply to scores before searching for smoothed maximum [Default: %default]')
  
  (options, args) = parser.parse_args()

  if len(args) == 3:
    # single worker
    params_file = args[0]
    model_folder = args[1]
    genes_gtf_file = args[2]
  else:
    parser.error('Must provide parameter file, model folder and GTF file')

  if not os.path.isdir(options.out_dir):
    os.mkdir(options.out_dir)

  options.folds = [int(fold) for fold in options.folds.split(',')]
  options.shifts = [int(shift) for shift in options.shifts.split(',')]
  
  options.tissue_files = [tissue for tissue in options.tissue_files.split(",")]
  options.tissues = [tissue for tissue in options.tissues.split(",")]

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
  orig_new_index = dict(zip(targets_df.index, np.arange(targets_df.shape[0])))
  targets_strand_pair = np.array([orig_new_index[ti] for ti in targets_df.strand_pair])
  targets_strand_df = targets_prep_strand(targets_df)
  num_targets = len(targets_strand_df)

  # specify relative target indices
  targets_df['row_index'] = np.arange(len(targets_df), dtype='int32')

  #################################################################
  # load first model fold to get parameters

  seqnn_model = seqnn.SeqNN(params_model)
  seqnn_model.restore(model_folder + "/f0c0/model0_best.h5", 0, by_name=False)
  seqnn_model.build_slice(targets_df.index, False)
  # seqnn_model.build_ensemble(options.rc, options.shifts)

  model_stride = seqnn_model.model_strides[0]
  model_crop = seqnn_model.target_crops[0]
  target_length = seqnn_model.target_lengths[0]

  #################################################################
  # read genes

  # parse GTF
  transcriptome = bgene.Transcriptome(genes_gtf_file)

  # order valid genes
  genome_open = pysam.Fastafile(options.genome_fasta)
  gene_list = sorted(transcriptome.genes.keys())
  num_genes = len(gene_list)
  
  #Make copy of unfiltered gene list
  gene_list_all = gene_list.copy()
  
  #################################################################
  # load tissue gene list
  
  #Load gene dataframe and select tissue
  gene_df = pd.read_csv(options.gene_file, sep='\t')
  gene_df = gene_df.query("tissue == '" + str(options.tissue) + "'").copy().reset_index(drop=True)
  gene_df = gene_df.drop(columns=['Unnamed: 0'])

  print("len(gene_df) = " + str(len(gene_df)))

  #Truncate by maximum number of genes
  gene_df = gene_df.iloc[:options.max_n_genes].copy().reset_index(drop=True)

  #Get list of genes for tissue
  tissue_genes = gene_df['gene_base'].values.tolist()

  #print("len(tissue_genes) = " + str(len(tissue_genes)))
  
  #Filter transcriptome gene list
  gene_list = [gene for gene in gene_list if gene.split(".")[0] in set(tissue_genes)]
  num_genes = len(gene_list)
  
  print("num_genes = " + str(num_genes))
  
  #################################################################
  # load h5 scores
  
  seqs = None
  strands = None
  chrs = None
  starts = None
  ends = None
  genes = None
  all_scores = []
  pseudo_counts = []
  
  for scores_h5_file, scores_h5_tissue in zip(options.tissue_files, options.tissues) :
    
    print("Reading '" + scores_h5_file + "'")

    with h5py.File(scores_h5_file, 'r') as score_file:

      #Get scores and onehots
      scores = score_file['grads'][()][..., 0]
      seqs = score_file['seqs'][()]

      #Get auxiliary information
      strands = score_file['strand'][()]
      strands = np.array([strands[j].decode() for j in range(strands.shape[0])])

      chrs = score_file['chr'][()]
      chrs = np.array([chrs[j].decode() for j in range(chrs.shape[0])])

      starts = np.array(score_file['start'][()])
      ends = np.array(score_file['end'][()])

      genes = score_file['gene'][()]
      genes = np.array([genes[j].decode() for j in range(genes.shape[0])]) #.split(".")[0]
      
      gene_dict = {gene : gene_i for gene_i, gene in enumerate(genes.tolist())}

      #Get index of rows to keep
      keep_index = []
      for gene in gene_list :
        keep_index.append(gene_dict[gene])
      
      #Optionally compute pseudo-counts
      if options.aggregate_tracks is not None and options.pseudo_qtl is not None :
        
        #Load gene dataframe and select active tissue
        gene_df_all = pd.read_csv(options.gene_file, sep='\t')
        gene_df_all = gene_df_all.query("tissue == '" + str(scores_h5_tissue) + "'").copy().reset_index(drop=True)
        gene_df_all = gene_df_all.drop(columns=['Unnamed: 0'])

        #Get list of genes for active tissue
        tissue_genes_all = gene_df_all['gene_base'].values.tolist()

        #Filter transcriptome gene list
        gene_list_tissue = [gene for gene in gene_list_all if gene.split(".")[0] in set(tissue_genes_all)]
        num_genes_tissue = len(gene_list_tissue)

        print(" - num_genes_tissue = " + str(num_genes_tissue))
        
        #Get index of genes beloning to active tissue
        gene_index = []
        for gene in gene_list_tissue :
          gene_index.append(gene_dict[gene])
        
        #Compute pseudo-count
        pseudo_count = np.quantile(np.array(score_file['preds'][()][gene_index, 0]), q=options.pseudo_qtl)
        pseudo_counts.append(pseudo_count)

      #Filter/sub-select data
      scores = scores[keep_index, ...]
      seqs = seqs[keep_index, ...]
      strands = strands[keep_index]
      chrs = chrs[keep_index]
      starts = starts[keep_index]
      ends = ends[keep_index]
      genes = genes[keep_index]

      #Append input-gated scores
      all_scores.append((scores * seqs)[None, ...])

      #Collect garbage
      gc.collect()

  #Collect final scores
  scores = np.concatenate(all_scores, axis=0)

  print("scores.shape = " + str(scores.shape))
  
  #Collect pseudo-counts
  pseudo_count = 0.
  if options.aggregate_tracks is not None and options.pseudo_qtl is not None :
    pseudo_count = np.array(pseudo_counts, dtype='float32')[None, :]
  
    print("pseudo_count = " + str(np.round(pseudo_count, 2)))
  else :
    print("pseudo_count = " + str(round(pseudo_count, 2)))

  #################################################################
  # setup output

  # choose gene sequences
  genes_chr = chrs.tolist()
  genes_start = starts.tolist()
  genes_end = ends.tolist()
  genes_strand = strands.tolist()
  
  #################################################################
  # calculate ism start and end positions per gene
  
  print("main_tissue_ix = " + str(options.main_tissue_ix))
  
  genes_ism_start = []
  genes_ism_end = []
  for gi in range(len(gene_list)) :
    score_2 = scores[options.main_tissue_ix, gi, ...]
    score_1 = np.mean(scores[np.arange(scores.shape[0]) != options.main_tissue_ix, gi, ...], axis=0)

    diff_score = np.sum(score_2 - score_1, axis=-1)
    
    #Apply gaussian filter
    diff_score = gaussian_filter1d(diff_score.astype('float32'), sigma=options.gaussian_sigma, truncate=2).astype('float16')

    max_pos = np.argmax(diff_score[options.min_padding:-options.min_padding]) + options.min_padding
    
    genes_ism_start.append(max_pos - options.ism_size // 2)
    genes_ism_end.append(max_pos + options.ism_size // 2)

  #################################################################
  # predict ISM scores, write output
  
  print("clip_soft = " + str(options.clip_soft))
  
  print("n genes = " + str(len(genes_chr)))
  
  # loop over folds
  for fold_ix in options.folds :
    print("-- Fold = " + str(fold_ix) + " --")
    
    # (re-)initialize HDF5
    scores_h5_file = '%s/ism_f%dc0.h5' % (options.out_dir, fold_ix)
    if os.path.isfile(scores_h5_file):
      os.remove(scores_h5_file)
    scores_h5 = h5py.File(scores_h5_file, 'w')
    scores_h5.create_dataset('seqs', dtype='bool',
      shape=(num_genes, options.ism_size, 4))
    scores_h5.create_dataset('isms', dtype='float16',
      shape=(num_genes, options.ism_size, 4, num_targets // (options.aggregate_tracks if options.aggregate_tracks is not None else 1)))
    scores_h5.create_dataset('gene', data=np.array(gene_list, dtype='S'))
    scores_h5.create_dataset('chr', data=np.array(genes_chr, dtype='S'))
    scores_h5.create_dataset('start', data=np.array(genes_start))
    scores_h5.create_dataset('end', data=np.array(genes_end))
    scores_h5.create_dataset('ism_start', data=np.array(genes_ism_start))
    scores_h5.create_dataset('ism_end', data=np.array(genes_ism_end))
    scores_h5.create_dataset('strand', data=np.array(genes_strand, dtype='S'))
    
    # load model fold
    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(model_folder + "/f" + str(fold_ix) + "c0/model0_best.h5", 0, by_name=False)
    seqnn_model.build_slice(targets_df.index, False)
  
    track_scale = targets_df.iloc[0]['scale']
    track_transform = 3. / 4.

    for shift in options.shifts :
      print('Processing shift %d' % shift, flush=True)

      for rev_comp in ([False, True] if options.rc == 1 else [False]) :

        if options.rc == 1 :
          print('Fwd/rev = %s' % ('fwd' if not rev_comp else 'rev'), flush=True)

        seq_1hots = []
        gene_slices = []
        gene_targets = []

        for gi, gene_id in enumerate(gene_list):
          
          if gi % 50 == 0 :
            print('Processing %d, %s' % (gi, gene_id), flush=True)
          
          gene = transcriptome.genes[gene_id]

          # make sequence
          seq_1hot = make_seq_1hot(genome_open, genes_chr[gi], genes_start[gi], genes_end[gi], seq_len)
          seq_1hot = dna_io.hot1_augment(seq_1hot, shift=shift)

          # determine output sequence start
          seq_out_start = genes_start[gi] + model_stride*model_crop
          seq_out_len = model_stride*target_length

          # determine output positions
          gene_slice = gene.output_slice(seq_out_start, seq_out_len, model_stride, options.span == 1)
          
          # determine ism window
          gene_ism_start = genes_ism_start[gi]
          gene_ism_end = genes_ism_end[gi]

          if rev_comp:
            seq_1hot = dna_io.hot1_rc(seq_1hot)
            gene_slice = target_length - gene_slice - 1
            
            gene_ism_start = seq_len - genes_ism_end[gi] - 1
            gene_ism_end = seq_len - genes_ism_start[gi] - 1

          # slice relevant strand targets
          if genes_strand[gi] == '+':
            gene_strand_mask = (targets_df.strand != '-') if not rev_comp else (targets_df.strand != '+')
          else:
            gene_strand_mask = (targets_df.strand != '+') if not rev_comp else (targets_df.strand != '-')

          gene_target = np.array(targets_df.index[gene_strand_mask].values)

          # broadcast to singleton batch
          seq_1hot = seq_1hot[None, ...]
          gene_slice = gene_slice[None, ...]
          gene_target = gene_target[None, ...]

          # ism computation
          ism = get_ism(
              seqnn_model,
              seq_1hot,
              gene_ism_start,
              gene_ism_end,
              head_i=0,
              target_slice=gene_target,
              pos_slice=gene_slice,
              track_scale=track_scale,
              track_transform=track_transform,
              clip_soft=options.clip_soft,
              pseudo_count=pseudo_count,
              no_transform=options.no_transform == 1,
              aggregate_tracks=options.aggregate_tracks,
              use_mean=False,
              use_ratio=False,
              use_logodds=False,
          )

          # undo augmentations and save ism
          ism = unaugment_grads(ism, fwdrc=(not rev_comp), shift=shift)
          
          # write to HDF5
          scores_h5['isms'][gi] += ism[genes_ism_start[gi]:genes_ism_end[gi], ...]
          
          # collect garbage
          gc.collect()

    # save sequences and normalize isms by total size of ensemble
    for gi, gene_id in enumerate(gene_list):
    
      # re-make original sequence
      seq_1hot = make_seq_1hot(genome_open, genes_chr[gi], genes_start[gi], genes_end[gi], seq_len)
      
      # write to HDF5
      scores_h5['seqs'][gi] = seq_1hot[genes_ism_start[gi]:genes_ism_end[gi], ...]
      scores_h5['isms'][gi] /= float((len(options.shifts) * (2 if options.rc == 1 else 1)))
    
    # collect garbage
    gc.collect()

  # close files
  genome_open.close()
  scores_h5.close()


def unaugment_grads(grads, fwdrc=False, shift=0):
  """ Undo sequence augmentation."""
  # reverse complement
  if not fwdrc:
    # reverse
    grads = grads[::-1, :, :]

    # swap A and T
    grads[:, [0, 3], :] = grads[:, [3, 0], :]

    # swap C and G
    grads[:, [1, 2], :] = grads[:, [2, 1], :]

  # undo shift
  if shift < 0:
    # shift sequence right
    grads[-shift:, :, :] = grads[:shift, :, :]

    # fill in left unknowns
    grads[:-shift, :, :] = 0

  elif shift > 0:
    # shift sequence left
    grads[:-shift, :, :] = grads[shift:, :, :]

    # fill in right unknowns
    grads[-shift:, :, :] = 0

  return grads


def make_seq_1hot(genome_open, chrm, start, end, seq_len):
  if start < 0:
    seq_dna = 'N'*(-start) + genome_open.fetch(chrm, 0, end)
  else:
    seq_dna = genome_open.fetch(chrm, start, end)
    
  # extend to full length
  if len(seq_dna) < seq_len:
    seq_dna += 'N'*(seq_len-len(seq_dna))

  seq_1hot = dna_io.dna_1hot(seq_dna)
  return seq_1hot

################################################################################
# __main__
# ###############################################################################
if __name__ == '__main__':
  main()
