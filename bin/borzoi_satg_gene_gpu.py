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

'''
borzoi_satg_gene_gpu.py

Perform a gradient saliency analysis for genes specified in a GTF file (GPU-friendly).
'''

# tf code for predicting raw sum-of-expression counts on GPU
@tf.function
def _count_func(model, seq_1hot, target_slice, pos_slice, pos_mask=None, track_scale=1., track_transform=1., clip_soft=None, use_mean=False) :
      
      # predict
      preds = tf.gather(model(seq_1hot, training=False), target_slice, axis=-1, batch_dims=1)
      
      # undo scale
      preds = preds / track_scale

      # undo soft_clip
      if clip_soft is not None :
        preds = tf.where(preds > clip_soft, (preds - clip_soft)**2 + clip_soft, preds)

      # undo sqrt
      preds = preds**(1. / track_transform)
      
      # aggregate over tracks (average)
      preds = tf.reduce_mean(preds, axis=-1)
      
      # slice specified positions
      preds_slice = tf.gather(preds, pos_slice, axis=-1, batch_dims=1)
      if pos_mask is not None :
        preds_slice = preds_slice * pos_mask
      
      # aggregate over positions
      if not use_mean :
        preds_agg = tf.reduce_sum(preds_slice, axis=-1)
      else :
        if pos_mask is not None :
          preds_agg = tf.reduce_sum(preds_slice, axis=-1) / tf.reduce_sum(pos_mask, axis=-1)
        else :
          preds_agg = tf.reduce_mean(preds_slice, axis=-1)

      return preds_agg

# code for getting model predictions from a tensor of input sequence patterns
def predict_counts(seqnn_model, seq_1hot, head_i=None, target_slice=None, pos_slice=None, pos_mask=None, chunk_size=None, batch_size=1, track_scale=1., track_transform=1., clip_soft=None, use_mean=False, dtype='float32'):
    
    # start time
    t0 = time.time()
    
    # choose model
    if seqnn_model.ensemble is not None:
      model = seqnn_model.ensemble
    elif head_i is not None:
      model = seqnn_model.models[head_i]
    else:
      model = seqnn_model.model
    
    # verify tensor shape(s)
    seq_1hot = seq_1hot.astype('float32')
    target_slice = np.array(target_slice).astype('int32')
    pos_slice = np.array(pos_slice).astype('int32')
    
    # convert constants to tf tensors
    track_scale = tf.constant(track_scale, dtype=tf.float32)
    track_transform = tf.constant(track_transform, dtype=tf.float32)
    if clip_soft is not None :
        clip_soft = tf.constant(clip_soft, dtype=tf.float32)
    
    if pos_mask is not None :
      pos_mask = np.array(pos_mask).astype('float32')
    
    if len(seq_1hot.shape) < 3:
      seq_1hot = seq_1hot[None, ...]
    
    if len(target_slice.shape) < 2:
      target_slice = target_slice[None, ...]
    
    if len(pos_slice.shape) < 2:
      pos_slice = pos_slice[None, ...]
    
    if pos_mask is not None and len(pos_mask.shape) < 2:
      pos_mask = pos_mask[None, ...]
    
    # chunk parameters
    num_chunks = 1
    if chunk_size is None :
      chunk_size = seq_1hot.shape[0]
    else :
      num_chunks = int(np.ceil(seq_1hot.shape[0] / chunk_size))
    
    # loop over chunks
    pred_chunks = []
    for ci in range(num_chunks) :
      
      # collect chunk
      seq_1hot_chunk = seq_1hot[ci * chunk_size:(ci+1) * chunk_size, ...]
      target_slice_chunk = target_slice[ci * chunk_size:(ci+1) * chunk_size, ...]
      pos_slice_chunk = pos_slice[ci * chunk_size:(ci+1) * chunk_size, ...]
      
      pos_mask_chunk = None
      if pos_mask is not None :
        pos_mask_chunk = pos_mask[ci * chunk_size:(ci+1) * chunk_size, ...]
      
      actual_chunk_size = seq_1hot_chunk.shape[0]
      
      # convert to tf tensors
      seq_1hot_chunk = tf.convert_to_tensor(seq_1hot_chunk, dtype=tf.float32)
      target_slice_chunk = tf.convert_to_tensor(target_slice_chunk, dtype=tf.int32)
      pos_slice_chunk = tf.convert_to_tensor(pos_slice_chunk, dtype=tf.int32)
      
      if pos_mask is not None :
        pos_mask_chunk = tf.convert_to_tensor(pos_mask_chunk, dtype=tf.float32)
      
      # batching parameters
      num_batches = int(np.ceil(actual_chunk_size / batch_size))
      
      # loop over batches
      pred_batches = []
      for bi in range(num_batches) :
        
        # collect batch
        seq_1hot_batch = seq_1hot_chunk[bi * batch_size:(bi+1) * batch_size, ...]
        target_slice_batch = target_slice_chunk[bi * batch_size:(bi+1) * batch_size, ...]
        pos_slice_batch = pos_slice_chunk[bi * batch_size:(bi+1) * batch_size, ...]
        
        pos_mask_batch = None
        if pos_mask is not None :
          pos_mask_batch = pos_mask_chunk[bi * batch_size:(bi+1) * batch_size, ...]

        pred_batch = _count_func(model, seq_1hot_batch, target_slice_batch, pos_slice_batch, pos_mask_batch, track_scale, track_transform, clip_soft, use_mean).numpy().astype(dtype)
      
        pred_batches.append(pred_batch)
    
      # concat predicted batches
      preds = np.concatenate(pred_batches, axis=0)
    
      pred_chunks.append(preds)
      
      # collect garbage
      gc.collect()
    
    # concat predicted chunks
    preds = np.concatenate(pred_chunks, axis=0)
    
    print('Made predictions in %ds' % (time.time()-t0))

    return preds


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
  parser.add_option('--smoothgrad', dest='smooth_grad',
      default=0, type='int',
      help='Run smoothgrad [Default: %default]')
  parser.add_option('--samples', dest='n_samples',
      default=5, type='int',
      help='Number of smoothgrad samples [Default: %default]')
  parser.add_option('--sampleprob', dest='sample_prob',
      default=0.875, type='float',
      help='Probability of not mutating a position in smoothgrad [Default: %default]')
  parser.add_option('--clip_soft', dest='clip_soft',
      default=None, type='float',
      help='Model clip_soft setting [Default: %default]')
  parser.add_option('--no_transform', dest='no_transform',
      default=0, type='int',
      help='Run gradients with no inverse transforms [Default: %default]')
  parser.add_option('--get_preds', dest='get_preds',
      default=0, type='int',
      help='Store scalar predictions in addition to their gradients [Default: %default]')
  parser.add_option('--pseudo_qtl', dest='pseudo_qtl',
      default=None, type='float',
      help='Quantile of predicted scalars to choose as pseudo count [Default: %default]')
  parser.add_option('--pseudo_tissue', dest='pseudo_tissue',
      default=None, type='str',
      help='Tissue to filter genes on when calculating pseudo count [Default: %default]')
  parser.add_option('--gene_file', dest='gene_file',
      default=None, type='str',
      help='Csv-file of gene metadata [Default: %default]')
  parser.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')
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
  num_targets = 1

  #Load gene dataframe and select tissue
  tissue_genes = None
  if options.gene_file is not None and options.pseudo_tissue is not None :
    gene_df = pd.read_csv(options.gene_file, sep='\t')
    gene_df = gene_df.query("tissue == '" + str(options.pseudo_tissue) + "'").copy().reset_index(drop=True)
    gene_df = gene_df.drop(columns=['Unnamed: 0'])

    #Get list of gene for tissue
    tissue_genes = gene_df['gene_base'].values.tolist()

    print("len(tissue_genes) = " + str(len(tissue_genes)))
  
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

  #################################################################
  # setup output

  min_start = -model_stride*model_crop

  # choose gene sequences
  genes_chr = []
  genes_start = []
  genes_end = []
  genes_strand = []
  for gene_id in gene_list:
    gene = transcriptome.genes[gene_id]
    genes_chr.append(gene.chrom)
    genes_strand.append(gene.strand)

    gene_midpoint = gene.midpoint()
    gene_start = max(min_start, gene_midpoint - seq_len//2)
    gene_end = gene_start + seq_len
    genes_start.append(gene_start)
    genes_end.append(gene_end)

  #################################################################
  # predict scores, write output
  
  buffer_size = 1024
  
  print("clip_soft = " + str(options.clip_soft))
  
  print("n genes = " + str(len(genes_chr)))
  
  # loop over folds
  for fold_ix in options.folds :
    print("-- Fold = " + str(fold_ix) + " --")
    
    # (re-)initialize HDF5
    scores_h5_file = '%s/scores_f%dc0.h5' % (options.out_dir, fold_ix)
    if os.path.isfile(scores_h5_file):
      os.remove(scores_h5_file)
    scores_h5 = h5py.File(scores_h5_file, 'w')
    scores_h5.create_dataset('seqs', dtype='bool',
        shape=(num_genes, seq_len, 4))
    scores_h5.create_dataset('grads', dtype='float16',
        shape=(num_genes, seq_len, 4, num_targets))
    if options.get_preds == 1 :
      scores_h5.create_dataset('preds', dtype='float32',
        shape=(num_genes, num_targets))
    scores_h5.create_dataset('gene', data=np.array(gene_list, dtype='S'))
    scores_h5.create_dataset('chr', data=np.array(genes_chr, dtype='S'))
    scores_h5.create_dataset('start', data=np.array(genes_start))
    scores_h5.create_dataset('end', data=np.array(genes_end))
    scores_h5.create_dataset('strand', data=np.array(genes_strand, dtype='S'))
    
    # load model fold
    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(model_folder + "/f" + str(fold_ix) + "c0/model0_best.h5", 0, by_name=False)
    seqnn_model.build_slice(targets_df.index, False)
  
    track_scale = targets_df.iloc[0]['scale']
    track_transform = 3. / 4.
    
    # optionally get (and store) scalar predictions before computing their gradients
    if options.get_preds == 1 :
      print(" - (prediction) - ", flush=True)
      
      for shift in options.shifts :
        print('Processing shift %d' % shift, flush=True)
  
        for rev_comp in ([False, True] if options.rc == 1 else [False]) :
  
          if options.rc == 1 :
            print('Fwd/rev = %s' % ('fwd' if not rev_comp else 'rev'), flush=True)
  
          seq_1hots = []
          gene_slices = []
          gene_targets = []
  
          for gi, gene_id in enumerate(gene_list):
            
            if gi % 500 == 0 :
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
  
            if rev_comp:
              seq_1hot = dna_io.hot1_rc(seq_1hot)
              gene_slice = target_length - gene_slice - 1
  
            # slice relevant strand targets
            if genes_strand[gi] == '+':
              gene_strand_mask = (targets_df.strand != '-') if not rev_comp else (targets_df.strand != '+')
            else:
              gene_strand_mask = (targets_df.strand != '+') if not rev_comp else (targets_df.strand != '-')
  
            gene_target = np.array(targets_df.index[gene_strand_mask].values)
  
            # accumulate data tensors
            seq_1hots.append(seq_1hot[None, ...])
            gene_slices.append(gene_slice[None, ...])
            gene_targets.append(gene_target[None, ...])
            
            if gi == len(gene_list) - 1 or len(seq_1hots) >= buffer_size :
  
              # concat sequences
              seq_1hots = np.concatenate(seq_1hots, axis=0)
  
              # pad gene slices to same length (mark valid positions in mask tensor)
              max_slice_len = int(np.max([gene_slice.shape[1] for gene_slice in gene_slices]))
  
              gene_masks = np.zeros((len(gene_slices), max_slice_len), dtype='float32')
              gene_slices_padded = np.zeros((len(gene_slices), max_slice_len), dtype='int32')
              for gii, gene_slice in enumerate(gene_slices) :
                for j in range(gene_slice.shape[1]) :
                  gene_masks[gii, j] = 1.
                  gene_slices_padded[gii, j] = gene_slice[0, j]
  
              gene_slices = gene_slices_padded
  
              # concat gene-specific targets
              gene_targets = np.concatenate(gene_targets, axis=0)
  
              # batch call count predictions
              preds = predict_counts(
                seqnn_model,
                seq_1hots,
                head_i=0,
                target_slice=gene_targets,
                pos_slice=gene_slices,
                pos_mask=gene_masks,
                chunk_size=buffer_size,
                batch_size=1,
                track_scale=track_scale,
                track_transform=track_transform,
                clip_soft=options.clip_soft,
                use_mean=False,
                dtype='float32'
              )
  
              # save predictions
              for gii, gene_slice in enumerate(gene_slices) :
                h5_gi = (gi // buffer_size) * buffer_size + gii
  
                # write to HDF5
                scores_h5['preds'][h5_gi, :] += (preds[gii] / float(len(options.shifts)))
  
              #clear sequence buffer
              seq_1hots = []
              gene_slices = []
              gene_targets = []
              
              # collect garbage
              gc.collect()

    # optionally set pseudo count from predictions
    pseudo_count = 0.
    if options.pseudo_qtl is not None :
      gene_preds = scores_h5['preds'][:]
      
      # filter on tissue
      tissue_preds = None
      
      if tissue_genes is not None :
        tissue_set = set(tissue_genes)
      
        # get subset of genes and predictions belonging to the pseudo count tissue
        tissue_preds = []
        for gi, gene_id in enumerate(gene_list) :
          if gene_id.split(".")[0] in tissue_set :
            tissue_preds.append(gene_preds[gi, 0])
      
        tissue_preds = np.array(tissue_preds, dtype='float32')
      else :
        tissue_preds = np.array(gene_preds[:, 0], dtype='float32')
      
      print("tissue_preds.shape[0] = " + str(tissue_preds.shape[0]))
      
      print("np.min(tissue_preds) = " + str(np.min(tissue_preds)))
      print("np.max(tissue_preds) = " + str(np.max(tissue_preds)))
      
      # set pseudo count based on quantile of predictions
      pseudo_count = np.quantile(tissue_preds, q=options.pseudo_qtl)
      
      print("")
      print("pseudo_count = " + str(round(pseudo_count, 6)))
    
    # compute gradients
    print(" - (gradients) - ", flush=True)
    
    for shift in options.shifts :
      print('Processing shift %d' % shift, flush=True)

      for rev_comp in ([False, True] if options.rc == 1 else [False]) :

        if options.rc == 1 :
          print('Fwd/rev = %s' % ('fwd' if not rev_comp else 'rev'), flush=True)

        seq_1hots = []
        gene_slices = []
        gene_targets = []

        for gi, gene_id in enumerate(gene_list):
          
          if gi % 500 == 0 :
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

          if rev_comp:
            seq_1hot = dna_io.hot1_rc(seq_1hot)
            gene_slice = target_length - gene_slice - 1

          # slice relevant strand targets
          if genes_strand[gi] == '+':
            gene_strand_mask = (targets_df.strand != '-') if not rev_comp else (targets_df.strand != '+')
          else:
            gene_strand_mask = (targets_df.strand != '+') if not rev_comp else (targets_df.strand != '-')

          gene_target = np.array(targets_df.index[gene_strand_mask].values)

          # accumulate data tensors
          seq_1hots.append(seq_1hot[None, ...])
          gene_slices.append(gene_slice[None, ...])
          gene_targets.append(gene_target[None, ...])
          
          if gi == len(gene_list) - 1 or len(seq_1hots) >= buffer_size :

            # concat sequences
            seq_1hots = np.concatenate(seq_1hots, axis=0)

            # pad gene slices to same length (mark valid positions in mask tensor)
            max_slice_len = int(np.max([gene_slice.shape[1] for gene_slice in gene_slices]))

            gene_masks = np.zeros((len(gene_slices), max_slice_len), dtype='float32')
            gene_slices_padded = np.zeros((len(gene_slices), max_slice_len), dtype='int32')
            for gii, gene_slice in enumerate(gene_slices) :
              for j in range(gene_slice.shape[1]) :
                gene_masks[gii, j] = 1.
                gene_slices_padded[gii, j] = gene_slice[0, j]

            gene_slices = gene_slices_padded

            # concat gene-specific targets
            gene_targets = np.concatenate(gene_targets, axis=0)

            # batch call gradient computation
            grads = seqnn_model.gradients(
              seq_1hots,
              head_i=0,
              target_slice=gene_targets,
              pos_slice=gene_slices,
              pos_mask=gene_masks,
              chunk_size=buffer_size if options.smooth_grad != 1 else buffer_size // options.n_samples,
              batch_size=1,
              track_scale=track_scale,
              track_transform=track_transform,
              clip_soft=options.clip_soft,
              pseudo_count=pseudo_count,
              no_transform=options.no_transform == 1,
              use_mean=False,
              use_ratio=False,
              use_logodds=False,
              subtract_avg=True,
              input_gate=False,
              smooth_grad=options.smooth_grad == 1,
              n_samples=options.n_samples,
              sample_prob=options.sample_prob,
              dtype='float16'
            )

            # undo augmentations and save gradients
            for gii, gene_slice in enumerate(gene_slices) :
              grad = unaugment_grads(grads[gii, :, :, None], fwdrc=(not rev_comp), shift=shift)
              
              h5_gi = (gi // buffer_size) * buffer_size + gii

              # write to HDF5
              scores_h5['grads'][h5_gi] += grad

            #clear sequence buffer
            seq_1hots = []
            gene_slices = []
            gene_targets = []
            
            # collect garbage
            gc.collect()

    # save sequences and normalize gradients by total size of ensemble
    for gi, gene_id in enumerate(gene_list):
    
      # re-make original sequence
      seq_1hot = make_seq_1hot(genome_open, genes_chr[gi], genes_start[gi], genes_end[gi], seq_len)
      
      # write to HDF5
      scores_h5['seqs'][gi] = seq_1hot
      scores_h5['grads'][gi] /= float((len(options.shifts) * (2 if options.rc == 1 else 1)))
    
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
