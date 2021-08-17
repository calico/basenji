#!/usr/bin/env python
from optparse import OptionParser

from collections import OrderedDict
import json
import os
import pdb
import sys

import numpy as np
import pickle
import pybedtools
import pysam
import tensorflow as tf

import pygene
from basenji import dna_io
from basenji import rnann
from basenji import vcf as bvcf
from basenji_sad_ref import make_alt_1hot

'''
saluki_ssd.py

Notes:
-trimming the RNA can result in removing the variant and producing a zero.
-consider removing NMD, splice sites, etc, from eQTL task.
'''

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params> <model> <vcf>'
  parser = OptionParser(usage)
  parser.add_option('-f', dest='genome_fasta',
    default='%s/data/hg38.fa' % os.environ['BASENJIDIR'],
    help='Genome FASTA for sequences [Default: %default]')
  parser.add_option('-g', dest='genes_gtf',
    default='/home/drk/rnaml/data/genes/gencode36_saluki.gtf',
    help='Genes GTF [Default: %default]')
  parser.add_option('-o', dest='out_dir',
    default='ssd',
    help='Output directory for tables and plots [Default: %default]')
  parser.add_option('--shifts', dest='shifts',
    default='0', type='str',
    help='Ensemble prediction shifts [Default: %default]')
  (options,args) = parser.parse_args()

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

  else:
    parser.error('Must provide parameters, model, and VCF')

  options.shifts = [int(shift) for shift in options.shifts.split(',')]

  os.makedirs(options.out_dir, exist_ok=True)

  #################################################################
  # variants to transcripts

  # hash variant identifier to a list of (saluki_score, transcript_id, gene_id)
  variant_transcripts = {}
  variants_bedt = pybedtools.BedTool(vcf_file)
  transcripts_bedt = pybedtools.BedTool(options.genes_gtf)
  for overlap in variants_bedt.intersect(transcripts_bedt, wo=True):
    if overlap[-8] == 'exon':
      vid = overlap[2]
      gene_kv = pygene.gtf_kv(overlap[-2])
      gene_tup = (gene_kv['saluki_score'], gene_kv['transcript_id'], gene_kv['gene_id'])
      variant_transcripts.setdefault(vid,[]).append(gene_tup)
    
  # assign variants to transcripts
  transcript_variants = OrderedDict()
  for vid in variant_transcripts:
    tid = sorted(variant_transcripts[vid])[-1][1]
    transcript_variants.setdefault(tid,[]).append(vid)

  print('Variants overlapping transcripts: %d' % len(variant_transcripts))
  print('Transcripts overlapping variants: %d' % len(transcript_variants))

  #################################################################
  # parameters and model

  # read model parameters
  with open(params_file) as params_open:
    params = json.load(params_open)
  params_model = params['model']
  params_train = params['train']

  # setup model
  seqnn_model = rnann.RnaNN(params_model)
  seqnn_model.restore(model_file)
  # seqnn_model.build_ensemble(False, options.shifts)

  #################################################################
  # generator

  # read variants
  variants_list = bvcf.vcf_snps(vcf_file)
  variants_dict = dict([(v.rsid,v) for v in variants_list])

  # read genes
  genes_gtf = pygene.GTF(options.genes_gtf)

  # note: VCF and GTF use 1-based indexing

  # open genome FASTA
  genome_open = pysam.Fastafile(options.genome_fasta)

  def seq_gen():
    for tx_id, tx_variants in transcript_variants.items():
      transcript = genes_gtf.transcripts[tx_id]

      # skip features
      # tx_features = np.array([len(transcript.exons)], dtype='float32')

      # encode RNA input
      tx_dna, tx_seq1 = rna_1hot_splice(transcript, genome_open)

      # correct strand
      if transcript.strand == '-':
        tx_seq1_yield = dna_io.hot1_rc(tx_seq1)
      else:
        tx_seq1_yield = tx_seq1

      # pad/trim to length
      tx_seq1_yield = set_rna_length(tx_seq1_yield, params_model['seq_length'])

      # yield
      yield tx_seq1_yield #, tx_features

      # for each variant
      for vid in tx_variants:
        variant = variants_dict[vid]

        # map genome position to transcript position
        variant.pos_tx = 0
        for exon in transcript.exons:
          if exon.start <= variant.pos <= exon.end:
            variant.pos_tx += variant.pos - exon.start
            break
          else:
            variant.pos_tx += exon.end - exon.start + 1

        if variant.ref_allele != tx_dna[variant.pos_tx]:
          print('%s ref allele %s does not match transcriptome %s' % \
            (variant.rsid, variant.ref_allele, tx_dna[variant.pos_tx]), file=sys.stderr)
          exit(1)

          # cannot flip alleles using this approach
          # if variant.alt_allele == tx_dna[variant.pos_tx]:
          #   variant.flip_alleles()
          # else:
          #   # bail
          #   print('%s ref allele %s and alt allele %s do not match transcriptome %s' % \
          #   (variant.rsid, variant.ref_allele, variant.alt_allele, tx_dna[variant.pos_tx]), file=sys.stderr)


        # substitute alternative allele
        vtx_seq1 = make_alt_1hot(tx_seq1, variant.pos_tx,
          variant.ref_allele, variant.alt_allele)

        # correct strand
        if transcript.strand == '-':
          vtx_seq1_yield = dna_io.hot1_rc(vtx_seq1)
        else:
          vtx_seq1_yield = vtx_seq1

        # pad/trim to length
        vtx_seq1_yield = set_rna_length(vtx_seq1_yield, params_model['seq_length'])

        # yield
        yield vtx_seq1_yield #, tx_features


  ################################################################
  # predict SNP scores, write output

  # initialize output table
  ssd_out = open('%s/ssd.tsv' % options.out_dir, 'w')
  headers = ['variant', 'transcript', 'SSD']
  print('\t'.join(headers), file=ssd_out)

  # initialize predictions stream
  preds_stream = PredStreamGen(seqnn_model, seq_gen(), params['train']['batch_size'])

  # predictions index
  pi = 0

  for tx_id, tx_variants in transcript_variants.items():
    # get reference prediction
    ref_preds = preds_stream[pi]
    pi += 1

    # for each variant
    for vid in tx_variants:
      # get alternative prediction
      alt_preds = preds_stream[pi]
      pi += 1

      # compute SNP stability difference
      ssd = alt_preds - ref_preds
      print(ssd.shape)

      # print
      cols = [vid, tx_id, '%.6f'%ssd]
      print('\t'.join(cols), file=ssd_out)

  # close genome
  genome_open.close()

  # close SNP output
  ssd_out.close()

def rna_1hot_splice(transcript, genome_open):
  """Extract RNA nucleotides and add splice and coding frame channels.
      Consider strand, but do not yet reverse complement, because its
      easier to handle variant substitution."""

   # exons RNA
  tx_exons = transcript.fasta_exons(genome_open, stranded=False)
  seq_len = len(tx_exons)

  # CDS RNA
  tx_cds = transcript.fasta_cds(genome_open, stranded=False)
  if len(tx_cds) % 3 == 0:
    valid_coding = True
  else:
    # print('WARNING: %d mod 3 == %d length:' % (len(tx_cds),len(tx_cds)%3), transcript)
    valid_coding = False
  cds_start = tx_exons.find(tx_cds)
  assert(cds_start != -1)
  cds_end = cds_start + len(tx_cds)

  # add frame channel
  frame_channel = np.zeros((seq_len,1))
  if valid_coding:
    aa_len = (cds_end - cds_start) // 3
    if transcript.strand == '+':
      frame_channel[cds_start:cds_end,0] = np.tile([1,0,0], aa_len)
    else:
      frame_channel[cds_start:cds_end,0] = np.tile([0,0,1], aa_len)

  # splice track
  splice_channel = np.zeros((seq_len,1))
  ti = 0
  exon_lens = [ex.end-ex.start for ex in transcript.exons]
  for el in exon_lens:
    if transcript.strand == '+':
      ti += el
      splice_channel[ti-1,0] = 1
    else:
      splice_channel[ti,0] = 1
      ti += el

  # 1 hot encode
  tx_seq1 = dna_io.dna_1hot(tx_exons, n_uniform=True)
  tx_seq1 = np.concatenate([tx_seq1, frame_channel, splice_channel], axis=-1)

  return tx_exons, tx_seq1.astype('float32')


def rna_1hot_splice_vikram(transcript, genome_open):
  """Extract RNA nucleotides and add splice and coding frame channels.
      Consider strand, but do not yet reverse complement, because its
      easier to handle variant substitution."""

  # exons RNA
  tx_exons = transcript.fasta_exons(genome_open, stranded=False)
  seq_len = len(tx_exons)

  # CDS RNA
  tx_cds = transcript.fasta_cds(genome_open, stranded=False)
  if len(tx_cds) % 3 != 0:
    pdb.set_trace()
  assert(len(tx_cds) % 3 == 0)
  cds_start = tx_exons.find(tx_cds)
  assert(cds_start != -1)
  cds_end = cds_start + len(tx_cds)

  # add frame channel
  frame_channel = np.zeros((seq_len,1))
  aa_len = (cds_end - cds_start) // 3
  if transcript.strand == '+':
    frame_channel[cds_start:cds_end,0] = np.tile([1,0,0], aa_len)
  else:
    frame_channel[cds_start:cds_end,0] = np.tile([0,0,1], aa_len)

  # splice tracks
  splice5_channel = np.zeros((seq_len,1))
  splice3_channel = np.zeros((seq_len,1))
  ti = 0
  exon_lens = [ex.end-ex.start for ex in transcript.exons]
  for el in exon_lens:
    if transcript.strand == '+':
      splice5_channel[ti,0] = 1
      ti += el
      splice3_channel[ti-1,0] = 1
    else:
      splice3_channel[ti,0] = 1
      ti += el
      splice5_channel[ti-1,0] = 1

  # 1 hot encode
  tx_seq1 = dna_io.dna_1hot(tx_exons, n_uniform=True)
  tx_seq1 = np.concatenate([tx_seq1, frame_channel, splice5_channel, splice3_channel], axis=-1)

  return tx_exons, tx_seq1.astype('float32')



def rna_1hot_cds(transcript, genome_open):
  """add cds channel"""

  # exons RNA
  tx_exons = transcript.fasta_exons(genome_open)
  seq_len = len(tx_exons)

  # CDS RNA
  tx_cds = transcript.fasta_cds(genome_open)
  cds_start = tx_exons.find(tx_cds)
  assert(cds_start != -1)
  cds_end = cds_start + len(tx_cds)

  # add CDS channel
  cds_channel = np.zeros((seq_len,1))
  cds_channel[cds_start:cds_end] = 1

  # 1 hot encode
  tx_seq1 = dna_io.dna_1hot(tx_exons, n_uniform=True)
  tx_seq1 = np.concatenate([tx_seq1, cds_channel], axis=-1)

  return tx_exons, tx_seq1.astype('float32')

def set_rna_length(tx_seq1, length):
  """Set RNA length to specific value by padding or trimming."""
  seq_len, seq_depth = tx_seq1.shape

  if seq_len < length:
    # pad
    pad_len = length - seq_len
    pad_seq1 = np.zeros((pad_len,seq_depth))
    tx_seq1 = np.concatenate([tx_seq1,pad_seq1], axis=0)

  elif seq_len > length:
    # trim 5'
    tx_seq1 = tx_seq1[-length:]

  return tx_seq1

class PredStreamGen:
  """ Interface to acquire predictions via a buffered stream mechanism
        rather than getting them all at once and using excessive memory.
        Accepts generator and constructs stream batches from it. """
  def __init__(self, model, seqs_gen, batch_size, stream_seqs=256, verbose=False):
    self.model = model
    self.seqs_gen = seqs_gen
    self.stream_seqs = stream_seqs
    self.batch_size = batch_size
    self.verbose = verbose

    self.stream_start = 0
    self.stream_end = 0

  def __getitem__(self, i):
    # acquire predictions, if needed
    if i >= self.stream_end:
      # update start
      self.stream_start = self.stream_end

      if self.verbose:
        print('Predicting from %d' % self.stream_start, flush=True)

      # predict
      self.stream_preds = self.model.predict(self.make_dataset())

      # update end
      self.stream_end = self.stream_start + self.stream_preds.shape[0]

    return self.stream_preds[i - self.stream_start]

  def make_dataset(self):
    """ Construct Dataset object for this stream chunk. """
    seqs_1hot = []
    stream_end = self.stream_start+self.stream_seqs
    for si in range(self.stream_start, stream_end):
      try:
        seq_1hot = self.seqs_gen.__next__()
        seqs_1hot.append(seq_1hot)
      except StopIteration:
        continue

    seqs_1hot = np.array(seqs_1hot)

    dataset = tf.data.Dataset.from_tensor_slices((seqs_1hot,))
    dataset = dataset.batch(self.batch_size)

    return dataset


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
