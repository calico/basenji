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

import os
import pdb
import pickle
from queue import Queue
import random
import sys
from threading import Thread

import h5py
import numpy as np
import pandas as pd
import pysam
import tensorflow as tf

import basenji.dna_io as dna_io
from basenji import params
from basenji import seqnn
from basenji.stream import PredStream

'''
basenji_sat_bed.py

Perform an in silico saturation mutagenesis of sequences in a BED file.
'''

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <model_file> <bed_file>'
  parser = OptionParser(usage)
  parser.add_option('-f', dest='genome_fasta',
      default='%s/assembly/hg19.fa' % os.environ['HG19'],
      help='Genome FASTA for sequences [Default: %default]')
  parser.add_option('-l', dest='mut_len',
      default=200, type='int',
      help='Length of center sequence to mutate [Default: %default]')
  parser.add_option('-o', dest='out_dir',
      default='sat_mut', help='Output directory [Default: %default]')
  parser.add_option('--plots', dest='plots',
      default=False, action='store_true',
      help='Make heatmap plots [Default: %default]')
  parser.add_option('-p', dest='processes',
      default=None, type='int',
      help='Number of processes, passed by multi script')
  parser.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Ensemble forward and reverse complement predictions [Default: %default]')
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

  elif len(args) == 5:
    # multi worker
    options_pkl_file = args[0]
    params_file = args[1]
    model_file = args[2]
    bed_file = args[3]
    worker_index = int(args[4])

    # load options
    options_pkl = open(options_pkl_file, 'rb')
    options = pickle.load(options_pkl)
    options_pkl.close()

    # update output directory
    options.out_dir = '%s/job%d' % (options.out_dir, worker_index)
  else:
    parser.error('Must provide parameter and model files and BED file')

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

  options.shifts = [int(shift) for shift in options.shifts.split(',')]

  #################################################################
  # read parameters and collet target information

  job = params.read_job_params(params_file)

  if options.targets_file is None:
    target_ids = ['t%d' % ti for ti in range(job['num_targets'])]
    target_labels = ['']*len(target_ids)
    target_subset = None

  else:
    targets_df = pd.read_table(options.targets_file)
    target_ids = targets_df.identifier
    target_labels = targets_df.description
    target_subset = targets_df.index
    if len(target_subset) == job['num_targets']:
        target_subset = None

  num_targets = len(target_ids)

  #################################################################
  # sequence dataset

  # read sequences from BED
  seqs_dna, seqs_coords = bed_seqs(bed_file, options.genome_fasta, job['seq_length'])

  # filter for worker SNPs
  if options.processes is not None:
    worker_bounds = np.linspace(0, len(seqs_dna), options.processes+1, dtype='int')
    seqs_dna = seqs_dna[worker_bounds[worker_index]:worker_bounds[worker_index+1]]
    seqs_coords = seqs_coords[worker_bounds[worker_index]:worker_bounds[worker_index+1]]

  num_seqs = len(seqs_dna)

  # determine mutation region limits
  seq_mid = job['seq_length'] // 2
  mut_start = seq_mid - options.mut_len // 2
  mut_end = mut_start + options.mut_len

  # make data ops
  data_ops = satmut_data_ops(seqs_dna, mut_start, mut_end, job['batch_size'])

  #################################################################
  # setup model

  # build model
  model = seqnn.SeqNN()
  model.build_sad(job, data_ops, target_subset=target_subset,
                  ensemble_rc=options.rc, ensemble_shifts=options.shifts)

  #################################################################
  # setup output

  scores_h5_file = '%s/scores.h5' % options.out_dir
  if os.path.isfile(scores_h5_file):
    os.remove(scores_h5_file)
  scores_h5 = h5py.File('%s/scores.h5' % options.out_dir)
  scores_h5.create_dataset('scores', dtype='float16',
      shape=(num_seqs, options.mut_len, 4, num_targets))
  scores_h5.create_dataset('seqs', dtype='bool',
      shape=(num_seqs, options.mut_len, 4))

  # store mutagenesis sequence coordinates
  seqs_chr, seqs_start, _ = zip(*seqs_coords)
  seqs_chr = np.array(seqs_chr, dtype='S')
  seqs_start = np.array(seqs_start) + mut_start
  seqs_end = seqs_start + options.mut_len
  scores_h5.create_dataset('chrom', data=seqs_chr)
  scores_h5.create_dataset('start', data=seqs_start)
  scores_h5.create_dataset('end', data=seqs_end)

  preds_per_seq = 1 + 3*options.mut_len

  score_threads = []
  score_queue = Queue()
  for i in range(1):
    sw = ScoreWorker(score_queue, scores_h5)
    sw.start()
    score_threads.append(sw)

  #################################################################
  # predict scores, write output

  # initialize saver
  saver = tf.train.Saver()

  with tf.Session() as sess:
    # coordinator
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(coord=coord)

    # load variables into session
    saver.restore(sess, model_file)

    # initialize predictions stream
    preds_stream = PredStream(sess, model, 32)

    # predictions index
    pi = 0

    for si in range(num_seqs):
      print('Predicting %d' % si, flush=True)

      # collect sequence predictions
      seq_preds = []
      for spi in range(preds_per_seq):
        seq_preds.append(preds_stream[pi])
        pi += 1

      # wait for previous to finish
      score_queue.join()

      # queue sequence for scoring
      score_queue.put((seqs_dna[si], seq_preds, si))

      # queue sequence for plotting
      if options.plots:
        plot_queue.put((seqs_dna[si], seq_preds, si))

  # finish queue
  print('Waiting for threads to finish.', flush=True)
  score_queue.join()

  # close output HDF5
  scores_h5.close()


def bed_seqs(bed_file, fasta_file, seq_len):
  """Extract and extend BED sequences to seq_len."""
  fasta_open = pysam.Fastafile(fasta_file)

  seqs_dna = []
  seqs_coords = []

  for line in open(bed_file):
    a = line.split()
    chrm = a[0]
    start = int(a[1])
    end = int(a[2])

    # determine sequence limits
    mid = (start + end) // 2
    seq_start = mid - seq_len//2
    seq_end = seq_start + seq_len

    # save
    seqs_coords.append((chrm,seq_start,seq_end))

    # initialize sequence
    seq_dna = ''

    # add N's for left over reach
    if seq_start < 0:
      print('Adding %d Ns to %s:%d-%s' % \
          (-seq_start,chrm,start,end), file=sys.stderr)
      seq_dna = 'N'*(-seq_start)
      seq_start = 0

    # get dna
    seq_dna += fasta_open.fetch(chrm, seq_start, seq_end).upper()

    # add N's for right over reach
    if len(seq_dna) < seq_len:
      print('Adding %d Ns to %s:%d-%s' % \
          (seq_len-len(seq_dna),chrm,start,end), file=sys.stderr)
      seq_dna += 'N'*(seq_len-len(seq_dna))

    # randomly set all N's
    seq_dna = list(seq_dna)
    for i in range(len(seq_dna)):
      if seq_dna[i] == 'N':
        seq_dna[i] = random.choice('ACGT')
    seq_dna = ''.join(seq_dna)

    # append
    seqs_dna.append(seq_dna)

  fasta_open.close()

  return seqs_dna, seqs_coords


def satmut_data_ops(seqs_dna, mut_start, mut_end, batch_size):
  """Construct 1 hot encoded saturation mutagenesis DNA sequences
      using tf.data."""

  # make sequence generator
  def seqs_gen():
    for seq_dna in seqs_dna:
      # 1 hot code DNA
      seq_1hot = dna_io.dna_1hot(seq_dna)
      yield {'sequence':seq_1hot}

      # for mutation positions
      for mi in range(mut_start, mut_end):
        # for each nucleotide
        for ni in range(4):
          # if non-reference
          if seq_1hot[mi,ni] == 0:
            # copy and modify
            seq_mut_1hot = np.copy(seq_1hot)
            seq_mut_1hot[mi,:] = 0
            seq_mut_1hot[mi,ni] = 1
            yield {'sequence':seq_mut_1hot}

  # auxiliary info
  seq_len = len(seqs_dna[0])
  seqs_types = {'sequence': tf.float32}
  seqs_shapes = {'sequence': tf.TensorShape([tf.Dimension(seq_len),
                                            tf.Dimension(4)])}

  # create dataset
  dataset = tf.data.Dataset().from_generator(seqs_gen,
                                             output_types=seqs_types,
                                             output_shapes=seqs_shapes)
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(2*batch_size)

  # make iterator ops
  iterator = dataset.make_one_shot_iterator()
  data_ops = iterator.get_next()

  return data_ops


class PlotWorker(Thread):
  """Compute summary statistics and write to HDF."""
  def __init__(self, plot_queue, out_dir):
    Thread.__init__(self)
    self.queue = plot_queue
    self.daemon = True
    self.out_dir = out_dir

  def run(self):
    while True:
      # unload predictions
      seq_dna, seq_preds, si = self.queue.get()
      print('Plotting %d' % si, flush=True)

      # communicate finished task
      self.queue.task_done()


class ScoreWorker(Thread):
  """Compute summary statistics and write to HDF."""
  def __init__(self, score_queue, scores_h5):
    Thread.__init__(self)
    self.queue = score_queue
    self.daemon = True
    self.scores_h5 = scores_h5

  def run(self):
    while True:
      try:
        # unload predictions
        seq_dna, seq_preds, si = self.queue.get()
        print('Writing %d' % si, flush=True)

        # seq_preds is (1 + 3*mut_len) x (target_len) x (num_targets)
        seq_preds = np.array(seq_preds)
        num_preds = seq_preds.shape[0]
        num_targets = seq_preds.shape[-1]

        # reverse engineer mutagenesis position parameters
        mut_len = (num_preds - 1) // 3
        mut_mid = len(seq_dna) // 2
        mut_start = mut_mid - mut_len//2
        mut_end = mut_start + mut_len

        # one hot code mutagenized DNA
        seq_dna_mut = seq_dna[mut_start:mut_end]
        seq_1hot_mut = dna_io.dna_1hot(seq_dna_mut)

        # initialize scores
        seq_scores = np.zeros((mut_len, 4, num_targets), dtype='float32')

        # sum across length
        seq_preds_sum = seq_preds.sum(axis=1, dtype='float32')

        # predictions index (starting at first mutagenesis)
        pi = 1

        # for each mutated position
        for mi in range(mut_len):
          # for each nucleotide
          for ni in range(4):
            if seq_1hot_mut[mi,ni]:
              # reference score
              seq_scores[mi,ni,:] = seq_preds_sum[0,:]
            else:
              # mutation score
              seq_scores[mi,ni,:] = seq_preds_sum[pi,:]
              pi += 1

        # normalize positions
        seq_scores -= seq_scores.mean(axis=1, keepdims=True)

        # write to HDF5
        self.scores_h5['scores'][si,:,:,:] = seq_scores.astype('float16')
        self.scores_h5['seqs'][si,:,:] = seq_1hot_mut

      except:
        # communicate error
        print('ERROR: Sequence %d failed' % si, file=sys.stderr, flush=True)

      # communicate finished task
      self.queue.task_done()


def expand_4l(sat_lg_ti, seq_1hot):
  """ Expand

    In:
        sat_lg_ti (l array): Sat mut loss/gain scores for a single sequence and
        target.
        seq_1hot (Lx4 array): One-hot coding for a single sequence.

    Out:
        sat_loss_4l (lx4 array): Score-hot coding?

    """

  # determine satmut length
  satmut_len = sat_lg_ti.shape[0]

  # jump to satmut region in one hot coded sequence
  ssi = int((seq_1hot.shape[0] - satmut_len) // 2)

  # filter sequence for satmut region
  seq_1hot_sm = seq_1hot[ssi:ssi + satmut_len, :]

  # tile loss scores to align
  sat_lg_tile = np.tile(sat_lg_ti, (4, 1)).T

  # element-wise multiple
  sat_lg_4l = np.multiply(seq_1hot_sm, sat_lg_tile)

  return sat_lg_4l


def delta_matrix(seqs_1hot, sat_preds, satmut_len):
  """ Compute the matrix of prediction deltas

    Args:
        seqs_1hot (Lx4 array): One-hot coding of all sequences.
        sat_preds: (SMxLxT array): Satmut sequence predictions.
        satmut_len: Saturated mutagenesis region length.

    Returns:
        sat_delta (4 x L_sm x T array): Delta matrix for saturated mutagenesis
        region.

    Todo:
        -Rather than computing the delta as the change at that nucleotide's
        prediction,
            compute it as the mean change across the sequence. That way, we
            better
            pick up on motif-flanking interactions.
    """
  seqs_n = int(sat_preds.shape[0] / (1 + 3 * satmut_len))
  num_targets = sat_preds.shape[2]

  # left-over from previous version
  # we're just expecting one sequence now
  si = 0

  # initialize
  sat_delta = np.zeros((4, satmut_len, num_targets), dtype='float64')

  # jump to si's mutated sequences
  smi = seqs_n + si * 3 * satmut_len

  # jump to satmut region in preds (incorrect with target pooling)
  # spi = int((sat_preds.shape[1] - satmut_len) // 2)

  # jump to satmut region in sequence
  ssi = int((seqs_1hot.shape[0] - satmut_len) // 2)

  # to study across sequence length
  # sat_delta_length = np.zeros((4, satmut_len, sat_preds.shape[1], num_targets))

  # compute delta matrix
  for li in range(satmut_len):
    for ni in range(4):
      if seqs_1hot[ssi + li, ni] == 1:
        sat_delta[ni, li, :] = 0
      else:
        # to study across sequence length
        # sat_delta_length[ni,li,:,:] = sat_preds[smi] - sat_preds[si]

        # sat_delta[ni,li,:] = sat_preds[smi,spi+li,:] - sat_preds[si,spi+li,:]
        sat_delta[ni, li, :] = sat_preds[smi].sum(
            axis=0, dtype='float64') - sat_preds[si].sum(
                axis=0, dtype='float64')
        smi += 1

  # to study across sequence length
  """
    sat_delta_length = sat_delta_length.mean(axis=0)

    if not os.path.isdir('length'):
        os.mkdir('length')
    for ti in range(sat_delta_length.shape[2]):
        plt.figure()
        sns.heatmap(sat_delta_length[:,:,ti], linewidths=0, cmap='RdBu_r')
        plt.savefig('length/delta_length_s%d_t%d.pdf' % (si,ti))
        plt.close()
    """

  return sat_delta


def loss_gain(sat_delta, sat_preds_si, satmut_len):
  # compute min and max
  sat_min = sat_delta.min(axis=0)
  sat_max = sat_delta.max(axis=0)

  # determine sat mut region
  sm_start = (sat_preds_si.shape[0] - satmut_len) // 2
  sm_end = sm_start + satmut_len

  # compute loss and gain matrixes
  sat_loss = sat_min - sat_preds_si[sm_start:sm_end, :]
  sat_gain = sat_max - sat_preds_si[sm_start:sm_end, :]

  return sat_loss, sat_gain


def plot_heat(ax, sat_delta_ti, min_limit):
  """ Plot satmut deltas.

    Args:
        ax (Axis): matplotlib axis to plot to.
        sat_delta_ti (4 x L_sm array): Single target delta matrix for saturated mutagenesis region,
        min_limit (float): Minimum heatmap limit.
    """
  vlim = max(min_limit, abs(sat_delta_ti).max())
  sns.heatmap(
      sat_delta_ti,
      linewidths=0,
      cmap='RdBu_r',
      vmin=-vlim,
      vmax=vlim,
      xticklabels=False,
      ax=ax)
  ax.yaxis.set_ticklabels('TGCA', rotation='horizontal')  # , size=10)


def plot_predictions(ax, preds, satmut_len, seq_len, buffer):
  """ Plot the raw predictions for a sequence and target
        across the specificed saturated mutagenesis region.

    Args:
        ax (Axis): matplotlib axis to plot to.
        preds (L array): Target predictions for one sequence.
        satmut_len (int): Satmut length from which to determine
                           the values to plot.
        seq_len (int): Full sequence length.
        buffer (int): Ignored buffer sequence on each side
    """

  # repeat preds across pool width
  target_pool = (seq_len - 2 * buffer) // preds.shape[0]
  epreds = preds.repeat(target_pool)

  satmut_start = (epreds.shape[0] - satmut_len) // 2
  satmut_end = satmut_start + satmut_len

  ax.plot(epreds[satmut_start:satmut_end], linewidth=1)
  ax.set_xlim(0, satmut_len)
  ax.axhline(0, c='black', linewidth=1, linestyle='--')
  for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(0.5)


def plot_sad(ax, sat_loss_ti, sat_gain_ti):
  """ Plot loss and gain SAD scores.

    Args:
        ax (Axis): matplotlib axis to plot to.
        sat_loss_ti (L_sm array): Minimum mutation delta across satmut length.
        sat_gain_ti (L_sm array): Maximum mutation delta across satmut length.
    """

  rdbu = sns.color_palette('RdBu_r', 10)

  ax.plot(-sat_loss_ti, c=rdbu[0], label='loss', linewidth=1)
  ax.plot(sat_gain_ti, c=rdbu[-1], label='gain', linewidth=1)
  ax.set_xlim(0, len(sat_loss_ti))
  ax.legend()
  # ax_sad.grid(True, linestyle=':')

  ax.xaxis.set_ticks([])
  for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(0.5)


def plot_seqlogo(ax, seq_1hot, sat_score_ti, pseudo_pct=0.05):
  """ Plot a sequence logo for the loss/gain scores.

    Args:
        ax (Axis): matplotlib axis to plot to.
        seq_1hot (Lx4 array): One-hot coding of a sequence.
        sat_score_ti (L_sm array): Minimum mutation delta across satmut length.
        pseudo_pct (float): % of the max to add as a pseudocount.
    """

  satmut_len = len(sat_score_ti)

  # add pseudocounts
  sat_score_ti += pseudo_pct * sat_score_ti.max()

  # expand
  sat_score_4l = expand_4l(sat_score_ti, seq_1hot)

  plots.seqlogo(sat_score_4l, ax)


'''
def plot_weblogo(ax, seq, sat_loss_ti, min_limit):
  """ Plot height-weighted weblogo sequence.

    Args:
        ax (Axis): matplotlib axis to plot to.
        seq ([ACGT]): DNA sequence
        sat_loss_ti (L_sm array): Minimum mutation delta across satmut length.
        min_limit (float): Minimum heatmap limit.
    """
  # trim sequence to the satmut region
  satmut_len = len(sat_loss_ti)
  satmut_start = int((len(seq) - satmut_len) // 2)
  satmut_seq = seq[satmut_start:satmut_start + satmut_len]

  # determine nt heights
  vlim = max(min_limit, np.max(-sat_loss_ti))
  seq_heights = 0.1 + 1.9 / vlim * (-sat_loss_ti)

  # make logo as eps
  eps_fd, eps_file = tempfile.mkstemp()
  seq_logo(satmut_seq, seq_heights, eps_file, color_mode='meme')

  # convert to png
  png_fd, png_file = tempfile.mkstemp()
  subprocess.call(
      'convert -density 1200 %s %s' % (eps_file, png_file), shell=True)

  # plot
  logo = Image.open(png_file)
  ax.imshow(logo)
  ax.set_axis_off()

  # clean up
  os.close(eps_fd)
  os.remove(eps_file)
  os.close(png_fd)
  os.remove(png_file)
'''

def satmut_seqs(seqs_1hot, satmut_len):
  """ Construct a new array with the given sequences and saturated
        mutagenesis versions of them. """

  seqs_n = seqs_1hot.shape[0]
  seq_len = seqs_1hot.shape[1]
  satmut_n = seqs_n + seqs_n * satmut_len * 3

  # initialize satmut seqs 1hot
  sat_seqs_1hot = np.zeros((satmut_n, seq_len, 4), dtype='bool')

  # copy over seqs_1hot
  sat_seqs_1hot[:seqs_n, :, :] = seqs_1hot

  satmut_start = (seq_len - satmut_len) // 2
  satmut_end = satmut_start + satmut_len

  # add saturated mutagenesis
  smi = seqs_n
  for si in range(seqs_n):
    for li in range(satmut_start, satmut_end):
      for ni in range(4):
        if seqs_1hot[si, li, ni] != 1:
          # copy sequence
          sat_seqs_1hot[smi, :, :] = seqs_1hot[si, :, :]

          # mutate to ni
          sat_seqs_1hot[smi, li, :] = np.zeros(4)
          sat_seqs_1hot[smi, li, ni] = 1

          # update index
          smi += 1

  return sat_seqs_1hot


def subplot_params(seq_len):
  """ Specify subplot layout parameters for various sequence lengths. """
  if seq_len < 500:
    spp = {
        'heat_cols': 400,
        'pred_start': 0,
        'pred_span': 322,
        'sad_start': 1,
        'sad_span': 321,
        'logo_start': 0,
        'logo_span': 323
    }
  else:
    spp = {
        'heat_cols': 400,
        'pred_start': 0,
        'pred_span': 321,
        'sad_start': 1,
        'sad_span': 320,
        'logo_start': 0,
        'logo_span': 322
    }

  return spp


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
