"""
General utility code for test-time interraction with a SeqNN model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from basenji.dna_io import hot1_augment
from basenji import accuracy


class SeqNNModel(object):

  def build_grads(self, layers=[0]):
    ''' Build gradient ops for predictions summed across the sequence for
         each target with respect to some set of layers.
    In
      layers: Optional layer subset list
    '''

    self.grad_layers = layers
    self.grad_ops = []

    for ti in range(self.num_targets):
      grad_ti_op = tf.gradients(self.preds_op[:,:,ti], [self.layer_reprs[li] for li in self.grad_layers])
      self.grad_ops.append(grad_ti_op)


  def build_grads_genes(self, gene_seqs, layers=[0]):
    ''' Build gradient ops for TSS position-specific predictions
         for each target with respect to some set of layers.
    In
      gene_seqs:  GeneSeq list, from which to extract TSS positions
      layers:     Layer subset list.
    '''

    # save layer indexes
    self.grad_layers = layers

    # initialize ops
    self.grad_pos_ops = []

    # determine TSS positions
    tss_pos = set()
    for gene_seq in gene_seqs:
      for tss in gene_seq.tss_list:
        tss_pos.add(tss.seq_bin(width=self.target_pool, pred_buffer=self.batch_buffer))

    # for each position
    for pi in range(self.preds_length):
      self.grad_pos_ops.append([])

      # if it's a TSS position
      if pi in tss_pos:
        # build position-specific, target-specific gradient ops
        for ti in range(self.num_targets):
          grad_piti_op = tf.gradients(self.preds_op[:,pi,ti], [self.layer_reprs[li] for li in self.grad_layers])
          self.grad_pos_ops[-1].append(grad_piti_op)


  def gradients(self,
                sess,
                batcher,
                rc=False,
                shifts=[0],
                mc_n=0,
                return_all=False):
    """ Compute predictions on a test set.

        In
         sess: TensorFlow session
         batcher: Batcher class with sequence(s)
         rc: Average predictions from the forward and reverse complement sequences.
         shifts:
         mc_n:
         return_all: Return all ensemble predictions.

        Out
         layer_grads: [S (sequences) x T (targets) x P (seq position) x U (Units layer i) array] * (L layers)
         layer_reprs: [S (sequences) x P (seq position) x U (Units layer i) array] * (L layers)
         preds:
        """

    #######################################################################
    # determine ensemble iteration parameters

    ensemble_fwdrc = []
    ensemble_shifts = []
    for shift in shifts:
      ensemble_fwdrc.append(True)
      ensemble_shifts.append(shift)
      if rc:
        ensemble_fwdrc.append(False)
        ensemble_shifts.append(shift)

    if mc_n > 0:
      # setup feed dict
      fd = self.set_mode('test_mc')

    else:
      # setup feed dict
      fd = self.set_mode('test')

      # co-opt the variable to represent
      # iterations per fwdrc/shift.
      mc_n = 1

    # total ensemble predictions
    all_n = mc_n * len(ensemble_fwdrc)


    #######################################################################
    # initialize data structures

    # initialize gradients
    #  (I need a list for layers because the sizes are different within)
    #  (Targets up front, because I need to run their ops one by one)
    layer_reprs = []
    layer_grads = []
    layer_reprs_all = []
    layer_grads_all = []

    for lii in range(len(self.grad_layers)):
      li = self.grad_layers[lii]
      layer_seq_len = self.layer_reprs[li].shape[1].value
      layer_units = self.layer_reprs[li].shape[2].value

      lr = np.zeros((batcher.num_seqs, layer_seq_len, layer_units),
                    dtype='float32')
      layer_reprs.append(lr)

      lg = np.zeros((self.num_targets, batcher.num_seqs,
                      layer_seq_len, layer_units),
                      dtype='float32')
      layer_grads.append(lg)

      if return_all:
        lra = np.zeros((batcher.num_seqs, layer_seq_len, layer_units, all_n),
                        dtype='float32')
        layer_reprs_all.append(lra)

        lgr = np.zeros((self.num_targets, batcher.num_seqs,
                        layer_seq_len, layer_units, all_n),
                        dtype='float32')
        layer_grads_all.append(lgr)


    # initialize predictions
    preds = np.zeros((batcher.num_seqs, self.preds_length, self.num_targets),
                      dtype='float32')

    if return_all:
      preds_all = np.zeros((batcher.num_seqs, self.preds_length,
                            self.num_targets, all_n),
                            dtype='float32')

    #######################################################################
    # compute

    # sequence index
    si = 0

    # get first batch
    Xb, _, _, Nb = batcher.next()

    while Xb is not None:
      # ensemble predict
      preds_batch, layer_reprs_batch, layer_grads_batch = self._gradients_ensemble(
        sess, fd, Xb, ensemble_fwdrc, ensemble_shifts, mc_n, return_all=return_all)

      # unpack
      if return_all:
        preds_batch, preds_batch_all = preds_batch
        layer_reprs_batch, layer_reprs_batch_all = layer_reprs_batch
        layer_grads_batch, layer_grads_batch_all = layer_grads_batch

      # accumulate predictions
      preds[si:si+Nb,:,:] = preds_batch[:Nb,:,:]
      if return_all:
        preds_all[si:si+Nb,:,:,:] = preds_batch_all[:Nb,:,:,:]

      # accumulate representations
      for lii in range(len(self.grad_layers)):
        layer_reprs[lii][si:si+Nb] = layer_reprs_batch[lii][:Nb]
        if return_all:
          layer_reprs_all[lii][si:si+Nb] = layer_reprs_batch_all[lii][:Nb]

      # accumulate gradients
      for lii in range(len(self.grad_layers)):
        for ti in range(self.num_targets):
          layer_grads[lii][ti,si:si+Nb,:,:] = layer_grads_batch[lii][ti,:Nb,:,:]
          if return_all:
            layer_grads_all[lii][ti,si:si+Nb,:,:,:] = layer_grads_batch_all[lii][ti,:Nb,:,:,:]

      # update sequence index
      si += Nb

      # next batch
      Xb, _, _, Nb = batcher.next()

    # reset training batcher
    batcher.reset()


    #######################################################################
    # modify and return

    # move sequences to front
    for lii in range(len(self.grad_layers)):
      layer_grads[lii] = np.transpose(layer_grads[lii], [1,0,2,3])
      if return_all:
        layer_grads_all[lii] = np.transpose(layer_grads_all[lii], [1,0,2,3,4])

    if return_all:
      return layer_grads, layer_reprs, preds, layer_grads_all, layer_reprs_all, preds_all
    else:
      return layer_grads, layer_reprs, preds


  def _gradients_ensemble(self, sess, fd, Xb, ensemble_fwdrc, ensemble_shifts, mc_n, return_var=False, return_all=False):
    """ Compute gradients over an ensemble of input augmentations.

      In
       sess: TensorFlow session
       fd: feed dict
       Xb: input data
       ensemble_fwdrc:
       ensemble_shifts:
       mc_n:
       return_var:
       return_all: Return all ensemble predictions.

      Out
       preds:
       layer_reprs:
       layer_grads
    """

    # initialize batch predictions
    preds = np.zeros((Xb.shape[0], self.preds_length, self.num_targets), dtype='float32')

    # initialize layer representations and gradients
    layer_reprs = []
    layer_grads = []
    for lii in range(len(self.grad_layers)):
      li = self.grad_layers[lii]
      layer_seq_len = self.layer_reprs[li].shape[1].value
      layer_units = self.layer_reprs[li].shape[2].value

      lr = np.zeros((Xb.shape[0], layer_seq_len, layer_units), dtype='float16')
      layer_reprs.append(lr)

      lg = np.zeros((self.num_targets, Xb.shape[0], layer_seq_len, layer_units), dtype='float32')
      layer_grads.append(lg)


    # initialize variance
    if return_var:
      preds_var = np.zeros(preds.shape, dtype='float32')

      layer_reprs_var = []
      layer_grads_var = []
      for lii in range(len(self.grad_layers)):
        layer_reprs_var.append(np.zeros(layer_reprs.shape, dtype='float32'))
        layer_grads_var.append(np.zeros(layer_grads.shape, dtype='float32'))
    else:
      preds_var = None
      layer_grads_var = [None]*len(self.grad_layers)


    # initialize all-saving arrays
    if return_all:
      all_n = mc_n * len(ensemble_fwdrc)
      preds_all = np.zeros((Xb.shape[0], self.preds_length, self.num_targets, all_n), dtype='float32')

      layer_reprs_all = []
      layer_grads_all = []
      for lii in range(len(self.grad_layers)):
        ls = tuple(list(layer_reprs[lii].shape) + [all_n])
        layer_reprs_all.append(np.zeros(ls, dtype='float32'))

        ls = tuple(list(layer_grads[lii].shape) + [all_n])
        layer_grads_all.append(np.zeros(ls, dtype='float32'))
    else:
      preds_all = None
      layer_grads_all = [None]*len(self.grad_layers)


    running_i = 0

    for ei in range(len(ensemble_fwdrc)):
      # construct sequence
      Xb_ensemble = hot1_augment(Xb, ensemble_fwdrc[ei], ensemble_shifts[ei])

      # update feed dict
      fd[self.inputs] = Xb_ensemble

      # for each monte carlo (or non-mc single) iteration
      for mi in range(mc_n):
        # print('ei=%d, mi=%d, fwdrc=%d, shifts=%d' % \
        #       (ei, mi, ensemble_fwdrc[ei], ensemble_shifts[ei]),
        #       flush=True)

        ##################################################
        # prediction

        # predict
        preds_ei, layer_reprs_ei = sess.run([self.preds_op, self.layer_reprs], feed_dict=fd)

        # reverse
        if ensemble_fwdrc[ei] is False:
          preds_ei = preds_ei[:,::-1,:]

        # save previous mean
        preds1 = preds

        # update mean
        preds = self.running_mean(preds1, preds_ei, running_i+1)

        # update variance sum
        if return_var:
          preds_var = self.running_varsum(preds_var, preds_ei, preds1, preds)

        # save iteration
        if return_all:
          preds_all[:,:,:,running_i] = preds_ei[:,:,:]


        ##################################################
        # representations

        for lii in range(len(self.grad_layers)):
          li = self.grad_layers[lii]

          # reverse
          if ensemble_fwdrc[ei] is False:
            layer_reprs_ei[li] = layer_reprs_ei[li][:,::-1,:]

          # save previous mean
          layer_reprs_lii1 = layer_reprs[lii]

          # update mean
          layer_reprs[lii] = self.running_mean(layer_reprs_lii1, layer_reprs_ei[li], running_i+1)

          # update variance sum
          if return_var:
            layer_reprs_var[lii] = self.running_varsum(layer_reprs_var[lii], layer_reprs_ei[li],
                                                  layer_reprs_lii1, layer_reprs[lii])

          # save iteration
          if return_all:
            layer_reprs_all[lii][:,:,:,running_i] = layer_reprs_ei[li]


        ##################################################
        # gradients

        # compute gradients for each target individually
        for ti in range(self.num_targets):
          # compute gradients
          layer_grads_ti_ei = sess.run(self.grad_ops[ti], feed_dict=fd)

          for lii in range(len(self.grad_layers)):
            # reverse
            if ensemble_fwdrc[ei] is False:
              layer_grads_ti_ei[lii] = layer_grads_ti_ei[lii][:,::-1,:]

            # save predious mean
            layer_grads_lii_ti1 = layer_grads[lii][ti]

            # update mean
            layer_grads[lii][ti] = self.running_mean(layer_grads_lii_ti1, layer_grads_ti_ei[lii], running_i+1)

            # update variance sum
            if return_var:
              layer_grads_var[lii][ti] = self.running_varsum(layer_grads_var[lii][ti], layer_grads_ti_ei[lii],
                                                          layer_grads_lii_ti1, layer_grads[lii][ti])

            # save iteration
            if return_all:
              layer_grads_all[lii][ti,:,:,:,running_i] = layer_grads_ti_ei[lii]

        # update running index
        running_i += 1

    if return_var:
      return (preds, preds_var), (layer_reprs, layer_reprs_var), (layer_grads, layer_grads_var)
    elif return_all:
      return (preds, preds_all), (layer_reprs, layer_reprs_all), (layer_grads, layer_grads_all)
    else:
      return preds, layer_reprs, layer_grads

  def gradients_genes(self, sess, batcher, gene_seqs, rc=False):
    ''' Compute predictions on a test set.
    In
     sess:       TensorFlow session
     batcher:    Batcher class with sequence(s)
     gene_seqs:  List of GeneSeq instances specifying gene positions in sequences.
     rc:         Average predictions from the forward and reverse complement sequences.
    Out
     layer_grads: [G (TSSs) x T (targets) x P (seq position) x U (Units layer i) array] * (L layers)
     layer_reprs: [S (sequences) x P (seq position) x U (Units layer i) array] * (L layers)
    Notes
     -Reverse complements aren't implemented yet. They're trickier here, because
      I'd need to build more gradient ops to match the flipped positions.
    '''

    # count TSSs
    tss_num = 0
    for gene_seq in gene_seqs:
      tss_num += len(gene_seq.tss_list)

    # initialize gradients and representations
    #  (I need a list for layers because the sizes are different within)
    #  (TSSxTargets up front, because I need to run their ops one by one)
    layer_grads = []
    layer_reprs = []
    for lii in range(len(self.grad_layers)):
      li = self.grad_layers[lii]
      layer_seq_len = self.layer_reprs[li].shape[1].value
      layer_units = self.layer_reprs[li].shape[2].value

      # gradients
      lg = np.zeros((tss_num, self.num_targets, layer_seq_len, layer_units), dtype='float32')
      layer_grads.append(lg)

      # representations
      lr = np.zeros((batcher.num_seqs, layer_seq_len, layer_units), dtype='float32')
      layer_reprs.append(lr)

    # setup feed dict for dropout
    fd = self.set_mode('test')

    # TSS index
    tss_i = 0

    # sequence index
    si = 0

    # get first batch
    Xb, _, _, Nb = batcher.next()

    while Xb is not None:
      # update feed dict
      fd[self.inputs] = Xb

      # predict
      reprs_batch, _ = sess.run([self.layer_reprs, self.preds_op], feed_dict=fd)

      # save representations
      for lii in range(len(self.grad_layers)):
        li = self.grad_layers[lii]
        layer_reprs[lii][si:si+Nb] = reprs_batch[li][:Nb]

      # compute gradients for each TSS position individually
      for bi in range(Nb):
        for tss in gene_seqs[si+bi].tss_list:
          # get TSS prediction bin position
          pi = tss.seq_bin(width=self.target_pool, pred_buffer=self.batch_buffer)

          for ti in range(self.num_targets):
            # compute gradients over all positions
            grads_batch = sess.run(self.grad_pos_ops[pi][ti], feed_dict=fd)

            # accumulate gradients
            for lii in range(len(self.grad_layers)):
              layer_grads[lii][tss_i,ti,:,:] = grads_batch[lii][bi]

          # update TSS index
          tss_i += 1

      # update sequence index
      si += Nb

      # next batch
      Xb, _, _, Nb = batcher.next()

    # reset training batcher
    batcher.reset()

    return layer_grads, layer_reprs


  def hidden(self, sess, batcher, layers=None):
    """ Compute hidden representations for a test set. """

    if layers is None:
      layers = list(range(self.cnn_layers))

    # initialize layer representation data structure
    layer_reprs = []
    for li in range(1 + np.max(layers)):
      layer_reprs.append([])
    preds = []

    # setup feed dict
    fd = self.set_mode('test')

    # get first batch
    Xb, _, _, Nb = batcher.next()

    while Xb is not None:
      # update feed dict
      fd[self.inputs] = Xb

      # compute predictions
      layer_reprs_batch, preds_batch = sess.run(
          [self.layer_reprs, self.preds_op], feed_dict=fd)

      # accumulate representationsmakes the number of members for self smaller and also
      for li in layers:
        # squeeze (conv_2d-expanded) second dimension
        if layer_reprs_batch[li].shape[1] == 1:
          layer_reprs_batch[li] = layer_reprs_batch[li].squeeze(axis=1)

        # append
        layer_reprs[li].append(layer_reprs_batch[li][:Nb].astype('float16'))

      # accumualte predictions
      preds.append(preds_batch[:Nb])

      # next batch
      Xb, _, _, Nb = batcher.next()

    # reset batcher
    batcher.reset()

    # accumulate representations
    for li in layers:
      layer_reprs[li] = np.vstack(layer_reprs[li])

    preds = np.vstack(preds)

    return layer_reprs, preds


  def _predict_ensemble(self,
                        sess,
                        fd,
                        Xb,
                        ensemble_fwdrc,
                        ensemble_shifts,
                        mc_n,
                        ds_indexes=None,
                        target_indexes=None,
                        return_var=False,
                        return_all=False,
                        penultimate=False):

    # determine predictions length
    preds_length = self.preds_length
    if ds_indexes is not None:
      preds_length = len(ds_indexes)

    # determine num targets
    if penultimate:
      num_targets = self.cnn_filters[-1]
    else:
      num_targets = self.num_targets
      if target_indexes is not None:
        num_targets = len(target_indexes)

    # initialize batch predictions
    preds_batch = np.zeros(
        (Xb.shape[0], preds_length, num_targets), dtype='float32')

    if return_var:
      preds_batch_var = np.zeros(preds_batch.shape, dtype='float32')
    else:
      preds_batch_var = None

    if return_all:
      all_n = mc_n * len(ensemble_fwdrc)
      preds_all = np.zeros(
          (Xb.shape[0], preds_length, num_targets, all_n), dtype='float32')
    else:
      preds_all = None

    running_i = 0

    for ei in range(len(ensemble_fwdrc)):
      # construct sequence
      Xb_ensemble = hot1_augment(Xb, ensemble_fwdrc[ei], ensemble_shifts[ei])

      # update feed dict
      fd[self.inputs] = Xb_ensemble

      # for each monte carlo (or non-mc single) iteration
      for mi in range(mc_n):
        # print('ei=%d, mi=%d, fwdrc=%d, shifts=%d' % (ei, mi, ensemble_fwdrc[ei], ensemble_shifts[ei]), flush=True)

        # predict
        if penultimate:
          preds_ei = sess.run(self.penultimate_op, feed_dict=fd)
        else:
          preds_ei = sess.run(self.preds_op, feed_dict=fd)

        # reverse
        if ensemble_fwdrc[ei] is False:
          preds_ei = preds_ei[:, ::-1, :]

        # down-sample
        if ds_indexes is not None:
          preds_ei = preds_ei[:, ds_indexes, :]
        if target_indexes is not None:
          preds_ei = preds_ei[:, :, target_indexes]

        # save previous mean
        preds_batch1 = preds_batch

        # update mean
        preds_batch = self.running_mean(preds_batch1, preds_ei, running_i + 1)

        # update variance sum
        if return_var:
          preds_batch_var = self.running_varsum(preds_batch_var, preds_ei,
                                                preds_batch1, preds_batch)

        # save iteration
        if return_all:
          preds_all[:, :, :, running_i] = preds_ei[:, :, :]

        # update running index
        running_i += 1

    return preds_batch, preds_batch_var, preds_all

  def predict(self,
              sess,
              batcher,
              rc=False,
              shifts=[0],
              mc_n=0,
              target_indexes=None,
              return_var=False,
              return_all=False,
              down_sample=1,
              penultimate=False,
              dtype='float32'):
    """ Compute predictions on a test set.

        In
         sess:           TensorFlow session
         batcher:        Batcher class with transcript-covering sequences.
         rc:             Average predictions from the forward and reverse
         complement sequences.
         shifts:         Average predictions from sequence shifts left/right.
         mc_n:           Monte Carlo iterations per rc/shift.
         target_indexes: Optional target subset list
         return_var:     Return variance estimates
         down_sample:    Int specifying to consider uniformly spaced sampled
         positions
         penultimate:    Predict the penultimate layer.
         dtype:          Float resolution to return.

        Out
         preds: S (sequences) x L (unbuffered length) x T (targets) array
        """

    # uniformly sample indexes
    ds_indexes = None
    preds_length = self.preds_length
    if down_sample != 1:
      ds_indexes = np.arange(0, self.preds_length, down_sample)
      preds_length = len(ds_indexes)

    # initialize prediction arrays
    if penultimate:
      num_targets = self.cnn_filters[-1]
    else:
      num_targets = self.num_targets
      if target_indexes is not None:
        num_targets = len(target_indexes)

    # determine ensemble iteration parameters
    ensemble_fwdrc = []
    ensemble_shifts = []
    for shift in shifts:
      ensemble_fwdrc.append(True)
      ensemble_shifts.append(shift)
      if rc:
        ensemble_fwdrc.append(False)
        ensemble_shifts.append(shift)

    if mc_n > 0:
      # setup feed dict
      fd = self.set_mode('test_mc')

    else:
      # setup feed dict
      fd = self.set_mode('test')

      # co-opt the variable to represent
      # iterations per fwdrc/shift.
      mc_n = 1

    # total ensemble predictions
    all_n = mc_n * len(ensemble_fwdrc)

    # initialize prediction data structures
    preds = np.zeros(
        (batcher.num_seqs, preds_length, num_targets), dtype=dtype)
    if return_var:
      if all_n == 1:
        print(
            'Cannot return prediction variance. Add rc, shifts, or mc.',
            file=sys.stderr)
        exit(1)
      preds_var = np.zeros(
          (batcher.num_seqs, preds_length, num_targets), dtype=dtype)
    if return_all:
      preds_all = np.zeros(
          (batcher.num_seqs, preds_length, num_targets, all_n), dtype=dtype)

    # sequence index
    si = 0

    # get first batch
    Xb, _, _, Nb = batcher.next()

    while Xb is not None:
      # make ensemble predictions
      preds_batch, preds_batch_var, preds_batch_all = self._predict_ensemble(
          sess, fd, Xb, ensemble_fwdrc, ensemble_shifts, mc_n, ds_indexes,
          target_indexes, return_var, return_all, penultimate)

      # accumulate predictions
      preds[si:si + Nb, :, :] = preds_batch[:Nb, :, :]
      if return_var:
        preds_var[si:si + Nb, :, :] = preds_batch_var[:Nb, :, :] / (all_n - 1)
      if return_all:
        preds_all[si:si + Nb, :, :, :] = preds_batch_all[:Nb, :, :, :]

      # update sequence index
      si += Nb

      # next batch
      Xb, _, _, Nb = batcher.next()

    # reset batcher
    batcher.reset()

    if return_var:
      if return_all:
        return preds, preds_var, preds_all
      else:
        return preds, preds_var
    else:
      return preds

  def predict_genes(self,
                    sess,
                    batcher,
                    gene_seqs,
                    rc=False,
                    shifts=[0],
                    mc_n=0,
                    target_indexes=None,
                    tss_radius=0,
                    penultimate=False,
                    dtype='float32'):
    """ Compute predictions on a test set.

        In
         sess:            TensorFlow session
         batcher:         Batcher class with transcript-covering sequences
         gene_seqs        List of GeneSeq instances specifying gene positions in sequences.
         index, position) tuples marking TSSs.
         rc:              Average predictions from the forward and reverse
         complement sequences.
         shifts:          Average predictions from sequence shifts left/right.
         mc_n:            Monte Carlo iterations per rc/shift.
         target_indexes:  Optional target subset list
         tss_radius:      Radius of bins to quantify TSS.
         penultimate:     Predict the penultimate layer.
         dtype:           Float resolution to return.

        Out
         transcript_preds: G (gene transcripts) X T (targets) array
        """

    # predict gene sequences
    gseq_preds = self.predict(sess, batcher, rc=rc, shifts=shifts, mc_n=mc_n,
                              target_indexes=target_indexes, penultimate=penultimate)

    # count TSSs
    tss_num = 0
    for gene_seq in gene_seqs:
      tss_num += len(gene_seq.tss_list)

    # initialize TSS preds
    tss_preds = np.zeros( (tss_num, gseq_preds.shape[-1]), dtype=dtype)

    # slice TSSs
    tss_i = 0
    for si in range(len(gene_seqs)):
      for tss in gene_seqs[si].tss_list:
        bi = tss.seq_bin(width=self.target_pool, pred_buffer=self.batch_buffer)
        tss_preds[tss_i,:] = gseq_preds[si,bi-tss_radius:bi+1+tss_radius,:].sum(axis=0)
        tss_i += 1

    # reset batcher
    batcher.reset()

    return tss_preds


  def test_from_data_ops(self, sess, num_test_batches=None):
    """ Compute model accuracy on a test set, where data is loaded from a queue.

        Args:
          sess:             TensorFlow session
          num_test_batches: Number of test batches to use.

        Returns:
          acc:              Accuracy object
        """

    # TODO(dbelanger) this ignores rc and shift ensembling for now.
    # Accuracy will be slightly lower than if we had used this.
    # The rc and shift data augmentation need to be pulled into the graph.

    fd = self.set_mode('test')

    # initialize prediction and target arrays
    preds = []
    targets = []
    targets_na = []

    batch_losses = []
    batch_target_losses = []

    # sequence index
    data_available = True
    batch_count = 0
    while data_available and (num_test_batches is None or
                              batch_count < num_test_batches):
      batch_count += 1
      try:
        # make non-ensembled predictions
        run_ops = [self.targets_op, self.preds_op, self.loss_op,
                   self.target_losses, self.targets, self.targets_na]
        run_returns = sess.run(run_ops, feed_dict=fd)
        targets_batch, preds_batch, loss_batch, target_losses_batch, Yb, NAb = run_returns

        # accumulate predictions and targets
        preds.append(preds_batch.astype('float16'))
        targets.append(targets_batch.astype('float16'))
        targets_na.append(np.zeros([self.batch_size, self.preds_length], dtype='bool'))

        # accumulate loss
        batch_losses.append(loss_batch)
        batch_target_losses.append(target_losses_batch)

      except tf.errors.OutOfRangeError:
        data_available = False

    # construct arrays
    targets = np.concatenate(targets, axis=0)
    preds = np.concatenate(preds, axis=0)
    targets_na = np.concatenate(targets_na, axis=0)

    # mean across batches
    batch_losses = np.mean(batch_losses)
    batch_target_losses = np.array(batch_target_losses).mean(axis=0)

    # instantiate accuracy object
    acc = accuracy.Accuracy(targets, preds, targets_na, batch_losses,
                            batch_target_losses)

    return acc

  def test(self,
           sess,
           batcher,
           rc=False,
           shifts=[0],
           mc_n=0,
           num_test_batches=None):
    """ Compute model accuracy on a test set.

        Args:
          sess:         TensorFlow session
          batcher:      Batcher object to provide data
          rc:             Average predictions from the forward and reverse
            complement sequences.
          shifts:         Average predictions from sequence shifts left/right.
          mc_n:           Monte Carlo iterations per rc/shift.
          num_test_batches: Number of test batches to use.

        Returns:
          acc:          Accuracy object
        """

    # determine ensemble iteration parameters
    ensemble_fwdrc = []
    ensemble_shifts = []
    for shift in shifts:
      ensemble_fwdrc.append(True)
      ensemble_shifts.append(shift)
      if rc:
        ensemble_fwdrc.append(False)
        ensemble_shifts.append(shift)

    if mc_n > 0:
      # setup feed dict
      fd = self.set_mode('test_mc')

    else:
      # setup feed dict
      fd = self.set_mode('test')

      # co-opt the variable to represent
      # iterations per fwdrc/shift.
      mc_n = 1

    # initialize prediction and target arrays
    preds = []
    targets = []
    targets_na = []

    batch_losses = []
    batch_target_losses = []

    # sequence index
    si = 0

    # get first batch
    Xb, Yb, NAb, Nb = batcher.next()

    batch_count = 0
    while Xb is not None and (num_test_batches is None or
                              batch_count < num_test_batches):
      batch_count += 1

      # make ensemble predictions
      preds_batch, preds_batch_var, preds_all = self._predict_ensemble(
          sess, fd, Xb, ensemble_fwdrc, ensemble_shifts, mc_n)

      # add target info
      fd[self.targets] = Yb
      fd[self.targets_na] = NAb

      targets_na.append(np.zeros([Nb, self.preds_length], dtype='bool'))

      # recompute loss w/ ensembled prediction
      fd[self.preds_adhoc] = preds_batch
      targets_batch, loss_batch, target_losses_batch = sess.run(
          [self.targets_op, self.loss_adhoc, self.target_losses_adhoc],
          feed_dict=fd)

      # accumulate predictions and targets
      if preds_batch.ndim == 3:
        preds.append(preds_batch[:Nb, :, :].astype('float16'))
        targets.append(targets_batch[:Nb, :, :].astype('float16'))

      else:
        for qi in range(preds_batch.shape[3]):
          # TEMP, ideally this will be in the HDF5 and set previously
          self.quantile_means = np.geomspace(0.1, 256, 16)

          # softmax
          preds_batch_norm = np.expand_dims(
              np.sum(np.exp(preds_batch[:Nb, :, :, :]), axis=3), axis=3)
          pred_probs_batch = np.exp(
              preds_batch[:Nb, :, :, :]) / preds_batch_norm

          # expectation over quantile medians
          preds.append(np.dot(pred_probs_batch, self.quantile_means))

          # compare to quantile median
          targets.append(self.quantile_means[targets_batch[:Nb, :, :] - 1])

      # accumulate loss
      batch_losses.append(loss_batch)
      batch_target_losses.append(target_losses_batch)

      # update sequence index
      si += Nb

      # next batch
      Xb, Yb, NAb, Nb = batcher.next()

    targets = np.concatenate(targets, axis=0)
    preds = np.concatenate(preds, axis=0)
    targets_na = np.concatenate(targets_na, axis=0)

    # reset batcher
    batcher.reset()

    # mean across batches
    batch_losses = np.mean(batch_losses)
    batch_target_losses = np.array(batch_target_losses).mean(axis=0)

    # instantiate accuracy object
    acc = accuracy.Accuracy(targets, preds, targets_na, batch_losses,
                            batch_target_losses)

    return acc

  def running_mean(self, u_k1, x_k, k):
    return u_k1 + (x_k - u_k1) / k

  def running_varsum(self, v_k1, x_k, m_k1, m_k):
    """ Computing the running variance numerator.

      Ref: https://www.johndcook.com/blog/standard_deviation/
      """
    return v_k1 + (x_k - m_k1) * (x_k - m_k)
