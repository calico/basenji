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

  def gradients(self,
                sess,
                batcher,
                target_indexes=None,
                layers=None,
                return_preds=False):
    """ Compute predictions on a test set.

        In
         sess: TensorFlow session
         batcher: Batcher class with sequence(s)
         target_indexes: Optional target subset list
         layers: Optional layer subset list

        Out
         grads: [S (sequences) x Li (layer i shape) x T (targets) array] * (L
         layers)
         preds:
        """

    # initialize target_indexes
    if target_indexes is None:
      target_indexes = np.array(range(self.num_targets))
    elif type(target_indexes) != np.ndarray:
      target_indexes = np.array(target_indexes)

    # initialize gradients
    #  (I need a list for layers because the sizes are different within)
    #  (I'm using a list for targets because I need to compute them individually)
    layer_grads = []
    for lii in range(len(layers)):
      layer_grads.append([])
      for tii in range(len(target_indexes)):
        layer_grads[lii].append([])

    # initialize layers
    if layers is None:
      layers = range(1 + self.cnn_layers)
    elif type(layers) != list:
      layers = [layers]

    # initialize predictions
    preds = None
    if return_preds:
      # determine non-buffer region
      buf_start = self.batch_buffer // self.target_pool
      buf_end = (self.seq_length - self.batch_buffer) // self.target_pool
      buf_len = buf_end - buf_start

      # initialize predictions
      preds = np.zeros(
          (batcher.num_seqs, buf_len, len(target_indexes)), dtype='float16')

      # sequence index
      si = 0

    # setup feed dict for dropout
    fd = self.set_mode('test')

    # get first batch
    Xb, _, _, Nb = batcher.next()

    while Xb is not None:
      # update feed dict
      fd[self.inputs] = Xb

      # predict
      preds_batch = sess.run(self.preds_op, feed_dict=fd)

      # compute gradients for each target individually
      for tii in range(len(target_indexes)):
        ti = target_indexes[tii]

        # compute gradients over all positions
        grads_op = tf.gradients(self.preds_op[:, :, ti],
                                [self.layer_reprs[li] for li in layers])
        grads_batch_raw = sess.run(grads_op, feed_dict=fd)

        for lii in range(len(layers)):
          # clean up
          grads_batch = grads_batch_raw[lii][:Nb].astype('float16')
          if grads_batch.shape[1] == 1:
            grads_batch = grads_batch.squeeze(axis=1)

          # save
          layer_grads[lii][tii].append(grads_batch)

      if return_preds:
        # filter for specific targets
        if target_indexes is not None:
          preds_batch = preds_batch[:, :, target_indexes]

        # accumulate predictions
        preds[si:si + Nb, :, :] = preds_batch[:Nb, :, :]

        # update sequence index
        si += Nb

      # next batch
      Xb, _, _, Nb = batcher.next()

    # reset training batcher
    batcher.reset()

    # stack into arrays
    for lii in range(len(layers)):
      for tii in range(len(target_indexes)):
        # stack sequences
        layer_grads[lii][tii] = np.vstack(layer_grads[lii][tii])

      # transpose targets to back
      layer_grads[lii] = np.array(layer_grads[lii])
      if layer_grads[lii].ndim == 4:
        # length dimension
        layer_grads[lii] = np.transpose(layer_grads[lii], [1, 2, 3, 0])
      else:
        # no length dimension
        layer_grads[lii] = np.transpose(layer_grads[lii], [1, 2, 0])

    if return_preds:
      return layer_grads, preds
    else:
      return layer_grads

  def gradients_pos(self,
                    sess,
                    batcher,
                    position_indexes,
                    target_indexes=None,
                    layers=None,
                    return_preds=False):
    """ Compute predictions on a test set.

        In
         sess: TensorFlow session
         batcher: Batcher class with sequence(s)
         position_indexes: Optional position subset list
         target_indexes: Optional target subset list
         layers: Optional layer subset list

        Out
         grads: [S (sequences) x Li (layer i shape) x T (targets) array] * (L
         layers)
         preds:
        """

    # initialize target_indexes
    if target_indexes is None:
      target_indexes = np.array(range(self.num_targets))
    elif type(target_indexes) != np.ndarray:
      target_indexes = np.array(target_indexes)

    # initialize layers
    if layers is None:
      layers = range(1 + self.cnn_layers)
    elif type(layers) != list:
      layers = [layers]

    # initialize gradients
    #  (I need a list for layers because the sizes are different within)
    #  (I'm using a list for positions/targets because I don't know the downstream object size)
    layer_grads = []
    for lii in range(len(layers)):
      layer_grads.append([])
      for pii in range(len(position_indexes)):
        layer_grads[lii].append([])
        for tii in range(len(target_indexes)):
          layer_grads[lii][pii].append([])

    # initialize layer reprs
    layer_reprs = []
    for lii in range(len(layers)):
      layer_reprs.append([])

    # initialize predictions
    preds = None
    if return_preds:
      # determine non-buffer region
      buf_start = self.batch_buffer // self.target_pool
      buf_end = (self.seq_length - self.batch_buffer) // self.target_pool
      buf_len = buf_end - buf_start

      # initialize predictions
      preds = np.zeros(
          (batcher.num_seqs, buf_len, len(target_indexes)), dtype='float16')

      # sequence index
      si = 0

    # setup feed dict for dropout
    fd = self.set_mode('test')

    # get first batch
    Xb, _, _, Nb = batcher.next()

    while Xb is not None:
      # update feed dict
      fd[self.inputs] = Xb

      # predict (allegedly takes zero time beyond the first sequence?)
      reprs_batch_raw, preds_batch = sess.run(
          [self.layer_reprs, self.preds_op], feed_dict=fd)

      # clean up layer repr
      reprs_batch = reprs_batch_raw[layers[lii]][:Nb].astype('float16')
      if reprs_batch.shape[1] == 1:
        reprs_batch = reprs_batch.squeeze(axis=1)

      # save repr
      layer_reprs[lii].append(reprs_batch)

      # for each target
      t0 = time.time()
      for tii in range(len(target_indexes)):
        ti = target_indexes[tii]

        # for each position
        for pii in range(len(position_indexes)):
          pi = position_indexes[pii]

          # adjust for buffer
          pi -= self.batch_buffer // self.target_pool

          # compute gradients
          grads_op = tf.gradients(self.preds_op[:, pi, ti],
                                  [self.layer_reprs[li] for li in layers])
          grads_batch_raw = sess.run(grads_op, feed_dict=fd)

          for lii in range(len(layers)):
            # clean up
            grads_batch = grads_batch_raw[lii][:Nb].astype('float16')
            if grads_batch.shape[1] == 1:
              grads_batch = grads_batch.squeeze(axis=1)

            # save
            layer_grads[lii][pii][tii].append(grads_batch)

      if return_preds:
        # filter for specific targets
        if target_indexes is not None:
          preds_batch = preds_batch[:, :, target_indexes]

        # accumulate predictions
        preds[si:si + Nb, :, :] = preds_batch[:Nb, :, :]

        # update sequence index
        si += Nb

      # next batch
      Xb, _, _, Nb = batcher.next()

    # reset training batcher
    batcher.reset()
    gc.collect()

    # stack into arrays
    for lii in range(len(layers)):
      layer_reprs[lii] = np.vstack(layer_reprs[lii])

      for pii in range(len(position_indexes)):
        for tii in range(len(target_indexes)):
          # stack sequences
          layer_grads[lii][pii][tii] = np.vstack(layer_grads[lii][pii][tii])

      # collapse position into arrays
      layer_grads[lii] = np.array(layer_grads[lii])

      # transpose positions and targets to back
      if layer_grads[lii].ndim == 5:
        # length dimension
        layer_grads[lii] = np.transpose(layer_grads[lii], [2, 3, 4, 0, 1])
      else:
        # no length dimension
        layer_grads[lii] = np.transpose(layer_grads[lii][2, 3, 0, 1])

    if return_preds:
      return layer_grads, layer_reprs, preds
    else:
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
                        return_all=False):

    # determine predictions length
    preds_length = self.preds_length
    if ds_indexes is not None:
      preds_length = len(ds_indexes)

    # determine num targets
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
          (Xb.shape[0], preds_length, num_targets, all_n), dtype='float16')
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
              down_sample=1):
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
        (batcher.num_seqs, preds_length, num_targets), dtype='float16')
    if return_var:
      if all_n == 1:
        print(
            'Cannot return prediction variance. Add rc, shifts, or mc.',
            file=sys.stderr)
        exit(1)
      preds_var = np.zeros(
          (batcher.num_seqs, preds_length, num_targets), dtype='float16')
    if return_all:
      preds_all = np.zeros(
          (batcher.num_seqs, preds_length, num_targets, all_n), dtype='float16')

    # sequence index
    si = 0

    # get first batch
    Xb, _, _, Nb = batcher.next()

    while Xb is not None:
      # make ensemble predictions
      preds_batch, preds_batch_var, preds_all = self._predict_ensemble(
          sess, fd, Xb, ensemble_fwdrc, ensemble_shifts, mc_n, ds_indexes,
          target_indexes, return_var, return_all)

      # accumulate predictions
      preds[si:si + Nb, :, :] = preds_batch[:Nb, :, :]
      if return_var:
        preds_var[si:si + Nb, :, :] = preds_batch_var[:Nb, :, :] / (all_n - 1)
      if return_all:
        preds_all[si:si + Nb, :, :, :] = preds_all[:Nb, :, :, :]

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
                    transcript_map,
                    rc=False,
                    shifts=[0],
                    mc_n=0,
                    target_indexes=None):
    """ Compute predictions on a test set.

        In
         sess:            TensorFlow session
         batcher:         Batcher class with transcript-covering sequences
         transcript_map:  OrderedDict mapping transcript id's to (sequence
         index, position) tuples marking TSSs.
         rc:              Average predictions from the forward and reverse
         complement sequences.
         shifts:          Average predictions from sequence shifts left/right.
         mc_n:            Monte Carlo iterations per rc/shift.
         target_indexes:  Optional target subset list

        Out
         transcript_preds: G (gene transcripts) X T (targets) array
        """

    # setup feed dict
    fd = self.set_mode('test')

    # initialize prediction arrays
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

    # initialize gene target predictions
    num_genes = len(transcript_map)
    gene_preds = np.zeros((num_genes, num_targets), dtype='float16')

    # construct an inverse map
    sequence_pos_transcripts = []
    txi = 0
    for transcript in transcript_map:
      si, pos = transcript_map[transcript]

      # extend sequence list
      while len(sequence_pos_transcripts) <= si:
        sequence_pos_transcripts.append({})

      # add gene to position set
      sequence_pos_transcripts[si].setdefault(pos, set()).add(txi)

      txi += 1

    # sequence index
    si = 0

    # get first batch
    Xb, _, _, Nb = batcher.next()

    while Xb is not None:
      # make ensemble predictions
      preds_batch, _, _ = self._predict_ensemble(
          sess,
          fd,
          Xb,
          ensemble_fwdrc,
          ensemble_shifts,
          mc_n,
          target_indexes=target_indexes)

      # for each sequence in the batch
      for pi in range(Nb):
        # for each position with a gene
        for tpos in sequence_pos_transcripts[si + pi]:
          # for each gene at that position
          for txi in sequence_pos_transcripts[si + pi][tpos]:
            # adjust for the buffer
            ppos = tpos - self.batch_buffer // self.target_pool

            # add prediction
            gene_preds[txi, :] += preds_batch[pi, ppos, :]

      # update sequence index
      si += Nb

      # next batch
      Xb, _, _, Nb = batcher.next()

    # reset batcher
    batcher.reset()

    return gene_preds

  def test_from_data_ops(self,
                         sess,
                         num_test_batches=0):
    """ Compute model accuracy on a test set, where data is loaded from a queue.

        Args:
          sess:         TensorFlow session
          num_test_batches: if > 0, only use this many test batches

        Returns:
          acc:          Accuracy object
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
    si = 0
    Nb = self.batch_size
    batch_count = 0
    while batch_count < num_test_batches:
      batch_count += 1
      # make non-ensembled predictions
      targets_batch, preds_batch, loss_batch, Yb, NAb = sess.run(
          [
              self.targets_op, self.preds_op, self.loss_op, self.targets,
              self.targets_na
          ],
          feed_dict=fd)
      target_losses_batch = loss_batch
      targets_na.append(np.zeros([Nb, self.preds_length], dtype='bool'))

      preds.append(preds_batch[:Nb, :, :].astype('float16'))
      targets.append(targets_batch[:Nb, :, :].astype('float16'))

      # accumulate loss
      batch_losses.append(loss_batch)
      batch_target_losses.append(target_losses_batch)

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
           num_test_batches=0):
    """ Compute model accuracy on a test set.

        Args:
          sess:         TensorFlow session
          batcher:      Batcher object to provide data
          rc:             Average predictions from the forward and reverse
            complement sequences.
          shifts:         Average predictions from sequence shifts left/right.
          mc_n:           Monte Carlo iterations per rc/shift.
          num_test_batches: if > 0, only use this many test batches

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
    while Xb is not None and (num_test_batches == 0 or
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
