# Copyright 2019 Calico LLC
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

import pdb
import sys
import time

from natsort import natsorted
import numpy as np
import tensorflow as tf

from basenji import blocks
from basenji import layers
from basenji import metrics

class SeqNN():

  def __init__(self, params):
    self.set_defaults()
    for key, value in params.items():
      self.__setattr__(key, value)
    self.build_model()
    self.ensemble = None

  def set_defaults(self):
    # only necessary for my bespoke parameters
    # others are best defaulted closer to the source
    self.augment_rc = False
    self.augment_shift = [0]
    self.strand_pair = []
    self.verbose = True

  def build_block(self, current, block_params):
    """Construct a SeqNN block.
    Args:
    Returns:
      current
    """
    block_args = {}

    # extract name
    block_name = block_params['name']

    # save upper_tri flatten
    self.preds_triu |= (block_name == 'upper_tri')
        
    # if Keras, get block variables names
    pass_all_globals = True
    if block_name[0].isupper():
      pass_all_globals = False
      block_func = blocks.keras_func[block_name]
      block_varnames = block_func.__init__.__code__.co_varnames

    # set global defaults
    global_vars = ['activation', 'batch_norm', 'bn_momentum', 'norm_type',
      'l2_scale', 'l1_scale', 'padding', 'kernel_initializer']
    for gv in global_vars:
      gv_value = getattr(self, gv, False)
      if gv_value and (pass_all_globals or gv in block_varnames):
        block_args[gv] = gv_value

    # set remaining params
    block_args.update(block_params)
    del block_args['name']

    # save representations
    if block_name.find('tower') != -1:
      block_args['reprs'] = self.reprs

    # U-net helper
    if block_name[-5:] == '_unet':
      # find matching representation
      unet_repr = None
      for seq_repr in reversed(self.reprs[:-1]):
        if seq_repr.shape[1] == current.shape[1]*2:
          unet_repr = seq_repr
          break
      if unet_repr is None:
        print('Could not find matching representation for length %d' % current.shape[1], sys.stderr)
        exit(1)
      block_args['unet_repr'] = unet_repr

    # switch for block
    if block_name[0].islower():
      block_func = blocks.name_func[block_name]
      current = block_func(current, **block_args)

    else:
      block_func = blocks.keras_func[block_name]
      current = block_func(**block_args)(current)

    return current

  def build_model(self, save_reprs=True):
    ###################################################
    # inputs
    ###################################################
    sequence = tf.keras.Input(shape=(self.seq_length, 4), name='sequence')
    current = sequence

    # augmentation
    if self.augment_rc:
      current, reverse_bool = layers.StochasticReverseComplement()(current)
    if self.augment_shift != [0]:
      current = layers.StochasticShift(self.augment_shift)(current)
    self.preds_triu = False
    
    ###################################################
    # build convolution blocks
    ###################################################
    self.reprs = []
    for bi, block_params in enumerate(self.trunk):
      current = self.build_block(current, block_params)
      if save_reprs:
        self.reprs.append(current)

    # final activation
    current = layers.activate(current, self.activation)

    # make model trunk
    trunk_output = current
    self.model_trunk = tf.keras.Model(inputs=sequence, outputs=trunk_output)

    ###################################################
    # heads
    ###################################################
    head_keys = natsorted([v for v in vars(self) if v.startswith('head')])
    self.heads = [getattr(self, hk) for hk in head_keys]

    self.head_output = []
    for hi, head in enumerate(self.heads):
      if not isinstance(head, list):
        head = [head]

      # reset to trunk output
      current = trunk_output

      # build blocks
      for bi, block_params in enumerate(head):
        current = self.build_block(current, block_params)

      if hi < len(self.strand_pair):
        strand_pair = self.strand_pair[hi]
      else:
        strand_pair = None

      # transform back from reverse complement
      if self.augment_rc:
        if self.preds_triu:
          current = layers.SwitchReverseTriu(self.diagonal_offset)([current, reverse_bool])
        else:
          current = layers.SwitchReverse(strand_pair)([current, reverse_bool])

      # save head output
      self.head_output.append(current)

    ###################################################
    # compile model(s)
    ###################################################
    self.models = []
    for ho in self.head_output:
      self.models.append(tf.keras.Model(inputs=sequence, outputs=ho))
    self.model = self.models[0]
    if self.verbose: print(self.model.summary())

    ###################################################
    # track pooling/striding and cropping
    ###################################################
    self.model_strides = []
    self.target_lengths = []
    self.target_crops = []
    for model in self.models:
      # determine model stride
      self.model_strides.append(1)
      for layer in self.model.layers:
        if hasattr(layer, 'strides') or hasattr(layer, 'size'):
          stride_factor = layer.input_shape[1] / layer.output_shape[1]
          self.model_strides[-1] *= stride_factor
      self.model_strides[-1] = int(self.model_strides[-1])

      # determine predictions length before cropping
      if type(sequence.shape[1]) == tf.compat.v1.Dimension:
        target_full_length = sequence.shape[1].value // self.model_strides[-1]
      else:
        target_full_length = sequence.shape[1] // self.model_strides[-1]

      # determine predictions length after cropping
      self.target_lengths.append(model.outputs[0].shape[1])
      if type(self.target_lengths[-1]) == tf.compat.v1.Dimension:
        self.target_lengths[-1] = self.target_lengths[-1].value
      self.target_crops.append((target_full_length - self.target_lengths[-1])//2)
    
    if self.verbose:
      print('model_strides', self.model_strides)
      print('target_lengths', self.target_lengths)
      print('target_crops', self.target_crops)


  def build_embed(self, conv_layer_i, batch_norm=True):
    if conv_layer_i == -1:
      self.model = self.model_trunk

    else:
      if batch_norm:
        conv_layer = self.get_bn_layer(conv_layer_i)
      else:
        conv_layer = self.get_conv_layer(conv_layer_i)

      self.model = tf.keras.Model(inputs=self.model.inputs,
                                  outputs=conv_layer.output)


  def build_ensemble(self, ensemble_rc=False, ensemble_shifts=[0]):
    """ Build ensemble of models computing on augmented input sequences. """
    if ensemble_rc or len(ensemble_shifts) > 1:
      # sequence input
      sequence = tf.keras.Input(shape=(self.seq_length, 4), name='sequence')
      sequences = [sequence]

      if len(ensemble_shifts) > 1:
        # generate shifted sequences
        sequences = layers.EnsembleShift(ensemble_shifts)(sequences)

      if ensemble_rc:
        # generate reverse complements and indicators
        sequences_rev = layers.EnsembleReverseComplement()(sequences)
      else:
        sequences_rev = [(seq,tf.constant(False)) for seq in sequences]

      if len(self.strand_pair) == 0:
        strand_pair = None
      else:
        strand_pair = self.strand_pair[0]

      # predict each sequence
      if self.preds_triu:
        preds = [layers.SwitchReverseTriu(self.diagonal_offset)
                  ([self.model(seq), rp]) for (seq,rp) in sequences_rev]
      else:
        preds = [layers.SwitchReverse(strand_pair)([self.model(seq), rp]) for (seq,rp) in sequences_rev]

      # create layer
      preds_avg = tf.keras.layers.Average()(preds)

      # create meta model
      self.ensemble = tf.keras.Model(inputs=sequence, outputs=preds_avg)


  def build_sad(self):
    # sequence input
    sequence = tf.keras.Input(shape=(self.seq_length, 4), name='sequence')

    # predict
    predictions = self.model(sequence)
    preds_len = predictions.shape[1]

    # sum pool
    sad = preds_len * tf.keras.layers.GlobalAveragePooling1D()(predictions)

    # replace model
    self.model = tf.keras.Model(inputs=sequence, outputs=sad)


  def build_slice(self, target_slice=None, target_sum=False):
    if target_slice is not None or target_sum:
      # sequence input
      sequence = tf.keras.Input(shape=(self.seq_length, 4), name='sequence')

      # predict
      predictions = self.model(sequence)

      # slice
      if target_slice is None:
        predictions_slice = predictions
      else:
        predictions_slice = tf.gather(predictions, target_slice, axis=-1)

      # sum
      if target_sum:
        predictions_sum = tf.reduce_sum(predictions_slice, keepdims=True, axis=-1)
      else:
        predictions_sum = predictions_slice

      # replace model
      self.model = tf.keras.Model(inputs=sequence, outputs=predictions_sum)


  def downcast(self, dtype=tf.float16, head_i=None):
    """ Downcast model output type. """
    # choose model
    if self.ensemble is not None:
      model = self.ensemble
    elif head_i is not None:
      model = self.models[head_i]
    else:
      model = self.model

    # sequence input
    sequence = tf.keras.Input(shape=(self.seq_length, 4), name='sequence')

    # predict and downcast
    preds = model(sequence)
    preds = tf.cast(preds, dtype)
    model_down = tf.keras.Model(inputs=sequence, outputs=preds)

    # replace model
    if self.ensemble is not None:
      self.ensemble = model_down
    elif head_i is not None:
      self.models[head_i] = model_down
    else:
      self.model = model_down


  def evaluate(self, seq_data, head_i=None, loss_label='poisson', loss_fn=None):
    """ Evaluate model on SeqDataset. """
    # choose model
    if self.ensemble is not None:
      model = self.ensemble
    elif head_i is not None:
      model = self.models[head_i]
    else:
      model = self.model

    # compile with dense metrics
    num_targets = model.output_shape[-1]

    if loss_fn is None:
      loss_fn = loss_label

    if loss_label == 'bce':
      model.compile(optimizer=tf.keras.optimizers.SGD(),
                    loss=loss_fn,
                    metrics=[metrics.SeqAUC(curve='ROC', summarize=False),
                             metrics.SeqAUC(curve='PR', summarize=False)])
    else:      
      model.compile(optimizer=tf.keras.optimizers.SGD(),
                    loss=loss_fn,
                    metrics=[metrics.PearsonR(num_targets, summarize=False),
                             metrics.R2(num_targets, summarize=False)])

    # evaluate
    return model.evaluate(seq_data.dataset)


  def get_bn_layer(self, bn_layer_i=0):
    """ Return specified batch normalization layer. """
    bn_layers = [layer for layer in self.model.layers if layer.name.startswith('batch_normalization')]
    return bn_layers[bn_layer_i]


  def get_conv_layer(self, conv_layer_i=0):
    """ Return specified convolution layer. """
    conv_layers = [layer for layer in self.model.layers if layer.name.startswith('conv')]
    return conv_layers[conv_layer_i]


  def get_dense_layer(self, layer_i=0):
    """ Return specified convolution layer. """
    dense_layers = [layer for layer in self.model.layers if layer.name.startswith('dense')]
    return dense_layers[layer_i]


  def get_conv_weights(self, conv_layer_i=0):
    """ Return kernel weights for specified convolution layer. """
    conv_layer = self.get_conv_layer(conv_layer_i)
    weights = conv_layer.weights[0].numpy()
    weights = np.transpose(weights, [2,1,0])
    return weights

  def gradients(self, seq_1hot, head_i=None, target_slice=None, pos_slice=None, pos_mask=None, pos_slice_denom=None, pos_mask_denom=None, chunk_size=None, batch_size=1, track_scale=1., track_transform=1., clip_soft=None, pseudo_count=0., no_transform=False, use_mean=False, use_ratio=False, use_logodds=False, subtract_avg=True, input_gate=True, smooth_grad=False, n_samples=5, sample_prob=0.875, dtype='float16'):
    """ Compute input gradients for sequences (GPU-friendly). """
    
    # start time
    t0 = time.time()
    
    # choose model
    if self.ensemble is not None:
      model = self.ensemble
    elif head_i is not None:
      model = self.models[head_i]
    else:
      model = self.model
    
    # verify tensor shape(s)
    seq_1hot = seq_1hot.astype('float32')
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
    
    if len(seq_1hot.shape) < 3:
      seq_1hot = seq_1hot[None, ...]
    
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
    
    # chunk parameters
    num_chunks = 1
    if chunk_size is None :
      chunk_size = seq_1hot.shape[0]
    else :
      num_chunks = int(np.ceil(seq_1hot.shape[0] / chunk_size))
    
    # loop over chunks
    grad_chunks = []
    for ci in range(num_chunks) :
      
      # collect chunk
      seq_1hot_chunk = seq_1hot[ci * chunk_size:(ci+1) * chunk_size, ...]
      target_slice_chunk = target_slice[ci * chunk_size:(ci+1) * chunk_size, ...]
      pos_slice_chunk = pos_slice[ci * chunk_size:(ci+1) * chunk_size, ...]
      
      pos_mask_chunk = None
      if pos_mask is not None :
        pos_mask_chunk = pos_mask[ci * chunk_size:(ci+1) * chunk_size, ...]
      
      pos_slice_denom_chunk = None
      pos_mask_denom_chunk = None
      if use_ratio and pos_slice_denom is not None :
        pos_slice_denom_chunk = pos_slice_denom[ci * chunk_size:(ci+1) * chunk_size, ...]
        
        if pos_mask_denom is not None :
          pos_mask_denom_chunk = pos_mask_denom[ci * chunk_size:(ci+1) * chunk_size, ...]
      
      actual_chunk_size = seq_1hot_chunk.shape[0]
      
      # sample noisy (discrete) perturbations of the input pattern chunk
      if smooth_grad :
        seq_1hot_chunk_corrupted = np.repeat(np.copy(seq_1hot_chunk), n_samples, axis=0)
    
        for example_ix in range(seq_1hot_chunk.shape[0]) :
          for sample_ix in range(n_samples) :
            corrupt_index = np.nonzero(np.random.rand(seq_1hot_chunk.shape[1]) >= sample_prob)[0]

            rand_nt_index = np.random.choice([0, 1, 2, 3], size=(corrupt_index.shape[0],))

            seq_1hot_chunk_corrupted[example_ix * n_samples + sample_ix, corrupt_index, :] = 0.
            seq_1hot_chunk_corrupted[example_ix * n_samples + sample_ix, corrupt_index, rand_nt_index] = 1.
        
        seq_1hot_chunk = seq_1hot_chunk_corrupted
        target_slice_chunk = np.repeat(np.copy(target_slice_chunk), n_samples, axis=0)
        pos_slice_chunk = np.repeat(np.copy(pos_slice_chunk), n_samples, axis=0)
        
        if pos_mask is not None :
          pos_mask_chunk = np.repeat(np.copy(pos_mask_chunk), n_samples, axis=0)
        
        if use_ratio and pos_slice_denom is not None :
          pos_slice_denom_chunk = np.repeat(np.copy(pos_slice_denom_chunk), n_samples, axis=0)
        
          if pos_mask_denom is not None :
            pos_mask_denom_chunk = np.repeat(np.copy(pos_mask_denom_chunk), n_samples, axis=0)

      # convert to tf tensors
      seq_1hot_chunk = tf.convert_to_tensor(seq_1hot_chunk, dtype=tf.float32)
      target_slice_chunk = tf.convert_to_tensor(target_slice_chunk, dtype=tf.int32)
      pos_slice_chunk = tf.convert_to_tensor(pos_slice_chunk, dtype=tf.int32)
      
      if pos_mask is not None :
        pos_mask_chunk = tf.convert_to_tensor(pos_mask_chunk, dtype=tf.float32)
      
      if use_ratio and pos_slice_denom is not None :
        pos_slice_denom_chunk = tf.convert_to_tensor(pos_slice_denom_chunk, dtype=tf.int32)
        
        if pos_mask_denom is not None :
          pos_mask_denom_chunk = tf.convert_to_tensor(pos_mask_denom_chunk, dtype=tf.float32)
      
      # batching parameters
      num_batches = int(np.ceil(actual_chunk_size * (n_samples if smooth_grad else 1) / batch_size))
      
      # loop over batches
      grad_batches = []
      for bi in range(num_batches) :
        
        # collect batch
        seq_1hot_batch = seq_1hot_chunk[bi * batch_size:(bi+1) * batch_size, ...]
        target_slice_batch = target_slice_chunk[bi * batch_size:(bi+1) * batch_size, ...]
        pos_slice_batch = pos_slice_chunk[bi * batch_size:(bi+1) * batch_size, ...]
        
        pos_mask_batch = None
        if pos_mask is not None :
          pos_mask_batch = pos_mask_chunk[bi * batch_size:(bi+1) * batch_size, ...]
        
        pos_slice_denom_batch = None
        pos_mask_denom_batch = None
        if use_ratio and pos_slice_denom is not None :
          pos_slice_denom_batch = pos_slice_denom_chunk[bi * batch_size:(bi+1) * batch_size, ...]
          
          if pos_mask_denom is not None :
            pos_mask_denom_batch = pos_mask_denom_chunk[bi * batch_size:(bi+1) * batch_size, ...]

        grad_batch = self.gradients_func(model, seq_1hot_batch, target_slice_batch, pos_slice_batch, pos_mask_batch, pos_slice_denom_batch, pos_mask_denom_batch, track_scale, track_transform, clip_soft, pseudo_count, no_transform, use_mean, use_ratio, use_logodds, subtract_avg, input_gate).numpy().astype(dtype)
      
        grad_batches.append(grad_batch)
    
      # concat gradient batches
      grads = np.concatenate(grad_batches, axis=0)
    
      # aggregate noisy gradient perturbations
      if smooth_grad :
        grads_smoothed = np.zeros((grads.shape[0] // n_samples, grads.shape[1], grads.shape[2]), dtype='float32')
        
        for example_ix in range(grads_smoothed.shape[0]) :
          for sample_ix in range(n_samples) :
            grads_smoothed[example_ix, ...] += grads[example_ix * n_samples + sample_ix, ...]
        
        grads = grads_smoothed / float(n_samples)
        grads = grads.astype(dtype)
      
      grad_chunks.append(grads)
      
      # collect garbage
      gc.collect()
    
    # concat gradient chunks
    grads = np.concatenate(grad_chunks, axis=0)
    
    # aggregate and broadcast to original input pattern
    if input_gate :
      grads = np.sum(grads, axis=-1, keepdims=True) * seq_1hot
    
    print('Completed gradient computation in %ds' % (time.time()-t0))

    return grads

  @tf.function
  def gradients_func(self, model, seq_1hot, target_slice, pos_slice, pos_mask=None, pos_slice_denom=None, pos_mask_denom=True, track_scale=1., track_transform=1., clip_soft=None, pseudo_count=0., no_transform=False, use_mean=False, use_ratio=False, use_logodds=False, subtract_avg=True, input_gate=True):
    
    with tf.GradientTape() as tape:
      tape.watch(seq_1hot)

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
      
      # aggregate over tracks (average)
      preds = tf.reduce_mean(preds, axis=-1)

      # slice specified positions
      preds_slice = tf.gather(preds, pos_slice, axis=-1, batch_dims=1)
      if pos_mask is not None :
        preds_slice = preds_slice * pos_mask
      
      # slice denominator positions
      if use_ratio and pos_slice_denom is not None:
        preds_slice_denom = tf.gather(preds, pos_slice_denom, axis=-1, batch_dims=1)
        if pos_mask_denom is not None :
          preds_slice_denom = preds_slice_denom * pos_mask_denom
      
      # aggregate over positions
      if not use_mean :
        preds_agg = tf.reduce_sum(preds_slice, axis=-1)
        if use_ratio and pos_slice_denom is not None:
          preds_agg_denom = tf.reduce_sum(preds_slice_denom, axis=-1)
      else :
        if pos_mask is not None :
          preds_agg = tf.reduce_sum(preds_slice, axis=-1) / tf.reduce_sum(pos_mask, axis=-1)
        else :
          preds_agg = tf.reduce_mean(preds_slice, axis=-1)
        
        if use_ratio and pos_slice_denom is not None:
          if pos_mask_denom is not None :
            preds_agg_denom = tf.reduce_sum(preds_slice_denom, axis=-1) / tf.reduce_sum(pos_mask_denom, axis=-1)
          else :
            preds_agg_denom = tf.reduce_mean(preds_slice_denom, axis=-1)

      # compute final statistic to take gradient of
      if no_transform :
        score_ratios = preds_agg
      elif not use_ratio :
        score_ratios = tf.math.log(preds_agg + pseudo_count + 1e-6)
      else :
        if not use_logodds :
          score_ratios = tf.math.log((preds_agg + pseudo_count) / (preds_agg_denom + pseudo_count) + 1e-6)
        else :
          score_ratios = tf.math.log(((preds_agg + pseudo_count) / (preds_agg_denom + pseudo_count)) / (1. - ((preds_agg + pseudo_count) / (preds_agg_denom + pseudo_count))) + 1e-6)

    # compute gradient
    grads = tape.gradient(score_ratios, seq_1hot)

    # zero mean each position
    if subtract_avg :
      grads = grads - tf.reduce_mean(grads, axis=-1, keepdims=True)
    
    # multiply by input 
    if input_gate :
      grads = grads * seq_1hot

    return grads

  def gradients_orig(self, seq_1hot, head_i=None, pos_slice=None, batch_size=2, dtype='float16'):
    """ Compute input gradients sequence (original version of code). """
    # choose model
    if self.ensemble is not None:
      model = self.ensemble
    elif head_i is not None:
      model = self.models[head_i]
    else:
      model = self.model

    # verify tensor shape
    seq_1hot = seq_1hot.astype('float32')
    seq_1hot = tf.convert_to_tensor(seq_1hot, dtype=tf.float32)
    if len(seq_1hot.shape) < 3:
      seq_1hot = tf.expand_dims(seq_1hot, axis=0)

    # batching parameters
    num_targets = model.output_shape[-1]
    num_batches = int(np.ceil(num_targets / batch_size))

    ti_start = 0
    grads = []
    for bi in range(num_batches):
      # sequence input
      sequence = tf.keras.Input(shape=(self.seq_length, 4), name='sequence')

      # predict
      predictions = model(sequence)

      # slice
      ti_end = min(num_targets, ti_start + batch_size)
      target_slice = np.arange(ti_start, ti_end)
      predictions_slice = tf.gather(predictions, target_slice, axis=-1)

      # replace model
      model_batch = tf.keras.Model(inputs=sequence, outputs=predictions_slice)

      t0 = time.time()

      # with tf.GradientTape() as tape:
      #   tape.watch(seq_1hot)

      #   # predict
      #   preds = model_batch(seq_1hot, training=False)

      #   if pos_slice is not None:
      #     # slice specified positions
      #     preds = tf.gather(preds, pos_slice, axis=-2)
        
      #   # sum across positions
      #   preds = tf.reduce_sum(preds, axis=-2)

      # # compute jacboian
      # grads_batch = tape.jacobian(preds, seq_1hot)
      # grads_batch = tf.squeeze(grads_batch)
      # grads_batch = tf.transpose(grads_batch, [1,2,0])

      # # zero mean each position
      # grads_batch = grads_batch - tf.reduce_mean(grads_batch, axis=-2, keepdims=True)

      
      grads_batch = self.gradients_func_orig(model_batch, seq_1hot, pos_slice)
      print('Batch gradient computation in %ds' % (time.time()-t0))

      # convert numpy dtype
      grads_batch = grads_batch.numpy().astype(dtype)
      grads.append(grads_batch)

      # next batch
      ti_start += batch_size

    # concat target batches
    grads = np.concatenate(grads, axis=-1)

    return grads

  @tf.function
  def gradients_func_orig(self, model, seq_1hot, pos_slice):
    with tf.GradientTape() as tape:
      tape.watch(seq_1hot)

      # predict
      preds = model(seq_1hot, training=False)

      if pos_slice is not None:
        # slice specified positions
        preds = tf.gather(preds, pos_slice, axis=-2)
      
      # sum across positions
      preds = tf.reduce_sum(preds, axis=-2)

    # compute jacboian
    grads = tape.jacobian(preds, seq_1hot)
    grads = tf.squeeze(grads)
    grads = tf.transpose(grads, [1,2,0])

    # zero mean each position
    grads = grads - tf.reduce_mean(grads, axis=-2, keepdims=True)

    return grads


  def num_targets(self, head_i=None):
    if head_i is None:
      return self.model.output_shape[-1]
    else:
      return self.models[head_i].output_shape[-1]


  def __call__(self, x, head_i=None, dtype='float32'):
    """ Predict targets for SeqDataset. """
    # choose model
    if self.ensemble is not None:
      model = self.ensemble
    elif head_i is not None:
      model = self.models[head_i]
    else:
      model = self.model

    return model(x).numpy().astype(dtype)
      

  def predict(self, seq_data, head_i=None, generator=False, stream=False, step=1, dtype='float32', **kwargs):
    """ Predict targets for SeqDataset. """
    # choose model
    if self.ensemble is not None:
      model = self.ensemble
    elif head_i is not None:
      model = self.models[head_i]
    else:
      model = self.model

    dataset = getattr(seq_data, 'dataset', None)
    if dataset is None:
      dataset = seq_data

    # step slice
    preds_len = model.outputs[0].shape[1]
    step_i = np.arange(0, preds_len, step)

    # predict
    if generator:
      preds = model.predict_generator(dataset, **kwargs).astype(dtype)
    elif stream:
      preds = []
      for x, y in seq_data.dataset:
        yh = model.predict(x, **kwargs)
        if step > 1:
          yh = yh[:,step_i,:]
        preds.append(yh.astype(dtype))
      preds = np.concatenate(preds, axis=0, dtype=dtype)
    else:
      preds = model.predict(dataset, **kwargs).astype(dtype)
    
    if not stream and step > 1:
      preds = preds[:,step_i,:]

    return preds


  def restore(self, model_file, head_i=0, trunk=False):
    """ Restore weights from saved model. """
    if trunk:
      self.model_trunk.load_weights(model_file)
    else:
      self.models[head_i].load_weights(model_file)
      self.model = self.models[head_i]


  def save(self, model_file, trunk=False):
    if trunk:
      self.model_trunk.save(model_file, include_optimizer=False)
    else:
      self.model.save(model_file, include_optimizer=False)

  def step(self, step=2, head_i=None):
    """ Step positions across sequence. """
    # choose model
    if self.ensemble is not None:
      model = self.ensemble
    elif head_i is not None:
      model = self.models[head_i]
    else:
      model = self.model

    # sequence input
    sequence = tf.keras.Input(shape=(self.seq_length, 4), name='sequence')

    # predict and step across positions
    preds = model(sequence)
    step_positions = np.arange(preds.shape[1], step=step)
    preds_step = tf.gather(preds, step_positions, axis=-2)
    model_step = tf.keras.Model(inputs=sequence, outputs=preds_step)

    # replace model
    if self.ensemble is not None:
      self.ensemble = model_step
    elif head_i is not None:
      self.models[head_i] = model_step
    else:
      self.model = model_step
