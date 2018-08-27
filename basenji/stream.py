# Copyright 2017 Calico LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from __future__ import print_function

import basenji

class PredStream:
  """ Interface to acquire predictions via a buffered stream mechanism
         rather than getting them all at once and using excessive memory. """

  def __init__(self, sess, model, stream_length, verbose=False):
    self.sess = sess
    self.model = model
    self.verbose = verbose

    self.stream_start = 0
    self.stream_end = 0

    if stream_length % self.model.hp.batch_size != 0:
      print(
          'Make the stream length a multiple of the batch size',
          file=sys.stderr)
      exit(1)
    else:
      self.stream_batches = stream_length // self.model.hp.batch_size

  def __getitem__(self, i):
    # acquire predictions, if needed
    if i >= self.stream_end:
      # update start
      self.stream_start = self.stream_end

      if self.verbose:
        print('Predicting from %d' % self.stream_start, flush=True)

      # predict
      self.stream_preds = self.model.predict_tfr(self.sess,
                                                test_batches=self.stream_batches)

      # update end
      self.stream_end = self.stream_start + self.stream_preds.shape[0]

    return self.stream_preds[i - self.stream_start]


class PredStreamFeed:
  """ Interface to acquire predictions via a buffered stream mechanism
         rather than getting them all at once and using excessive memory. """

  def __init__(self, sess, model, seqs_1hot, stream_length):
    self.sess = sess
    self.model = model

    self.seqs_1hot = seqs_1hot

    self.stream_length = stream_length
    self.stream_start = 0
    self.stream_end = 0

    if self.stream_length % self.model.hp.batch_size != 0:
      print(
          'Make the stream length a multiple of the batch size',
          file=sys.stderr)
      exit(1)

  def __getitem__(self, i):
    # acquire predictions, if needed
    if i >= self.stream_end:
      self.stream_start = self.stream_end
      self.stream_end = min(self.stream_start + self.stream_length,
                            self.seqs_1hot.shape[0])

      # subset sequences
      stream_seqs_1hot = self.seqs_1hot[self.stream_start:self.stream_end]

      # initialize batcher
      batcher = basenji.batcher.Batcher(
          stream_seqs_1hot, batch_size=self.model.hp.batch_size)

      # predict
      self.stream_preds = self.model.predict(self.sess, batcher, rc_avg=False)

    return self.stream_preds[i - self.stream_start]


class PredGradStream:
  """ Interface to acquire predictions and gradients via a buffered stream

         mechanism rather than getting them all at once and using excessive
         memory.
  """

  def __init__(self, sess, model, seqs_1hot, stream_length):
    self.sess = sess
    self.model = model

    self.seqs_1hot = seqs_1hot

    self.stream_length = stream_length
    self.stream_start = 0
    self.stream_end = 0

    if self.stream_length % self.model.hp.batch_size != 0:
      print(
          'Make the stream length a multiple of the batch size',
          file=sys.stderr)
      exit(1)

  def __getitem__(self, i):
    # acquire predictions, if needed
    if i >= self.stream_end:
      self.stream_start = self.stream_end
      self.stream_end = min(self.stream_start + self.stream_length,
                            self.seqs_1hot.shape[0])

      # subset sequences
      stream_seqs_1hot = self.seqs_1hot[self.stream_start:self.stream_end]

      # initialize batcher
      batcher = basenji.batcher.Batcher(
          stream_seqs_1hot, batch_size=self.model.hp.batch_size)

      # predict
      self.stream_grads, self.stream_preds = self.model.gradients(
          self.sess, batcher, layers=[0], return_preds=True)

      # take first layer
      self.stream_grads = self.stream_grads[0]

    return self.stream_preds[i - self.stream_start], self.stream_grads[
        i - self.stream_start]
