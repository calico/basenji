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
import sys
import time

import h5py
import numpy as np
import tensorflow as tf

import basenji

"""basenji_average.py

Average parameters for multiple models.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <model1_file> <model2_file>'
  parser = OptionParser(usage)
  parser.add_option('-o', dest='out_model', default='./model_avg.tf')
  (options, args) = parser.parse_args()

  if len(args) < 3:
    parser.error('Must provide parameters file and models files.')
  else:
    params_file = args[0]
    model_files = args[1:]

  #######################################################
  # model parameters and placeholders

  job = basenji.dna_io.read_job_params(params_file)

  model = basenji.seqnn.SeqNN()
  model.build(job)

  # initialize saver
  saver = tf.train.Saver()

  with tf.Session() as sess:
    model_weights = {}

    for mi in range(0, len(model_files)):
      # load variables into session
      saver.restore(sess, model_files[mi])

      # hash weight evals
      for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        model_weights.setdefault(v.name,[]).append(v.eval())

    # initialize assign ops
    assign_ops = []
    for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
      # if v.name == 'final/dense/kernel:0':
      #   pdb.set_trace()
      weights_avg = np.mean(np.array(model_weights[v.name]), axis=0)
      assign_ops.append(tf.assign(v, weights_avg))

    # run assign ops
    sess.run(assign_ops)

    # save model
    saver.save(sess, options.out_model)

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
