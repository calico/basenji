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
import sys
import time

import h5py
import numpy as np
import tensorflow as tf

import basenji

"""basenji_final.py

Write the weights from the final model layer.

"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <model_file>'
  parser = OptionParser(usage)
  parser.add_option('-o', dest='out_npy', default='final.npy')
  (options, args) = parser.parse_args()

  if len(args) != 2:
    parser.error('Must provide parameters, model, and test data HDF5')
  else:
    params_file = args[0]
    model_file = args[1]

  #######################################################
  # model parameters and placeholders

  job = basenji.dna_io.read_job_params(params_file)
  model = basenji.seqnn.SeqNN()
  model.build(job)

  # initialize saver
  saver = tf.train.Saver()

  with tf.Session() as sess:
    # load variables into session
    saver.restore(sess, model_file)

    for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
      if v.name == 'final/dense/kernel:0':
        np.save(options.out_npy, v.eval())


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
