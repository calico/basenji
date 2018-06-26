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

import h5py
import tensorflow as tf

from basenji import batcher
from basenji import params
from basenji import seqnn

"""
basenji_convert.py

Convert older models to the new format. I don't expect this particular issue to
occur again, but I'm committing the script to keep an example around of how to do it.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <in_model_tf> <out_model_tf>'
  parser = OptionParser(usage)
  (options, args) = parser.parse_args()

  if len(args) != 3:
    parser.error('Must provide parameters file and input and out model stems.')
  else:
    params_file = args[0]
    in_model_tf = args[1]
    out_model_tf = args[2]

  # read parameters
  job = params.read_job_params(params_file)
  model = seqnn.SeqNN()
  model.build(job)

  # transform variables names
  restore_dict = {}
  for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    # names have ":0" suffix that Saver dislikes.
    v_key = v.name.split(':')[0]

    if v_key == 'global_step':
      pass
    elif v_key.startswith('final'):
      # conv1d to dense
      v_key = v_key.replace('dense', 'conv1d')
      restore_dict[v_key] = v
    else:
      restore_dict[v_key] = v

  # initialize savers (reshape is critical for conv1d -> dense)
  saver_read = tf.train.Saver(restore_dict, reshape=True)
  saver_write = tf.train.Saver()

  with tf.Session() as sess:
    # initialize variables
    sess.run(tf.global_variables_initializer())

    # load variables into session
    saver_read.restore(sess, in_model_tf)

    # re-save w/ new names
    saver_write.save(sess, out_model_tf)


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
