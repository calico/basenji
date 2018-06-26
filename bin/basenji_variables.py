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
import tensorflow as tf

from basenji import params
from basenji import seqnn

"""
basenji_variables.py

Print a model's variables, typically for debugging purposes.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file>'
  parser = OptionParser(usage)
  (options, args) = parser.parse_args()

  if len(args) != 1:
    parser.error('Must provide parameters file.')
  else:
    params_file = args[0]

  #######################################################
  # model parameters and placeholders

  job = params.read_job_params(params_file)
  model = seqnn.SeqNN()
  model.build(job)

  for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    print(v.name, v.shape)


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
