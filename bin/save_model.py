#!/usr/bin/env python
# Copyright 2020 Calico LLC
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

import json
import tensorflow as tf
if tf.__version__[0] == '1':
  tf.compat.v1.enable_eager_execution()

from basenji import seqnn

"""
save_model.py

Restore a model, and then re-save in a different format and/or with the trunk only.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <in_model_file> <out_model_file>'
  parser = OptionParser(usage)
  parser.add_option('-t','--trunk', dest='trunk',
    default=False, action='store_true',
    help='Save only trunk [Default: %default]')
  (options, args) = parser.parse_args()

  if len(args) != 3:
    parser.error('Must provide parameters, input model, and output model')
  else:
    params_file = args[0]
    in_model_file = args[1]
    out_model_file = args[2]

  # read model parameters
  with open(params_file) as params_open:
    params = json.load(params_open)
  params_model = params['model']
  params_train = params['train']

  # restore model
  seqnn_model = seqnn.SeqNN(params_model)
  seqnn_model.restore(in_model_file)

  # save
  seqnn_model.save(out_model_file, trunk=options.trunk)

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
