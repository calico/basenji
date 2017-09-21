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

from optparse import OptionParser

import h5py
import numpy as np
"""basenji_hdf5_sample.py

Samples sequences from an HDF5 training file.
"""


################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <in_h5> <out_h5>'
  parser = OptionParser(usage)
  parser.add_option(
      '-s',
      dest='seqs_pct',
      default=0.25,
      type='float',
      help='Propostion of sequencs to sample [Default: %default]')
  # parser.add_option('-t', dest='targets_pct', default=1, type='float', help='Proportion of targets to sample [Default: %default]')
  (options, args) = parser.parse_args()

  if len(args) != 2:
    parser.error('Must provide input and output HDF5')
  else:
    in_h5 = args[0]
    out_h5 = args[1]

  in_h5_open = h5py.File(in_h5)
  out_h5_open = h5py.File(out_h5, 'w')

  for key in in_h5_open.keys():
    print(key)
    if key[-3:] == '_in' or key[-4:] == '_out':
      n_in = in_h5_open[key].shape[0]
      n_out = int(n_in * options.seqs_pct)
      out_h5_open.create_dataset(key, data=in_h5_open[key][:n_out])
    else:
      out_h5_open.create_dataset(key, data=in_h5_open[key])

  in_h5_open.close()
  out_h5_open.close()


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
