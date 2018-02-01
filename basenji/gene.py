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
from collections import OrderedDict
import pdb

import numpy as np

class Gene:
  def __init__(self, gene_id, tss_list):
    self.gene_id = gene_id
    self.tss_list = tss_list


class GeneSeq:
  def __init__(self, chrom, start, end, tss_list=None):
    self.chrom = chrom
    self.start = start
    self.end = end

    if tss_list is None:
      self.tss_list = []
    else:
      self.tss_list = tss_list
    self.num_tss = len(self.tss_list)

    # map genes to TSS indexes
    self.gene_tss = OrderedDict()
    for tss_i in range(self.num_tss):
      gene_id = self.tss_list[tss_i].gene_id
      self.gene_tss.setdefault(gene_id,[]).append(tss_i)
    self.num_genes = len(self.gene_tss)

  def gene_names(self, tss=False):
    ''' Return gene/TSS names. '''
    if tss:
      return [tss.identifier for tss in self.tss_list]
    else:
      return list(self.gene_tss.keys())

  def num_tss(self):
    return len(self.tss_list)

  def __str__(self):
    return '%s:%d-%s %d TSSs' % (self.chrom, self.start, self.end, self.num_tss())


class TSS:
  def __init__(self, identifier, gene_id, chrom, pos, gene_seq, seq_index=False, strand=None):
    self.identifier = identifier
    self.gene_id = gene_id
    self.chrom = chrom
    self.pos = pos
    self.strand = strand
    self.gene_seq = gene_seq
    self.seq_index = seq_index

    # gene_seq refers to the GeneSeq object via the GeneData interface, but
    # it refers to the gene_seq's index in basenji_hdf5_genes.py

  def seq_bin(self, width=128, pred_buffer=0):
    if self.seq_index:
      raise ValueError('TSS gene_seq refers to index')

    # determine position within gene sequence
    seq_pos = self.pos - self.gene_seq.start

    # account for prediction buffer
    seq_pos -= pred_buffer

    # account for bin width
    return seq_pos // width

  def __str__(self):
    return '%s %s %s:%d' % (self.identifier, self.gene_id, self.chrom, self.pos)


def map_tss_genes(tss_values, tss_list, tss_radius=0):
  if tss_radius > 0:
    print("I don't know how to obtain an exact gene expression measurement with tss_radius > 0", file=sys.stderr)
    exit(1)

  # map genes to TSS indexes
  gene_tss = OrderedDict()
  for tss_i in range(len(tss_list)):
    gene_tss.setdefault(tss_list[tss_i].gene_id,[]).append(tss_i)

  # initialize gene values
  gene_values = np.zeros((len(gene_tss), tss_values.shape[1]), dtype='float16')

  # sum TSS into genes
  gi = 0
  for gene_id, tss_list in gene_tss.items():
    for tss_i in tss_list:
        gene_values[gi,:] += tss_values[tss_i,:]
    gi += 1

  return gene_values, list(gene_tss.keys())
