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

from collections import OrderedDict
import sys

import h5py
import numpy as np
import pandas as pd

import basenji.dna_io

class GeneData:
    def __init__(self, genes_hdf5_file, worker_index=None, workers=None):
        # open HDF5
        self.genes_hdf5_in = h5py.File(genes_hdf5_file)

        # simple stats
        self.num_seqs, self.seq_length, self.seq_depth = self.genes_hdf5_in['seqs_1hot'].shape
        self.pool_width = int(np.array(self.genes_hdf5_in['pool_width']))

        #########################################
        # gene sequences

        seq_chrom = [chrom.decode('UTF-8') for chrom in self.genes_hdf5_in['seq_chrom']]
        seq_start = list(self.genes_hdf5_in['seq_start'])
        seq_end = list(self.genes_hdf5_in['seq_end'])
        self.seq_coords = list(zip(seq_chrom,seq_start,seq_end))

        self.seqs_1hot = np.array(self.genes_hdf5_in['seqs_1hot'])


        #########################################
        # transcript information

        # map transcripts to their sequences and positions

        self.transcripts = [tx.decode('UTF-8') for tx in self.genes_hdf5_in['transcripts']]
        transcript_index = list(self.genes_hdf5_in['transcript_index'])
        transcript_pos = list(self.genes_hdf5_in['transcript_pos'])

        self.transcript_map = OrderedDict()
        for ti in range(len(self.transcripts)):
            self.transcript_map[self.transcripts[ti]] = (transcript_index[ti], transcript_pos[ti])
        self.num_transcripts = len(self.transcript_map)


        # map sequences to their transcripts

        self.seq_transcripts = []
        for si in range(len(self.seq_coords)):
            self.seq_transcripts.append([])

        for transcript in self.transcript_map:
            tx_index, tx_pos = self.transcript_map[transcript]
            self.seq_transcripts[tx_index].append((transcript,tx_pos))


        #########################################
        # gene information

        self.genes = [gid.decode('UTF-8') for gid in self.genes_hdf5_in['genes']]

        self.transcript_genes = {}
        for ti in range(len(self.transcripts)):
            self.transcript_genes[self.transcripts[ti]] = self.genes[ti]

        # map transcript indexes to gene indexes
        self.gene_indexes = OrderedDict()
        self.transcript_gene_indexes = []
        for gid in self.genes:
            if gid not in self.gene_indexes:
                self.gene_indexes[gid] = len(self.gene_indexes)
            self.transcript_gene_indexes.append(self.gene_indexes[gid])


        # determine genes split across sequences
        gene_seqs = {}
        for seq_i in range(len(self.seq_coords)):
            for transcript, tx_pos in self.seq_transcripts[seq_i]:
                gene = self.transcript_genes[transcript]
                gene_seqs.setdefault(gene,set()).add(seq_i)

        self.multi_seq_genes = set()
        for gene in gene_seqs:
            if len(gene_seqs[gene]) > 1:
                self.multi_seq_genes.add(gene)


        #########################################
        # target information

        if 'transcript_targets' in self.genes_hdf5_in:
            self.transcript_targets = self.genes_hdf5_in['transcript_targets']
            self.target_labels = [tl.decode('UTF-8') for tl in self.genes_hdf5_in['target_labels']]
            self.num_targets = len(self.target_labels)

        else:
            self.transcript_targets = None
            self.target_labels = None
            self.num_targets = None


    def subset_transcripts(self, transcripts):
        ''' Limit the sequences to a subset containing the given transcripts. '''

        seq_mask = np.zeros(self.num_seqs, dtype='bool')
        for si in range(self.num_seqs):
            # check this sequence's transcripts for matches
            seq_si_mask = [tx_id in transcripts for tx_id, tx_pos in self.seq_transcripts[si]]

            # if some transcripts match
            if np.sum(seq_si_mask) > 0:
                # keep the sequence
                seq_mask[si] = True

                # filter the transcript list
                self.seq_transcripts[si] = [self.seq_transcripts[si][sti] for sti in range(len(seq_si_mask)) if seq_si_mask[sti]]

        # filter the sequence data structures
        self.seq_coords = [self.seq_coords[si] for si in range(self.num_seqs) if seq_mask[si]]
        self.seqs_1hot = self.seqs_1hot[seq_mask,:,:]
        self.seq_transcripts = [self.seq_transcripts[si] for si in range(self.num_seqs) if seq_mask[si]]
        self.num_seqs = len(self.seq_coords)

        # transcript_map will point to the wrong sequences


    def worker(self, wi, worker_num):
        ''' Limit the sequences to one worker's share. '''

        worker_mask = np.array([si % worker_num == wi for si in range(self.num_seqs)])

        self.seqs_1hot = self.seqs_1hot[worker_mask,:,:]
        self.seq_coords = [self.seq_coords[si] for si in range(self.num_seqs) if worker_mask[si]]
        self.seq_transcripts = [self.seq_transcripts[si] for si in range(self.num_seqs) if worker_mask[si]]
        self.num_seqs = len(self.seq_coords)

        # transcript_map will point to the wrong sequences


    def __exit__(self):
        # close HDF5
        self.genes_hdf5_in.close()
