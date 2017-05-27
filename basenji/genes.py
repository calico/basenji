#!/usr/bin/env python
import sys

import h5py
import numpy as np
import pandas as pd

if sys.version[0] == '2':
    import hail

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

        self.transcript_map = {}
        for ti in range(len(self.transcripts)):
            self.transcript_map[self.transcripts[ti]] = (transcript_index[ti], transcript_pos[ti])


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
        gene_indexes = {}
        self.transcript_gene_indexes = []
        for gid in self.genes:
            if gid not in gene_indexes:
                gene_indexes[gid] = len(gene_indexes)
            self.transcript_gene_indexes.append(gene_indexes[gid])


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


    def gene_seqs(self):
        ''' Generate GeneSeq objects. '''

        for si in range(self.num_seqs):
            chrom, start, end = self.seq_coords[si]
            yield GeneSeq(chrom, start, end, self.seqs_1hot[si,:,:], self.seq_transcripts[si])


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



class GeneSeq:
    def __init__(self, chrom, start, end, seq_1hot, transcripts_si):
        self.chrom = chrom
        self.start = start
        self.end = end
        self.seq_1hot = seq_1hot

        self.transcript_map = {}
        for tx_id, tx_pos in transcripts_si:
            self.transcript_map[tx_id] = tx_pos


    def genotypes(self, vds_open):
        ''' Load the genotypes overlapping this gene sequence from a Hail VDS. '''

        hail_contig = str(self.chrom)
        if hail_contig.startswith('chr'):
            hail_contig = hail_contig[3:]

        # filter to variants associated with this sequence
        interval_str = '%s:%d-%d' % (hail_contig, self.start, self.end)
        gene_vds_open = vds_open.filter_intervals(hail.Interval.parse(interval_str))

        # extract variant info and genotypes
        self.variants_df = gene_vds_open.make_table('v = v', ['gtj = g.gtj', 'gtk = g.gtk']).to_pandas()
        self.num_variants = self.variants_df.shape[0]


    def gene_preds(self, haplotype_preds):
        ''' Return sample gene predictions for this sequence. '''

        # get sample labels
        samples2 = [sample[:-4] for sample in self.variants_df.columns[4:]]
        samples = [samples2[i] for i in range(len(samples2)) if i%2==0]

        # construct a DataFrame with columns sample_label / gene / target / prediction
        col_sample = []
        col_transcript = []
        col_target = []
        col_pred = []

        # for each sample
        for ii in range(len(self.sample_haplotype1)):
            # get haplotype indexes
            hi1 = self.sample_haplotype1[ii]
            hi2 = self.sample_haplotype2[ii]

            # for each transcript
            for tx_id in self.transcript_map:
                tx_pos = self.transcript_map[tx_id]

                # average haplotype predictions
                preds1 = haplotype_preds[hi1,tx_pos,:]
                preds2 = haplotype_preds[hi2,tx_pos,:]
                preds = (preds1 + preds2)*0.5

                # for each target
                for ti in range(haplotype_preds.shape[2]):
                    # append values
                    col_sample.append(samples[ii])
                    col_transcript.append(tx_id)
                    col_target.append(ti)
                    col_pred.append(preds[ti])

        return pd.DataFrame({'sample':col_sample, 'transcript':col_transcript, 'target':col_target, 'pred':col_pred}, columns=['sample','transcript','target','pred'])


    def haplotypes(self):
        ''' Define a non-redundant set of haplotypes from the genotypes. '''

        # extract genotype matrix from variants dataframe
        genotype_matrix = self.variants_df.iloc[:,4:].values
        N = genotype_matrix.shape[1] // 2

        # compute non-redundant haplotypes
        # map samples to them
        haplotype_indexes = {}
        self.haplotype_genotypes = []
        self.sample_haplotype1 = np.zeros(N, dtype='int')
        self.sample_haplotype2 = np.zeros(N, dtype='int')

        for ii in range(N):
            # form hashable genotype tuple for allele 1
            hap1_key = tuple(genotype_matrix[:,2*ii])
            if hap1_key in haplotype_indexes:
                self.sample_haplotype1[ii] = haplotype_indexes[hap1_key]
            else:
                hi = len(haplotype_indexes)
                haplotype_indexes[hap1_key] = hi
                self.sample_haplotype1[ii] = hi
                self.haplotype_genotypes.append(list(hap1_key))

            # form hashable genotype tuple for allele 2
            hap2_key = tuple(genotype_matrix[:,2*ii+1])
            if hap2_key in haplotype_indexes:
                self.sample_haplotype2[ii] = haplotype_indexes[hap2_key]
            else:
                hi = len(haplotype_indexes)
                haplotype_indexes[hap2_key] = hi
                self.sample_haplotype2[ii] = hi
                self.haplotype_genotypes.append(list(hap2_key))

        # save haplotype genotypes array
        self.haplotype_genotypes = np.array(self.haplotype_genotypes)


    def haplotypes_1hot(self):
        ''' Construct an array of 1-hot coded haplotype sequences for predicting. '''

        # set all major alleles
        for vi in range(self.num_variants):
            v = self.variants_df.iloc[vi]
            assert(len(v['v.ref']) == 1)   # not ready for indels yet
            vpos = v['v.start']
            vpos_seq = vpos - self.start
            basenji.dna_io.hot1_set(self.seq_1hot, vpos_seq, v['v.ref'])

        # construct haplotype variants
        haps_1hot = []

        # for each haplotype
        for hi in range(self.haplotype_genotypes.shape[0]):
            hap_geno = self.haplotype_genotypes[hi,:]
            hap_1hot = np.copy(self.seq_1hot)

            # for each variant
            for vi in range(self.haplotype_genotypes.shape[1]):
                hv = self.haplotype_genotypes[hi,vi]

                # if minor allele
                if hv > 0:
                    v = self.variants_df.iloc[vi]
                    allele = v['v.altAlleles'][hv-1].alt
                    assert(len(allele) == 1)   # not ready for indels yet
                    vpos = v['v.start']
                    vpos_seq = vpos - self.start

                    # set allele
                    basenji.dna_io.hot1_set(hap_1hot, vpos_seq, allele)

            haps_1hot.append(hap_1hot)

        return np.array(haps_1hot)

