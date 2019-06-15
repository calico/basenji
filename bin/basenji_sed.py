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
import gc
import os
import pdb
import pickle
import subprocess
import sys

import h5py
import numpy as np
import pyBigWig

from basenji import batcher
from basenji import dna_io
from basenji import gene
from basenji import genedata
from basenji import params
from basenji import seqnn
from basenji import vcf as bvcf
from basenji_test import bigwig_open

import tensorflow as tf

"""basenji_sed.py

Compute SNP expression difference (SED) scores for SNPs in a VCF file.
"""

################################################################################
# main
################################################################################
def main():
    usage = (
        "usage: %prog [options] <params_file> <model_file> <genes_hdf5_file>"
        " <vcf_file>"
    )
    parser = OptionParser(usage)
    parser.add_option(
        "-a",
        dest="all_sed",
        default=False,
        action="store_true",
        help="Print all variant-gene pairs, as opposed to only nonzero [Default: %default]",
    )
    parser.add_option(
        "-b",
        dest="batch_size",
        default=None,
        type="int",
        help="Batch size [Default: %default]",
    )
    parser.add_option(
        "-c",
        dest="csv",
        default=False,
        action="store_true",
        help="Print table as CSV [Default: %default]",
    )
    parser.add_option(
        "-g",
        dest="genome_file",
        default="%s/data/human.hg19.genome" % os.environ["BASENJIDIR"],
        help="Chromosome lengths file [Default: %default]",
    )
    parser.add_option(
        "-o",
        dest="out_dir",
        default="sed",
        help="Output directory for tables and plots [Default: %default]",
    )
    parser.add_option(
        "-p",
        dest="processes",
        default=None,
        type="int",
        help="Number of processes, passed by multi script",
    )
    parser.add_option(
        "--pseudo",
        dest="log_pseudo",
        default=0.125,
        type="float",
        help="Log2 pseudocount [Default: %default]",
    )
    parser.add_option(
        "-r",
        dest="tss_radius",
        default=0,
        type="int",
        help="Radius of bins considered to quantify TSS transcription [Default: %default]",
    )
    parser.add_option(
        "--rc",
        dest="rc",
        default=False,
        action="store_true",
        help="Average the forward and reverse complement predictions when testing [Default: %default]",
    )
    parser.add_option(
        "--shifts",
        dest="shifts",
        default="0",
        help="Ensemble prediction shifts [Default: %default]",
    )
    parser.add_option(
        "-t",
        dest="targets_file",
        default=None,
        help="File specifying target indexes and labels in table format",
    )
    parser.add_option(
        "-u",
        dest="penultimate",
        default=False,
        action="store_true",
        help="Compute SED in the penultimate layer [Default: %default]",
    )
    parser.add_option(
        "-x",
        dest="tss_table",
        default=False,
        action="store_true",
        help="Print TSS table in addition to gene [Default: %default]",
    )
    (options, args) = parser.parse_args()

    if len(args) == 4:
        # single worker
        params_file = args[0]
        model_file = args[1]
        genes_hdf5_file = args[2]
        vcf_file = args[3]

    elif len(args) == 6:
        # multi worker
        options_pkl_file = args[0]
        params_file = args[1]
        model_file = args[2]
        genes_hdf5_file = args[3]
        vcf_file = args[4]
        worker_index = int(args[5])

        # load options
        options_pkl = open(options_pkl_file, "rb")
        options = pickle.load(options_pkl)
        options_pkl.close()

        # update output directory
        options.out_dir = "%s/job%d" % (options.out_dir, worker_index)

    else:
        parser.error(
            "Must provide parameters and model files, genes HDF5 file, and QTL VCF"
            " file"
        )

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    options.shifts = [int(shift) for shift in options.shifts.split(",")]

    #################################################################
    # reads in genes HDF5

    gene_data = genedata.GeneData(genes_hdf5_file)

    # filter for worker sequences
    if options.processes is not None:
        gene_data.worker(worker_index, options.processes)

    #################################################################
    # prep SNPs

    # load SNPs
    snps = bvcf.vcf_snps(vcf_file)

    # intersect w/ segments
    print("Intersecting gene sequences with SNPs...", end="")
    sys.stdout.flush()
    seqs_snps = bvcf.intersect_seqs_snps(vcf_file, gene_data.gene_seqs, vision_p=0.5)
    print("%d sequences w/ SNPs" % len(seqs_snps))

    #################################################################
    # setup model

    job = params.read_job_params(params_file)

    job["seq_length"] = gene_data.seq_length
    job["seq_depth"] = gene_data.seq_depth
    job["target_pool"] = gene_data.pool_width

    if "num_targets" not in job and gene_data.num_targets is not None:
        job["num_targets"] = gene_data.num_targets

    if "num_targets" not in job:
        print(
            "Must specify number of targets (num_targets) in the \
            parameters file.",
            file=sys.stderr,
        )
        exit(1)

    if options.targets_file is None:
        target_labels = gene_data.target_labels
        target_subset = None
        target_ids = gene_data.target_ids
        if target_ids is None:
            target_ids = ["t%d" % ti for ti in range(job["num_targets"])]
            target_labels = [""] * len(target_ids)

    else:
        # Unfortunately, this target file differs from some others
        # in that it needs to specify the indexes from the original
        # set. In the future, I should standardize to this version.

        target_ids = []
        target_labels = []
        target_subset = []

        for line in open(options.targets_file):
            a = line.strip().split("\t")
            target_subset.append(int(a[0]))
            target_ids.append(a[1])
            target_labels.append(a[3])

            if len(target_subset) == job["num_targets"]:
                target_subset = None

    # build model
    model = seqnn.SeqNN()
    model.build_feed(job, target_subset=target_subset)

    if options.penultimate:
        # labels become inappropriate
        target_ids = [""] * model.hp.cnn_filters[-1]
        target_labels = target_ids

    #################################################################
    # compute, collect, and print SEDs

    header_cols = (
        "rsid",
        "ref",
        "alt",
        "gene",
        "tss_dist",
        "ref_pred",
        "alt_pred",
        "sed",
        "ser",
        "target_index",
        "target_id",
        "target_label",
    )
    if options.csv:
        sed_gene_out = open("%s/sed_gene.csv" % options.out_dir, "w")
        print(",".join(header_cols), file=sed_gene_out)
        if options.tss_table:
            sed_tss_out = open("%s/sed_tss.csv" % options.out_dir, "w")
            print(",".join(header_cols), file=sed_tss_out)

    else:
        sed_gene_out = open("%s/sed_gene.txt" % options.out_dir, "w")
        print(" ".join(header_cols), file=sed_gene_out)
        if options.tss_table:
            sed_tss_out = open("%s/sed_tss.txt" % options.out_dir, "w")
            print(" ".join(header_cols), file=sed_tss_out)

    # helper variables
    pred_buffer = model.hp.batch_buffer // model.hp.target_pool

    # initialize saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # load variables into session
        saver.restore(sess, model_file)

        # for each gene sequence
        for seq_i in range(gene_data.num_seqs):
            gene_seq = gene_data.gene_seqs[seq_i]
            print(gene_seq)

            # if it contains SNPs
            seq_snps = [snps[snp_i] for snp_i in seqs_snps[seq_i]]
            if len(seq_snps) > 0:

                # one hot code allele sequences
                aseqs_1hot = alleles_1hot(
                    gene_seq, gene_data.seqs_1hot[seq_i], seq_snps
                )

                # initialize batcher
                batcher_gene = batcher.Batcher(
                    aseqs_1hot, batch_size=model.hp.batch_size
                )

                # construct allele gene_seq's
                allele_gene_seqs = [gene_seq] * aseqs_1hot.shape[0]

                # predict alleles
                allele_tss_preds = model.predict_genes(
                    sess,
                    batcher_gene,
                    allele_gene_seqs,
                    rc=options.rc,
                    shifts=options.shifts,
                    embed_penultimate=options.penultimate,
                    tss_radius=options.tss_radius,
                )

                # reshape (Alleles x TSSs) x Targets to Alleles x TSSs x Targets
                allele_tss_preds = allele_tss_preds.reshape(
                    (aseqs_1hot.shape[0], gene_seq.num_tss, -1)
                )

                # extract reference and SNP alt predictions
                ref_tss_preds = allele_tss_preds[0]  # TSSs x Targets
                alt_tss_preds = allele_tss_preds[1:]  # SNPs x TSSs x Targets

                # compute TSS SED scores
                snp_tss_sed = alt_tss_preds - ref_tss_preds
                snp_tss_ser = np.log2(alt_tss_preds + options.log_pseudo) - np.log2(
                    ref_tss_preds + options.log_pseudo
                )

                # compute gene-level predictions
                ref_gene_preds, gene_ids = gene.map_tss_genes(
                    ref_tss_preds, gene_seq.tss_list, options.tss_radius
                )
                alt_gene_preds = []
                for snp_i in range(len(seq_snps)):
                    agp, _ = gene.map_tss_genes(
                        alt_tss_preds[snp_i], gene_seq.tss_list, options.tss_radius
                    )
                    alt_gene_preds.append(agp)
                alt_gene_preds = np.array(alt_gene_preds)

                # compute gene SED scores
                gene_sed = alt_gene_preds - ref_gene_preds
                gene_ser = np.log2(alt_gene_preds + options.log_pseudo) - np.log2(
                    ref_gene_preds + options.log_pseudo
                )

                # for each SNP
                for snp_i in range(len(seq_snps)):
                    snp = seq_snps[snp_i]

                    # initialize gene data structures
                    snp_dist_gene = {}

                    # for each TSS
                    for tss_i in range(gene_seq.num_tss):
                        tss = gene_seq.tss_list[tss_i]

                        # SNP distance to TSS
                        snp_dist = abs(snp.pos - tss.pos)
                        if tss.gene_id in snp_dist_gene:
                            snp_dist_gene[tss.gene_id] = min(
                                snp_dist_gene[tss.gene_id], snp_dist
                            )
                        else:
                            snp_dist_gene[tss.gene_id] = snp_dist

                        # for each target
                        if options.tss_table:
                            for ti in range(ref_tss_preds.shape[1]):

                                # check if nonzero
                                if options.all_sed or not np.isclose(
                                    tss_sed[snp_i, tss_i, ti], 0, atol=1e-4
                                ):

                                    # print
                                    cols = (
                                        snp.rsid,
                                        bvcf.cap_allele(snp.ref_allele),
                                        bvcf.cap_allele(snp.alt_alleles[0]),
                                        tss.identifier,
                                        snp_dist,
                                        ref_tss_preds[tss_i, ti],
                                        alt_tss_preds[snp_i, tss_i, ti],
                                        tss_sed[snp_i, tss_i, ti],
                                        tss_ser[snp_i, tss_i, ti],
                                        ti,
                                        target_ids[ti],
                                        target_labels[ti],
                                    )
                                    if options.csv:
                                        print(
                                            ",".join([str(c) for c in cols]),
                                            file=sed_tss_out,
                                        )
                                    else:
                                        print(
                                            "%-13s %s %5s %16s %5d %7.4f %7.4f %7.4f %7.4f %4d %12s %s"
                                            % cols,
                                            file=sed_tss_out,
                                        )

                    # for each gene
                    for gi in range(len(gene_ids)):
                        gene_str = gene_ids[gi]
                        if gene_ids[gi] in gene_data.multi_seq_genes:
                            gene_str = "%s_multi" % gene_ids[gi]

                        # print rows to gene table
                        for ti in range(ref_gene_preds.shape[1]):

                            # check if nonzero
                            if options.all_sed or not np.isclose(
                                gene_sed[snp_i, gi, ti], 0, atol=1e-4
                            ):

                                # print
                                cols = [
                                    snp.rsid,
                                    bvcf.cap_allele(snp.ref_allele),
                                    bvcf.cap_allele(snp.alt_alleles[0]),
                                    gene_str,
                                    snp_dist_gene[gene_ids[gi]],
                                    ref_gene_preds[gi, ti],
                                    alt_gene_preds[snp_i, gi, ti],
                                    gene_sed[snp_i, gi, ti],
                                    gene_ser[snp_i, gi, ti],
                                    ti,
                                    target_ids[ti],
                                    target_labels[ti],
                                ]
                                if options.csv:
                                    print(
                                        ",".join([str(c) for c in cols]),
                                        file=sed_gene_out,
                                    )
                                else:
                                    print(
                                        "%-13s %s %5s %16s %5d %7.4f %7.4f %7.4f %7.4f %4d %12s %s"
                                        % tuple(cols),
                                        file=sed_gene_out,
                                    )

                # clean up
                gc.collect()

    sed_gene_out.close()
    if options.tss_table:
        sed_tss_out.close()


def alleles_1hot(gene_seq, seq_1hot, seq_snps):
    """ One hot code for gene sequence alleles. """

    # initialize one hot coding
    aseqs_1hot = []

    # add reference allele sequence
    aseqs_1hot.append(np.copy(seq_1hot))

    # set all reference alleles
    for snp in seq_snps:

        # determine SNP position wrt sequence
        snp_seq_pos = snp.pos - 1 - gene_seq.start

        # verify that the reference allele matches the reference
        seq_ref = dna_io.hot1_dna(
            aseqs_1hot[0][snp_seq_pos : snp_seq_pos + len(snp.ref_allele), :]
        )
        if seq_ref != snp.ref_allele:
            print(
                "WARNING: %s - ref allele %s does not match reference genome %s; changing reference genome to match."
                % (snp.rsid, snp.ref_allele, seq_ref),
                file=sys.stderr,
            )

            if len(seq_ref) == len(snp.ref_allele):
                # SNP
                dna_io.hot1_set(aseqs_1hot[0], snp_seq_pos, snp.ref_allele)

            # not confident in these operations

            # elif len(seq_ref) > len(snp.ref_allele):
            #   # deletion
            #   delete_len = len(seq_ref) - len(snp.ref_allele)
            #   dna_io.hot1_delete(aseqs_1hot[0], snp_seq_pos + 1, delete_len)

            # else:
            #   # insertion
            #   dna_io.hot1_insert(aseqs_1hot[0], snp_seq_pos + 1, snp.ref_allele[1:])

            else:
                raise Exception(
                    "ERROR: reference mismatch indels cannot yet be handled."
                )

    # for each SNP
    for snp in seq_snps:

        # determine SNP position wrt sequence
        snp_seq_pos = snp.pos - 1 - gene_seq.start

        # add minor allele sequence
        aseqs_1hot.append(np.copy(aseqs_1hot[0]))
        if len(snp.ref_allele) == len(snp.alt_alleles[0]):
            # SNP
            dna_io.hot1_set(aseqs_1hot[-1], snp_seq_pos, snp.alt_alleles[0])

        elif len(snp.ref_allele) > len(snp.alt_alleles[0]):
            # deletion
            delete_len = len(snp.ref_allele) - len(snp.alt_alleles[0])
            assert snp.ref_allele[0] == snp.alt_alleles[0][0]
            dna_io.hot1_delete(aseqs_1hot[-1], snp_seq_pos + 1, delete_len)

        else:
            # insertion
            assert snp.ref_allele[0] == snp.alt_alleles[0][0]
            dna_io.hot1_insert(aseqs_1hot[-1], snp_seq_pos + 1, snp.alt_alleles[0][1:])

    # finalize
    aseqs_1hot = np.array(aseqs_1hot)

    return aseqs_1hot


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
    # pdb.runcall(main)
