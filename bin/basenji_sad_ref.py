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
import pdb
import pickle
import os
from queue import Queue
import sys
from threading import Thread
import time

import h5py
import numpy as np
import pandas as pd
import pysam
import tensorflow as tf

import basenji.dna_io as dna_io
import basenji.params as params
import basenji.seqnn as seqnn
import basenji.vcf as bvcf
from basenji.stream import PredStream

from basenji_sad import initialize_output_h5

"""
basenji_sad_ref.py

Compute SNP Activity Difference (SAD) scores for SNPs in a VCF file.
This versions saves computation by clustering nearby SNPs in order to
make a single reference prediction for several SNPs.
"""

################################################################################
# main
################################################################################
def main():
    usage = "usage: %prog [options] <params_file> <model_file> <vcf_file>"
    parser = OptionParser(usage)
    parser.add_option(
        "-c",
        dest="center_pct",
        default=0.25,
        type="float",
        help="Require clustered SNPs lie in center region [Default: %default]",
    )
    parser.add_option(
        "-f",
        dest="genome_fasta",
        default="%s/data/hg19.fa" % os.environ["BASENJIDIR"],
        help="Genome FASTA for sequences [Default: %default]",
    )
    parser.add_option(
        "-g",
        dest="genome_file",
        default="%s/data/human.hg19.genome" % os.environ["BASENJIDIR"],
        help="Chromosome lengths file [Default: %default]",
    )
    parser.add_option(
        "--local",
        dest="local",
        default=1024,
        type="int",
        help="Local SAD score [Default: %default]",
    )
    parser.add_option("-n", dest="norm_file", default=None, help="Normalize SAD scores")
    parser.add_option(
        "-o",
        dest="out_dir",
        default="sad",
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
        default=1,
        type="float",
        help="Log2 pseudocount [Default: %default]",
    )
    parser.add_option(
        "--rc",
        dest="rc",
        default=False,
        action="store_true",
        help="Average forward and reverse complement predictions [Default: %default]",
    )
    parser.add_option(
        "--shifts",
        dest="shifts",
        default="0",
        type="str",
        help="Ensemble prediction shifts [Default: %default]",
    )
    parser.add_option(
        "--stats",
        dest="sad_stats",
        default="SAD",
        help="Comma-separated list of stats to save. [Default: %default]",
    )
    parser.add_option(
        "-t",
        dest="targets_file",
        default=None,
        type="str",
        help="File specifying target indexes and labels in table format",
    )
    parser.add_option(
        "--ti",
        dest="track_indexes",
        default=None,
        type="str",
        help="Comma-separated list of target indexes to output BigWig tracks",
    )
    parser.add_option(
        "-u",
        dest="penultimate",
        default=False,
        action="store_true",
        help="Compute SED in the penultimate layer [Default: %default]",
    )
    (options, args) = parser.parse_args()

    if len(args) == 3:
        # single worker
        params_file = args[0]
        model_file = args[1]
        vcf_file = args[2]

    elif len(args) == 5:
        # multi worker
        options_pkl_file = args[0]
        params_file = args[1]
        model_file = args[2]
        vcf_file = args[3]
        worker_index = int(args[4])

        # load options
        options_pkl = open(options_pkl_file, "rb")
        options = pickle.load(options_pkl)
        options_pkl.close()

        # update output directory
        options.out_dir = "%s/job%d" % (options.out_dir, worker_index)

    else:
        parser.error("Must provide parameters and model files and QTL VCF file")

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    if options.track_indexes is None:
        options.track_indexes = []
    else:
        options.track_indexes = [int(ti) for ti in options.track_indexes.split(",")]
        if not os.path.isdir("%s/tracks" % options.out_dir):
            os.mkdir("%s/tracks" % options.out_dir)

    options.shifts = [int(shift) for shift in options.shifts.split(",")]
    options.sad_stats = options.sad_stats.split(",")

    #################################################################
    # read parameters and collet target information

    job = params.read_job_params(params_file, require=["seq_length", "num_targets"])

    if options.targets_file is None:
        target_ids = ["t%d" % ti for ti in range(job["num_targets"])]
        target_labels = [""] * len(target_ids)
        target_subset = None

    else:
        targets_df = pd.read_table(options.targets_file, index_col=0)
        target_ids = targets_df.identifier
        target_labels = targets_df.description
        target_subset = targets_df.index
        if len(target_subset) == job["num_targets"]:
            target_subset = None

    #################################################################
    # load SNPs

    # read sorted SNPs from VCF
    snps = bvcf.vcf_snps(
        vcf_file,
        require_sorted=True,
        flip_ref=False,
        validate_ref_fasta=options.genome_fasta,
    )

    # filter for worker SNPs
    if options.processes is not None:
        worker_bounds = np.linspace(0, len(snps), options.processes + 1, dtype="int")
        snps = snps[worker_bounds[worker_index] : worker_bounds[worker_index + 1]]

    num_snps = len(snps)

    # cluster SNPs by position
    snp_clusters = cluster_snps(snps, job["seq_length"], options.center_pct)

    # delimit sequence boundaries
    [sc.delimit(job["seq_length"]) for sc in snp_clusters]

    # open genome FASTA
    genome_open = pysam.Fastafile(options.genome_fasta)

    # make SNP sequence generator
    def snp_gen():
        for sc in snp_clusters:
            snp_1hot_list = sc.get_1hots(genome_open)
            for snp_1hot in snp_1hot_list:
                yield {"sequence": snp_1hot}

    snp_types = {"sequence": tf.float32}
    snp_shapes = {
        "sequence": tf.TensorShape([tf.Dimension(job["seq_length"]), tf.Dimension(4)])
    }

    dataset = tf.data.Dataset.from_generator(
        snp_gen, output_types=snp_types, output_shapes=snp_shapes
    )
    dataset = dataset.batch(job["batch_size"])
    dataset = dataset.prefetch(2 * job["batch_size"])
    # dataset = dataset.apply(tf.contrib.data.prefetch_to_device('/device:GPU:0'))

    iterator = dataset.make_one_shot_iterator()
    data_ops = iterator.get_next()

    #################################################################
    # setup model

    # build model
    t0 = time.time()
    model = seqnn.SeqNN()
    model.build_sad(
        job,
        data_ops,
        ensemble_rc=options.rc,
        ensemble_shifts=options.shifts,
        embed_penultimate=options.penultimate,
        target_subset=target_subset,
    )
    print("Model building time %f" % (time.time() - t0), flush=True)

    if options.penultimate:
        # labels become inappropriate
        target_ids = [""] * model.hp.cnn_filters[-1]
        target_labels = target_ids

    # read target normalization factors
    target_norms = np.ones(len(target_labels))
    if options.norm_file is not None:
        ti = 0
        for line in open(options.norm_file):
            target_norms[ti] = float(line.strip())
            ti += 1

    num_targets = len(target_ids)

    #################################################################
    # setup output

    sad_out = initialize_output_h5(
        options.out_dir, options.sad_stats, snps, target_ids, target_labels
    )

    snp_threads = []

    snp_queue = Queue()
    for i in range(1):
        sw = SNPWorker(snp_queue, sad_out, options.sad_stats, options.log_pseudo)
        sw.start()
        snp_threads.append(sw)

    #################################################################
    # predict SNP scores, write output

    # initialize saver
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # load variables into session
        saver.restore(sess, model_file)

        # initialize predictions stream
        preds_stream = PredStream(sess, model, 32)

        # predictions index
        pi = 0

        # SNP index
        si = 0

        for snp_cluster in snp_clusters:
            ref_preds = preds_stream[pi]
            pi += 1

            for snp in snp_cluster.snps:
                # print(snp, flush=True)

                alt_preds = preds_stream[pi]
                pi += 1

                # queue SNP
                snp_queue.put((ref_preds, alt_preds, si))

                # update SNP index
                si += 1

    # finish queue
    print("Waiting for threads to finish.", flush=True)
    snp_queue.join()

    # close genome
    genome_open.close()

    ###################################################
    # compute SAD distributions across variants

    # define percentiles
    d_fine = 0.001
    d_coarse = 0.01
    percentiles_neg = np.arange(d_fine, 0.1, d_fine)
    percentiles_base = np.arange(0.1, 0.9, d_coarse)
    percentiles_pos = np.arange(0.9, 1, d_fine)

    percentiles = np.concatenate([percentiles_neg, percentiles_base, percentiles_pos])
    sad_out.create_dataset("percentiles", data=percentiles)
    pct_len = len(percentiles)

    for sad_stat in options.sad_stats:
        sad_stat_pct = "%s_pct" % sad_stat

        # compute
        sad_pct = np.percentile(sad_out[sad_stat], 100 * percentiles, axis=0).T
        sad_pct = sad_pct.astype("float16")

        # save
        sad_out.create_dataset(sad_stat_pct, data=sad_pct, dtype="float16")

    sad_out.close()


def cluster_snps(snps, seq_len, center_pct):
    """Cluster a sorted list of SNPs into regions that will satisfy
     the required center_pct."""
    valid_snp_distance = int(seq_len * center_pct)

    snp_clusters = []
    cluster_chr = None

    for snp in snps:
        if snp.chr == cluster_chr and snp.pos < cluster_pos0 + valid_snp_distance:
            # append to latest cluster
            snp_clusters[-1].add_snp(snp)
        else:
            # initialize new cluster
            snp_clusters.append(SNPCluster())
            snp_clusters[-1].add_snp(snp)
            cluster_chr = snp.chr
            cluster_pos0 = snp.pos

    return snp_clusters


class SNPCluster:
    def __init__(self):
        self.snps = []
        self.chr = None
        self.start = None
        self.end = None

    def add_snp(self, snp):
        self.snps.append(snp)

    def delimit(self, seq_len):
        positions = [snp.pos for snp in self.snps]
        pos_min = np.min(positions)
        pos_max = np.max(positions)
        pos_mid = (pos_min + pos_max) // 2

        self.chr = self.snps[0].chr
        self.start = pos_mid - seq_len // 2
        self.end = self.start + seq_len

        for snp in self.snps:
            snp.seq_pos = snp.pos - 1 - self.start

    def get_1hots(self, genome_open):
        seqs1_list = []

        # extract reference
        if self.start < 0:
            ref_seq = (
                "N" * (-self.start) + genome_open.fetch(self.chr, 0, self.end).upper()
            )
        else:
            ref_seq = genome_open.fetch(self.chr, self.start, self.end).upper()

        # extend to full length
        if len(ref_seq) < self.end - self.start:
            ref_seq += "N" * (self.end - self.start - len(ref_seq))

        # verify reference alleles
        for snp in self.snps:
            ref_n = len(snp.ref_allele)
            ref_snp = ref_seq[snp.seq_pos : snp.seq_pos + ref_n]
            if snp.ref_allele != ref_snp:
                print(
                    "ERROR: %s does not match reference %s" % (snp, ref_snp),
                    file=sys.stderr,
                )
                exit(1)

        # 1 hot code reference sequence
        ref_1hot = dna_io.dna_1hot(ref_seq)
        seqs1_list = [ref_1hot]

        # make alternative 1 hot coded sequences
        #  (assuming SNP is 1-based indexed)
        for snp in self.snps:
            alt_1hot = make_alt_1hot(
                ref_1hot, snp.seq_pos, snp.ref_allele, snp.alt_alleles[0]
            )
            seqs1_list.append(alt_1hot)

        return seqs1_list


class SNPWorker(Thread):
    """Compute summary statistics and write to HDF."""

    def __init__(self, snp_queue, sad_out, stats, log_pseudo=1):
        Thread.__init__(self)
        self.queue = snp_queue
        self.daemon = True
        self.sad_out = sad_out
        self.stats = stats
        self.log_pseudo = log_pseudo

    def run(self):
        while True:
            # unload predictions
            ref_preds, alt_preds, szi = self.queue.get()

            # sum across length
            ref_preds_sum = ref_preds.sum(axis=0, dtype="float64")
            alt_preds_sum = alt_preds.sum(axis=0, dtype="float64")

            # compare reference to alternative via mean subtraction
            if "SAD" in self.stats:
                sad = alt_preds_sum - ref_preds_sum
                self.sad_out["SAD"][szi, :] = sad.astype("float16")

            # compare reference to alternative via mean log division
            if "SAR" in self.stats:
                sar = np.log2(alt_preds_sum + self.log_pseudo) - np.log2(
                    ref_preds_sum + self.log_pseudo
                )
                self.sad_out["SAR"][szi, :] = sar.astype("float16")

            # compare geometric means
            if "geoSAD" in self.stats:
                sar_vec = np.log2(
                    alt_preds.astype("float64") + self.log_pseudo
                ) - np.log2(ref_preds.astype("float64") + self.log_pseudo)
                geo_sad = sar_vec.sum(axis=0)
                self.sad_out["geoSAD"][szi, :] = geo_sad.astype("float16")

            # communicate finished task
            self.queue.task_done()


def make_alt_1hot(ref_1hot, snp_seq_pos, ref_allele, alt_allele):
    """Return alternative allele one hot coding."""
    ref_n = len(ref_allele)
    alt_n = len(alt_allele)

    # copy reference
    alt_1hot = np.copy(ref_1hot)

    if alt_n == ref_n:
        # SNP
        dna_io.hot1_set(alt_1hot, snp_seq_pos, alt_allele)

    elif ref_n > alt_n:
        # deletion
        delete_len = ref_n - alt_n
        assert ref_allele[0] == alt_allele[0]
        dna_io.hot1_delete(alt_1hot, snp_seq_pos + 1, delete_len)

    else:
        # insertion
        assert ref_allele[0] == alt_allele[0]
        dna_io.hot1_insert(alt_1hot, snp_seq_pos + 1, alt_allele[1:])

    return alt_1hot


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
