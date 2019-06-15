#!/usr/bin/env python
# Copyright 2017 Calico LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from optparse import OptionParser
import gc
import os
import pdb
import sys
import time

import h5py
import numpy as np
import pyBigWig
from scipy.stats import ttest_1samp
import tensorflow as tf

import basenji
from basenji_map import score_write

"""
basenji_map_seqs.py

Visualize a sequence's prediction's gradients as a map of influence across
the genomic region.
"""

################################################################################
# main
################################################################################
def main():
    usage = "usage: %prog [options] <params_file> <model_file> <hdf5_file> <bed_file>"
    parser = OptionParser(usage)
    parser.add_option(
        "-g",
        dest="genome_file",
        default="%s/data/human.hg19.genome" % os.environ["BASENJIDIR"],
        help="Chromosome lengths file [Default: %default]",
    )
    parser.add_option(
        "-l", dest="gene_list", help="Process only gene ids in the given file"
    )
    parser.add_option(
        "--mc",
        dest="mc_n",
        default=0,
        type="int",
        help="Monte carlo test iterations [Default: %default]",
    )
    parser.add_option(
        "-n",
        dest="norm",
        default=None,
        type="int",
        help="Compute saliency norm [Default% default]",
    )
    parser.add_option(
        "-o",
        dest="out_dir",
        default="grad_map",
        help="Output directory [Default: %default]",
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
        "-t", dest="target_indexes", default=None, help="Target indexes to plot"
    )

    if len(args) != 4:
        parser.error("Must provide parameters, model, and genomic position")
    else:
        params_file = args[0]
        model_file = args[1]
        hdf5_file = args[2]
        bed_file = args[3]

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    options.shifts = [int(shift) for shift in options.shifts.split(",")]

    #######################################################
    # load data

    data_open = h5py.File(hdf5_file)
    test_seqs = data_open["test_in"]

    # extract sequence chrom and start
    seqs_chrom = []
    seqs_pos = []
    for line in open(bed_file):
        a = line.split()
        if a[3] == "test":
            seqs_chrom.append(a[0])
            seqs_pos.append(int(a[1]))

    #######################################################
    # model parameters and placeholders

    job = basenji.dna_io.read_job_params(params_file)

    job["seq_length"] = test_seqs.shape[1]
    job["seq_depth"] = test_seqs.shape[2]
    job["target_pool"] = int(np.array(data_open.get("pool_width", 1)))

    if "num_targets" not in job:
        print(
            "Must specify number of targets (num_targets) in the parameters file.",
            file=sys.stderr,
        )
        exit(1)

    # set target indexes
    if options.target_indexes is not None:
        options.target_indexes = [int(ti) for ti in options.target_indexes.split(",")]
        target_subset = options.target_indexes
    else:
        options.target_indexes = list(range(job["num_targets"]))
        target_subset = None

    # build model
    model = basenji.seqnn.SeqNN()
    model.build(job, target_subset=target_subset)

    # determine latest pre-dilated layer
    dilated_mask = np.array(model.cnn_dilation) > 1
    dilated_indexes = np.where(dilated_mask)[0]
    pre_dilated_layer = np.min(dilated_indexes)
    print("Pre-dilated layer: %d" % pre_dilated_layer)

    # build gradients ops
    t0 = time.time()
    print("Building target/position-specific gradient ops.", end="")
    model.build_grads(layers=[pre_dilated_layer])
    print(" Done in %ds" % (time.time() - t0), flush=True)

    #######################################################
    # acquire gradients

    # initialize saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # load variables into session
        saver.restore(sess, model_file)

        # score sequences and write bigwigs
        score_write(sess, model, options, test_seqs, seqs_chrom, seqs_pos)


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
