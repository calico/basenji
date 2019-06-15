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
from collections import OrderedDict
import multiprocessing
import sys
import time

import h5py
import joblib
import numpy as np
import pyBigWig
import pysam
from sklearn import preprocessing
from sklearn.decomposition import PCA
import tensorflow as tf

import basenji

################################################################################
# basenji_scent.py
#
# Train an autoencoder to project the full functional profiles defined by a set
# of Bigwig files into a lower dimension latent space that simultaneously
# smooths the signal using cross-dataset correlations and compresses the space
# required to store it. Win-win.
################################################################################


################################################################################
# main
################################################################################
def main():
    usage = (
        "usage: %prog [options] <genome_file> <sample_wigs_file> " "<model_out_file>"
    )
    parser = OptionParser(usage)
    parser.add_option(
        "-g", dest="gaps_file", help="Genome assembly gaps BED [Default: %default]"
    )
    parser.add_option(
        "-l", dest="load_targets_file", help="Load the sampled target set from disk"
    )
    parser.add_option("-m", dest="params_file", help="Model parameters")
    parser.add_option(
        "-p",
        dest="processes",
        default=1,
        type="int",
        help="Number parallel processes to load data [Default: %default]",
    )
    parser.add_option(
        "-n",
        dest="num_samples",
        type="int",
        default=1000,
        help="Genomic positions to sample for training data [Default: %default]",
    )
    parser.add_option(
        "-r",
        dest="reconstruct_out_pre",
        help="Save the recontructed validation targets to disk",
    )
    parser.add_option(
        "-s", dest="save_targets_file", help="Save the sampled target set to disk"
    )
    parser.add_option(
        "-v",
        dest="valid_pct",
        type="float",
        default=0.1,
        help="Proportion of the data for validation [Default: %default]",
    )
    parser.add_option(
        "-w",
        dest="whiten",
        default=False,
        action="store_true",
        help="Whiten functional data [Default: %default]",
    )
    (options, args) = parser.parse_args()

    if len(args) != 3:
        parser.error(
            "Must provide genome file, sample Wig/BigWig labels and paths, and "
            "model output file"
        )
    else:
        genome_file = args[0]
        sample_wigs_file = args[1]
        model_out_file = args[2]

    if options.load_targets_file is not None:
        targets = np.load(options.load_targets_file)

    else:
        #######################################################
        # sample genome
        #######################################################
        chrom_segments = basenji.genome.load_chromosomes(genome_file)
        if options.gaps_file:
            chrom_segments = basenji.genome.split_contigs(
                chrom_segments, options.gaps_file
            )

        # determine how frequently to sample
        genome_sum = 0
        for chrom in chrom_segments:
            for seg_start, seg_end in chrom_segments[chrom]:
                genome_sum += seg_end - seg_start

        sample_every = genome_sum // options.num_samples

        # sample positions
        chrom_samples = {}
        for chrom in chrom_segments:
            chrom_samples[chrom] = []
            sample_counter = 0
            for seg_start, seg_end in chrom_segments[chrom]:
                seg_i = seg_start
                while seg_i + (sample_every - sample_counter) < seg_end:
                    seg_i += sample_every - sample_counter
                    chrom_samples[chrom].append(seg_i)
                    sample_counter = 0

                sample_counter += seg_end - seg_i

        #######################################################
        # read from bigwigs
        #######################################################
        # get wig files and labels
        target_wigs = OrderedDict()
        for line in open(sample_wigs_file):
            a = line.split()
            target_wigs[a[0]] = a[1]

        print("Loading from BigWigs")
        sys.stdout.flush()
        t0 = time.time()

        pool = multiprocessing.Pool(options.processes)
        targets_t = pool.starmap(
            bigwig_read,
            [(wig_file, chrom_samples) for wig_file in target_wigs.values()],
        )

        # convert and transpose
        targets = np.array(targets_t).T

        # shuffle
        np.random.shuffle(targets)

        if options.save_targets_file is not None:
            np.save(options.save_targets_file, targets)

        print("%ds" % (time.time() - t0))

    print("\nSampled dataset", targets.shape, "\n")

    # pre-process
    targets = np.nan_to_num(targets)
    if options.whiten:
        targets = preprocessing.scale(targets)

    # divide train and valid
    tv_line = int(options.valid_pct * targets.shape[0])

    #######################################################
    # model parameters and placeholders
    #######################################################
    # read parameters
    job = basenji.dna_io.read_job_params(options.params_file)

    job["num_targets"] = targets.shape[1]

    # construct model
    print("Constructing model")
    sys.stdout.flush()

    if job.get("model", "autoencoder") == "pca":
        # construct
        model = PCA(n_components=job["latent_dim"])

        # train
        model.fit(targets[tv_line:])

        # validate
        latent_valid = model.transform(targets[:tv_line])
        recon_valid = model.inverse_transform(latent_valid)
        valid_var = targets[:tv_line].var()
        recon_var = (targets[:tv_line] - recon_valid).var()
        r2 = 1.0 - np.divide(recon_var, valid_var)
        print("Valid R2: %7.5f" % r2.mean())

        # save
        joblib.dump(model, model_out_file)

        if options.reconstruct_out_pre:
            np.save(options.reconstruct_out_pre, recon_valid)

    else:
        model = basenji.autoencoder.AE(job)

        #######################################################
        # train
        #######################################################
        # initialize batcher
        batcher_train = basenji.batcher.BatcherT(
            targets[tv_line:], model.batch_size, shuffle=True
        )
        batcher_valid = basenji.batcher.BatcherT(targets[:tv_line], model.batch_size)

        # checkpoints
        saver = tf.train.Saver()

        with tf.Session() as sess:
            t0 = time.time()

            # initialize variables
            sess.run(tf.initialize_all_variables())

            train_loss = None
            best_r2 = -1000
            early_stop_i = 0

            for epoch in range(1000):
                if early_stop_i < model.early_stop:
                    t0 = time.time()

                    # save previous
                    train_loss_last = train_loss

                    # train
                    train_loss = model.train_epoch(sess, batcher_train)

                    # validate
                    valid_loss, valid_r2 = model.test(sess, batcher_valid)

                    best_str = ""
                    if valid_r2 > best_r2:
                        best_r2 = valid_r2
                        best_str = "best!"
                        early_stop_i = 0
                        saver.save(sess, model_out_file)
                    else:
                        early_stop_i += 1

                    # measure time
                    et = time.time() - t0
                    if et < 600:
                        time_str = "%3ds" % et
                    elif et < 6000:
                        time_str = "%3dm" % (et / 60)
                    else:
                        time_str = "%3.1fh" % (et / 3600)

                    # print update
                    print(
                        "Epoch %3d: Train loss: %7.5f, Valid loss: %7.5f, Valid R2: "
                        "%7.5f, Time: %s %s"
                        % (
                            epoch + 1,
                            train_loss,
                            valid_loss,
                            valid_r2,
                            time_str,
                            best_str,
                        )
                    )
                    sys.stdout.flush()

            if options.reconstruct_out_pre:
                preds = model.predict(sess, batcher_valid)
                np.save(options.reconstruct_out_pre, preds)


def bigwig_read(wig_file, chrom_samples):
    print("  %s" % wig_file)
    sys.stdout.flush()

    # initialize target values
    targets = []

    # open wig
    wig_in = pyBigWig.open(wig_file)

    # read position values
    for chrom in chrom_samples:
        for pos in chrom_samples[chrom]:
            try:
                pos_val = wig_in.values(chrom, pos, pos + 1)[0]
            except:
                print(chrom, pos)
                exit(1)
            targets.append(pos_val)

    return targets


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
