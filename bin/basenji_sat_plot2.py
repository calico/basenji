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
import pdb
import random
import sys

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns

import h5py
import numpy as np
import pandas as pd
from skbio.sequence import DNA
from skbio.alignment import global_pairwise_align_nucleotide

from basenji import dna_io
from basenji import plots
from basenji_sat_plot import delta_matrix, plot_heat

"""
basenji_sat_plot2.py
"""

################################################################################
# main
################################################################################
def main():
    usage = "usage: %prog [options] <scores1_file> <scores2_file>"
    parser = OptionParser(usage)
    parser.add_option(
        "-a",
        dest="activity_enrich",
        default=1,
        type="float",
        help="Enrich for the most active top % of sequences [Default: %default]",
    )
    parser.add_option(
        "-f",
        dest="figure_width",
        default=20,
        type="float",
        help="Figure width [Default: %default]",
    )
    parser.add_option(
        "-g",
        dest="gain",
        default=False,
        action="store_true",
        help="Draw a sequence logo for the gain score, too [Default: %default]",
    )
    parser.add_option(
        "-l",
        dest="plot_len",
        default=300,
        type="int",
        help="Length of centered sequence to mutate [Default: %default]",
    )
    parser.add_option(
        "-m",
        dest="min_limit",
        default=0.01,
        type="float",
        help="Minimum heatmap limit [Default: %default]",
    )
    parser.add_option(
        "-o",
        dest="out_dir",
        default="sat_plot2",
        help="Output directory [Default: %default]",
    )
    parser.add_option(
        "-r",
        dest="rng_seed",
        default=1,
        type="float",
        help="Random number generator seed [Default: %default]",
    )
    parser.add_option(
        "-s",
        dest="sample",
        default=None,
        type="int",
        help="Sample N sequences from the set [Default:%default]",
    )
    parser.add_option(
        "-t",
        dest="targets_file",
        default=None,
        type="str",
        help="File specifying target indexes and labels in table format",
    )
    (options, args) = parser.parse_args()

    if len(args) != 2:
        parser.error("Must provide two scores HDF5 file")
    else:
        scores1_h5_file = args[0]
        scores2_h5_file = args[1]

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    np.random.seed(options.rng_seed)

    # determine targets
    targets_df = pd.read_table(options.targets_file, index_col=0)
    num_targets = targets_df.shape[0]

    # open scores
    scores1_h5 = h5py.File(scores1_h5_file)
    scores2_h5 = h5py.File(scores2_h5_file)
    num_seqs = scores1_h5["seqs"].shape[0]
    mut_len = scores1_h5["scores"].shape[1]

    # determine plot region
    if options.plot_len > mut_len:
        print(
            "Scored mutations length %d is less than requested plot length %d"
            % (mut_len, options.plot_len),
            file=sys.stderr,
        )
        plot_start = 0
        plot_end = mut_len
    else:
        mut_mid = mut_len // 2
        plot_start = mut_mid - (options.plot_len // 2)
        plot_end = plot_start + options.plot_len

    # plot attributes
    sns.set(style="white", font_scale=1)

    # determine sequences
    seq_indexes = np.arange(num_seqs)

    if options.sample and options.sample < num_seqs:
        seq_indexes = np.random.choice(seq_indexes, size=options.sample, replace=False)

    for si in seq_indexes:
        # read sequences
        seq1_1hot = scores1_h5["seqs"][si, plot_start:plot_end]
        seq2_1hot = scores2_h5["seqs"][si, plot_start:plot_end]

        # align sequences
        seq1_align, seq2_align = global_align(seq1_1hot, seq2_1hot)
        align_len = len(seq1_align)

        # QC alignment
        aligned_nt = 2 * options.plot_len - align_len
        align_rate = aligned_nt / options.plot_len
        if aligned_nt / options.plot_len < 0.25:
            print(
                "WARNING: skipping sequence %d due to poor alignment %.3f"
                % (si, align_rate),
                file=sys.stderr,
            )
        else:
            # read scores
            scores1 = scores1_h5["scores"][si, plot_start:plot_end, :, :]
            scores2 = scores2_h5["scores"][si, plot_start:plot_end, :, :]

            # expand scores alignment
            esa = expand_scores_align(
                scores1, scores2, seq1_1hot, seq2_1hot, seq1_align, seq2_align
            )
            ascores1, ascores2, ascores1_ref, ascores2_ref = esa

            # for each target
            for tii in range(num_targets):
                ti = targets_df["index"].iloc[tii]

                # slice target
                scores1_ti = ascores1[:, :, ti]
                scores2_ti = ascores2[:, :, ti]

                # compute scores relative to reference
                delta1_ti = scores1_ti - ascores1_ref[:, [ti]]
                delta2_ti = scores2_ti - ascores2_ref[:, [ti]]

                # setup plot
                axes_list = setup_plot(options.figure_width, options.plot_len)
                xi = 0

                # for each sequence
                delta_aligns = [(delta1_ti, seq1_align), (delta2_ti, seq2_align)]
                for delta_ti, seq_align in delta_aligns:

                    # compute loss and gain
                    delta_loss = delta_ti.min(axis=1)
                    delta_gain = delta_ti.max(axis=1)

                    # plot sequence logo
                    plot_seqlogo(axes_list[xi], seq_align, -delta_loss)
                    xi += 1
                    # if options.gain:
                    #   plot_seqlogo(axes_list[xi], seq_align, delta_gain)

                    # plot heat map
                    plot_heat(axes_list[xi], delta_ti.T, options.min_limit)
                    xi += 1

                # finish plot
                plt.tight_layout()
                plt.savefig("%s/seq%d_t%d.pdf" % (options.out_dir, si, ti), dpi=600)
                plt.close()


def expand_4l(sat_lg_ti, seq_align):
    """ Expand a sat mut score array to another dimension
       representing nucleotides.

    In:
        sat_lg_ti (l array): Sat mut loss/gain scores for a single
                             sequence and target.
        seq_align (l array): Sequence nucleotides, with gaps.

    Out:
        sat_loss_4l (lx4 array): Score-hot coding?

    """

    # helper variables
    satmut_len = len(sat_lg_ti)
    align_len = len(seq_align)

    # initialize score-hot coding
    sat_lg_4l = np.zeros((satmut_len, 4))

    # set score-hot coding using aligned sequence
    for li in range(align_len):
        if seq_align[li] == "A":
            sat_lg_4l[li, 0] = sat_lg_ti[li]
        elif seq_align[li] == "C":
            sat_lg_4l[li, 1] = sat_lg_ti[li]
        elif seq_align[li] == "G":
            sat_lg_4l[li, 2] = sat_lg_ti[li]
        elif seq_align[li] == "T":
            sat_lg_4l[li, 3] = sat_lg_ti[li]

    return sat_lg_4l


def expand_scores_align(scores1, scores2, seq1_1hot, seq2_1hot, seq1_align, seq2_align):
    """Expand two scores arrays according to a sequence alignment with NaNs in gaps."""
    # reference scores
    scores1_ref = scores1[seq1_1hot]
    scores2_ref = scores2[seq2_1hot]

    # initialize new score arrays
    align_len = len(seq1_align)
    scores1_align = np.zeros((align_len, 4, scores1.shape[-1]))
    scores2_align = np.zeros((align_len, 4, scores2.shape[-1]))
    scores1_align_ref = np.zeros((align_len, scores1.shape[-1]))
    scores2_align_ref = np.zeros((align_len, scores2.shape[-1]))

    # expand to alignment
    ci1 = 0
    ci2 = 0
    for ai in range(align_len):
        # sequence 1
        if seq1_align[ai] == "-":
            scores1_align[ai, :, :] = np.nan
            scores1_align_ref[ai, :] = np.nan
        else:
            scores1_align[ai, :, :] = scores1[ci1, :, :]
            scores1_align_ref[ai, :] = scores1_ref[ci1, :]
            ci1 += 1

        # sequence 2
        if seq2_align[ai] == "-":
            scores2_align[ai, :, :] = np.nan
            scores2_align_ref[ai, :] = np.nan
        else:
            scores2_align[ai, :, :] = scores2[ci2, :, :]
            scores2_align_ref[ai, :] = scores2_ref[ci2, :]
            ci2 += 1

    return scores1_align, scores2_align, scores1_align_ref, scores2_align_ref


def global_align(seq1_1hot, seq2_1hot):
    """Align two 1-hot encoded sequences."""

    align_opts = {
        "gap_open_penalty": 10,
        "gap_extend_penalty": 1,
        "match_score": 5,
        "mismatch_score": -4,
    }

    seq1_dna = DNA(dna_io.hot1_dna(seq1_1hot))
    seq2_dna = DNA(dna_io.hot1_dna(seq2_1hot))
    # seq_align = global_pairwise_align_nucleotide(seq1_dna, seq2_dna, *align_opts)[0]
    seq_align = global_pairwise_align_nucleotide(
        seq1_dna,
        seq2_dna,
        gap_open_penalty=10,
        gap_extend_penalty=1,
        match_score=5,
        mismatch_score=-4,
    )[0]
    seq1_align = str(seq_align[0])
    seq2_align = str(seq_align[1])
    return seq1_align, seq2_align


def plot_seqlogo(ax, seq_align, sat_score_ti, pseudo_pct=0.05):
    """ Plot a sequence logo for the loss/gain scores.

    Args:
        ax (Axis): matplotlib axis to plot to.
        seq_align (L array): Sequence nucleotides, with gaps.
        sat_score_ti (L_sm array): Minimum mutation delta across satmut length.
        pseudo_pct (float): % of the max to add as a pseudocount.
    """

    satmut_len = len(sat_score_ti)

    # add pseudocounts
    sat_score_ti += pseudo_pct * np.nanmax(sat_score_ti)

    # expand
    sat_score_4l = expand_4l(sat_score_ti, seq_align)

    plots.seqlogo(sat_score_4l, ax)


def setup_plot(figure_width, plot_len):
    spp = subplot_params(plot_len)

    plt.figure(figsize=(figure_width, 6))

    axes_list = []
    axes_list.append(
        plt.subplot2grid(
            (4, spp["heat_cols"]), (0, spp["logo_start"]), colspan=spp["logo_span"]
        )
    )
    axes_list.append(
        plt.subplot2grid((4, spp["heat_cols"]), (1, 0), colspan=spp["heat_cols"])
    )
    axes_list.append(
        plt.subplot2grid(
            (4, spp["heat_cols"]), (2, spp["logo_start"]), colspan=spp["logo_span"]
        )
    )
    axes_list.append(
        plt.subplot2grid((4, spp["heat_cols"]), (3, 0), colspan=spp["heat_cols"])
    )

    return axes_list


def subplot_params(seq_len):
    """ Specify subplot layout parameters for various sequence lengths. """
    if seq_len < 500:
        spp = {
            "heat_cols": 400,
            "sad_start": 1,
            "sad_span": 321,
            "logo_start": 0,
            "logo_span": 323,
        }
    else:
        spp = {
            "heat_cols": 400,
            "sad_start": 1,
            "sad_span": 320,
            "logo_start": 0,
            "logo_span": 322,
        }

    return spp


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
