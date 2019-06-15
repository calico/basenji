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

import numpy as np
import pyBigWig

################################################################################
# bigwig_hdf5.py
#
#
################################################################################


################################################################################
# main
################################################################################
def main():
    usage = "usage: %prog [options] <wig_file> <seg_bed_file> <out_file>"
    parser = OptionParser(usage)
    parser.add_option(
        "-l",
        dest="seq_length",
        default=1024,
        type="int",
        help="Sequence length [Default: %default]",
    )
    parser.add_option(
        "-s",
        dest="stride",
        default=None,
        type="int",
        help="Stride to advance segments [Default: seq_length]",
    )
    parser.add_option(
        "-w",
        dest="pool_width",
        type="int",
        default=1,
        help="Average pooling width [Default: %default]",
    )
    (options, args) = parser.parse_args()

    if len(args) != 3:
        parser.error("")
    else:
        wig_file = args[0]
        seg_bed_file = args[1]
        out_file = args[2]

    # read segments
    segments = []
    for line in open(seg_bed_file):
        a = line.split()
        chrom = a[0]
        seg_start = int(a[1])
        seg_end = int(a[2])
        segments.append((chrom, seg_start, seg_end))

    # initialize target values
    targets = []

    # open wig
    wig_in = pyBigWig.open(wig_file)

    for chrom, seg_start, seg_end in segments:
        # read values
        try:
            seg_values = np.array(
                wig_in.values(chrom, seg_start, seg_end), dtype="float16"
            )
        except:
            print(
                "WARNING: %s doesn't see %s:%d-%d. Setting to all zeros."
                % (wig_file, chrom, seg_start, seg_end)
            )
            seg_values = np.zeros(seg_end - seg_start, dtype="float16")

        # set NaN's to zero
        seg_values = np.nan_to_num(seg_values)

        # break up into batchable sequences (as below in segments_1hot)
        bstart = 0
        bend = bstart + options.seq_length
        while bend <= len(seg_values):
            # extract (and pool)
            if options.pool_width == 1:
                sv = seg_values[bstart:bend]
            else:
                sv = (
                    seg_values[bstart:bend]
                    .reshape(
                        (options.seq_length // options.pool_width, options.pool_width)
                    )
                    .sum(axis=1)
                )

            # append
            targets.append(sv)

            # update
            bstart += options.stride
            bend += options.stride

    # save to file
    np.save(out_file, targets)


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
