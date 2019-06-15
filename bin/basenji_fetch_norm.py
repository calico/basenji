#!/usr/bin/env python
from optparse import OptionParser
import glob
import os
import pdb
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import h5py

from basenji.sad5 import ChrSAD5

"""
basenji_fetch_norm.py

Fit Cauchy distribution parameters across chromosomes.
"""

################################################################################
# main
################################################################################
def main():
    usage = "usage: %prog [options] <sad_h5_path> <vcf_file>"
    parser = OptionParser(usage)
    parser.add_option(
        "-s",
        dest="sample",
        default=131072,
        type="int",
        help="Sampled SNPs to fit distribution [Default: %default]",
    )
    (options, args) = parser.parse_args()

    if len(args) != 1:
        parser.error("Must provide SAD HDF5 path.")
    else:
        sad_h5_path = args[0]

    # index SNPs
    csad5 = ChrSAD5(sad_h5_path, index_chr=True, compute_norm=False)

    # fit Cauchy
    csad5.fit_cauchy(options.sample)

    # normalize
    csad5.norm_cauchy()


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
