#!/usr/bin/env python
from optparse import OptionParser

import os
import pdb
import shutil
import subprocess
import unittest

import h5py
import numpy as np


class TestSatMut(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params_file = "data/params.txt"
        cls.model_file = "data/model_best.tf"
        cls.targets_file = "data/targets_sat.txt"
        cls.bed_file = "data/cage_peaks.bed"

    def test_bed(self):
        mut_len = 10
        out_dir = "sat_out"

        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)

        cmd = "basenji_sat_bed.py -l %d -o %s --rc -t %s %s %s %s" % (
            mut_len,
            out_dir,
            self.targets_file,
            self.params_file,
            self.model_file,
            self.bed_file,
        )
        subprocess.call(cmd, shell=True)

        scores_h5 = h5py.File("%s/scores.h5" % out_dir)
        scores = np.array(scores_h5["scores"])

        # check sequences
        bed_in = open(self.bed_file)
        num_seqs = len(bed_in.readlines())
        bed_in.close()
        self.assertEqual(scores.shape[0], num_seqs)

        # check mutagenesis length
        self.assertEqual(scores.shape[1], mut_len)

        # check targets
        targets_in = open(self.targets_file)
        num_targets = len(targets_in.readlines()) - 1
        targets_in.close()
        self.assertEqual(scores.shape[-1], num_targets)

        # check non-zero
        scores_sum = np.abs(scores).sum(dtype="float64")
        self.assertGreater(scores_sum, 1e-6)


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    unittest.main()
