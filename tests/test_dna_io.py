#!/usr/bin/env python
from optparse import OptionParser

import unittest

import numpy as np

import basenji.dna_io

class TestDelete(unittest.TestCase):
    def test_insert(self):
        seq = 'GATTACA'
        seq_1hot = basenji.dna_io.dna_1hot(seq)

        basenji.dna_io.hot1_delete(seq_1hot, 3, 2)

        self.assertEqual('GATCANN', basenji.dna_io.hot1_dna(seq_1hot))


class TestInsert(unittest.TestCase):
    def test_insert(self):
        seq = 'GATTACA'
        seq_1hot = basenji.dna_io.dna_1hot(seq)

        basenji.dna_io.hot1_insert(seq_1hot, 3, 'AG')

        self.assertEqual('GATAGTA', basenji.dna_io.hot1_dna(seq_1hot))


class TestRC(unittest.TestCase):
    def test_rc(self):
        #########################################
        # construct sequences
        seq1 = 'GATTACA'
        seq1_1hot = basenji.dna_io.dna_1hot(seq1)

        seq2 = 'TAGATAC'
        seq2_1hot = basenji.dna_io.dna_1hot(seq2)

        seqs_1hot = np.array([seq1_1hot,seq2_1hot])

        #########################################
        # reverse complement
        seqs_1hot_rc = basenji.dna_io.hot1_rc(seqs_1hot)

        seq1_rc = basenji.dna_io.hot1_dna(seqs_1hot_rc[0])
        seq2_rc = basenji.dna_io.hot1_dna(seqs_1hot_rc[1])

        #########################################
        # compare
        self.assertEqual('TGTAATC', seq1_rc)
        self.assertEqual('GTATCTA', seq2_rc)

        #########################################
        # reverse complement again
        seqs_1hot_rcrc = basenji.dna_io.hot1_rc(seqs_1hot_rc)

        seq1_rcrc = basenji.dna_io.hot1_dna(seqs_1hot_rcrc[0])
        seq2_rcrc = basenji.dna_io.hot1_dna(seqs_1hot_rcrc[1])

        #########################################
        # compare
        self.assertEqual(seq1, seq1_rcrc)
        self.assertEqual(seq2, seq2_rcrc)


class TestAugment(unittest.TestCase):
    def test_augment(self):
        seq = 'GATTACA'
        seq1 = basenji.dna_io.dna_1hot(seq)
        seqs1 = np.array([seq1])

        # forward, shift 0
        aseqs1_fwd0 = basenji.dna_io.hot1_augment(seqs1, True, 0)
        aseq_fwd0 = basenji.dna_io.hot1_dna(aseqs1_fwd0)[0]
        self.assertEqual('GATTACA', aseq_fwd0)

        # reverse, shift 0
        aseqs1_rc0 = basenji.dna_io.hot1_augment(seqs1, False, 0)
        aseq_rc0 = basenji.dna_io.hot1_dna(aseqs1_rc0)[0]
        self.assertEqual('TGTAATC', aseq_rc0)

        # forward, shift 1
        aseqs1_fwd1 = basenji.dna_io.hot1_augment(seqs1, True, 1)
        aseq_fwd1 = basenji.dna_io.hot1_dna(aseqs1_fwd1)[0]
        self.assertEqual('NGATTAC', aseq_fwd1)

        # reverse, shift 1
        aseqs1_rc1 = basenji.dna_io.hot1_augment(seqs1, False, 1)
        aseq_rc1 = basenji.dna_io.hot1_dna(aseqs1_rc1)[0]
        self.assertEqual('GTAATCN', aseq_rc1)

        # forward, shift 1
        aseqs1_fwd_m1 = basenji.dna_io.hot1_augment(seqs1, True, -1)
        aseq_fwd_m1 = basenji.dna_io.hot1_dna(aseqs1_fwd_m1)[0]
        self.assertEqual('ATTACAN', aseq_fwd_m1)

        # reverse, shift 1
        aseqs1_rc_m1 = basenji.dna_io.hot1_augment(seqs1, False, -1)
        aseq_rc_m1 = basenji.dna_io.hot1_dna(aseqs1_rc_m1)[0]
        self.assertEqual('NTGTAAT', aseq_rc_m1)


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    unittest.main()
