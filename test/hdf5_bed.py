#!/usr/bin/env python
from optparse import OptionParser
import os

import h5py
import pysam

import basenji.dna_io

################################################################################
# hdf5_bed.py
#
# Checking that the BED regions output by basenji_hdf5.py match the one hot
# coded sequences in the HDF5.
################################################################################

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <hdf5_file> <bed_file>'
    parser = OptionParser(usage)
    parser.add_option('-f', dest='fasta_file', default='%s/assembly/hg19.fa'%os.environ['HG19'], help='FASTA file [Default: %default]')
    parser.add_option('-n', dest='check_num', default=100, type='int', help='Number of sequences to check [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 2:
        parser.error('Must provide HDF5 and BED files')
    else:
        hdf5_file = args[0]
        bed_file = args[1]

    fasta = pysam.Fastafile(options.fasta_file)
    hdf5_in = h5py.File(hdf5_file)

    si = 0
    for line in open(bed_file):
        a = line.split()
        if a[-1] == 'train':
            chrom = a[0]
            start = int(a[1])
            end = int(a[2])

            bed_seq = fasta.fetch(chrom, start, end).upper()
            hdf5_seq = basenji.dna_io.hot1_dna(hdf5_in['train_in'][si:si+1])[0]

            print(bed_seq[:10], len(bed_seq))
            assert(bed_seq == hdf5_seq)

        si += 1
        if si > options.check_num:
            break


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
