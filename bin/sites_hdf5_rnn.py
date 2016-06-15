#!/usr/bin/env python
from optparse import OptionParser
from collections import OrderedDict
import random

import h5py
import numpy as np
import pyBigWig
import pysam

import basenji.io

################################################################################
# sites_hdf5_rnn.py
#
#
################################################################################


################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <fasta_file> <sites_bed_file> <sample_wigs_file> <out_hdf5_file>'
    parser = OptionParser(usage)
    parser.add_option('-b', dest='bin_size', default=1, type='int', help='Bin size to take the mean track value [Default: %default]')
    parser.add_option('-f', dest='function', default='mean', help='Function to compute in each bin [Default: %default]')
    parser.add_option('-s', dest='random_seed', default=1, type='int', help='numpy.random seed [Default: %default]')
    parser.add_option('-t', dest='test_num', default=0, type='int', help='Test % [Default: %default]')
    parser.add_option('-v', dest='valid_num', default=0, type='int', help='Validation % [Default: %default]')
    parser.add_option('--vt', dest='valid_test', default=False, action='store_true', help='Use validation as test, too [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 4:
        parser.error('Must provide FASTA file, sites BED, file with Wig/BigWig labels and paths, and output HDF5 file')
    else:
        fasta_file = args[0]
        sites_bed_file = args[1]
        sample_wigs_file = args[2]
        out_hdf5_file = args[3]

    #################################################################
    # setup
    #################################################################
    # read in sites
    sites = []
    for line in open(sites_bed_file):
        a = line.split()
        start = int(a[1])
        end = int(a[2])
        site_len = end - start
        sites.append(Site(a[0], start))
    num_sites = len(sites)

    # shuffle sites
    random.shuffle(sites)

    # get wig files and labels
    target_wigs = OrderedDict()
    for line in open(sample_wigs_file):
        a = line.split()
        target_wigs[a[0]] = a[1]
    num_targets = len(target_wigs)

    #################################################################
    # initialize train, valid, test HDF5
    #################################################################
    train_num = num_sites - options.valid_num - options.test_num

    # create datasets
    out_hdf5_open = h5py.File(out_hdf5_file, 'w')

    # input
    train_in = out_hdf5_open.create_dataset('train_in', (train_num,site_len,4), dtype='float16')
    if options.valid_num:
        valid_in = out_hdf5_open.create_dataset('valid_in', (options.valid_num,site_len,4), dtype='float16')
    if options.test_num > 0:
        test_in = out_hdf5_open.create_dataset('test_in', (options.test_num,site_len,4), dtype='float16')

    # output
    train_out = out_hdf5_open.create_dataset('train_out', (train_num,site_len,num_targets), dtype='float16')
    if options.valid_num:
        valid_out = out_hdf5_open.create_dataset('valid_out', (options.valid_num,site_len,num_targets), dtype='float16')
    if options.test_num:
        test_out = out_hdf5_open.create_dataset('test_out', (options.test_num,site_len,num_targets), dtype='float16')


    #################################################################
    # one hot code sites
    #################################################################
    fasta_in = pysam.FastaFile(fasta_file)

    for si in range(len(sites)):
        # choose dataset
        if si < train_num:
            ds_in = train_in
            ds_i = si
        elif si < train_num + options.valid_num:
            ds_in = valid_in
            ds_i = si - train_num
        else:
            ds_in = test_in
            ds_i = si - train_num - options.valid_num

        # one hot code sequence
        ds_in[ds_i,:,:] = basenji.io.dna_1hot(sites[ds_i].seq(fasta_in, site_len))

    fasta_in.close()


    #################################################################
    # process target wigs
    #################################################################
    wi = 0
    for sample in target_wigs:
        wig_file = target_wigs[sample]
        print(wig_file)

        # open wig
        wig_in = pyBigWig.open(wig_file)

        for si in range(len(sites)):
            s = sites[si]

            # pull stats from wig
            if options.bin_size == 1:
                site_stats = wig_in.values(s.chrom, s.start, s.start+site_len)
            else:
                # compute num bins and round up
                num_bins = int(np.ceil(site_len/options.bin_size))
                # extend end to match bins
                site_end = s.start + num_bins*options.bin_size

                site_stats = wig_in.stats(s.chrom, s.start, site_end, type=options.function, nBins=num_bins)

            # choose dataset
            if si < train_num:
                ds_out = train_out
                ds_i = si
            elif si < train_num + options.valid_num:
                ds_out = valid_out
                ds_i = si - train_num
            else:
                ds_out = test_out
                ds_i = si - train_num - options.valid_num

            # write to HDF5
            ds_out[ds_i,:,wi] = np.array(site_stats, dtype='float16')

        wi += 1


    #################################################################
    # finalize
    #################################################################
    out_hdf5_open.create_dataset('target_labels', data=target_wigs.keys())

    if options.valid_test:
        out_hdf5_open.create_dataset('test_in', data=valid_in)
        out_hdf5_open.create_dataset('test_out', data=valid_out)

    out_hdf5_open.close()


class Site:
    def __init__(self, chrom, start):
        self.chrom = chrom
        self.start = start

    def seq(self, fasta_in, site_len):
        return fasta_in.fetch(self.chrom, self.start, self.start + site_len)


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
