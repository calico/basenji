#!/usr/bin/env python
from optparse import OptionParser
import os

import numpy as np
import pysam
import pyBigWig
import tensorflow as tf

from basenji.dna_io import hot1_dna

'''
Name

Description...
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <tfr_dir> <out_bw>'
    parser = OptionParser(usage)
    parser.add_option('-f', dest='fasta_file',
            default='%s/assembly/hg38.fa' % os.environ['HG38'])
    parser.add_option('-g', dest='genome_file',
            default='%s/assembly/hg38.human.genome' % os.environ['HG38'])
    parser.add_option('-l', dest='target_length',
            default=1024, type='int',
            help='TFRecord target length [Default: %default]')
    parser.add_option('-s', dest='data_split', default='train')
    parser.add_option('-t', dest='target_i',
            default=0, type='int', help='Target index [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 2:
        parser.error('Must provide TF Records directory and output BigWig')
    else:
        tfr_dir = args[0]
        out_bw_file = args[1]

    # initialize output BigWig
    out_bw_open = pyBigWig.open(out_bw_file, 'w')

    # construct header
    header = []
    for line in open(options.genome_file):
        a = line.split()
        header.append((a[0], int(a[1])))

    # write header
    out_bw_open.addHeader(header)

    # initialize chr dictionary
    chr_values = {}
    for chrm, clen in header:
        chr_values[chrm] = np.zeros(clen, dtype='float16')

    # open sequences BED
    seq_bed_open = open('%s/../sequences0.bed' % tfr_dir)

    # open FASTA
    fasta_open = pysam.Fastafile(options.fasta_file)

    # initialize one shot iterator
    next_op = make_next_op('%s/%s-0-0.tfr' % (tfr_dir, options.data_split))

    # read sequence values
    with tf.Session() as sess:
      next_datum = sess.run(next_op)
      while next_datum:
        # read sequence
        seq_bed_line = seq_bed_open.readline()
        a = seq_bed_line.rstrip().split('\t')
        chrm = a[0]
        start = int(a[1])
        end = int(a[2])
        target_pool = (end - start) // options.target_length

        # check sequence
        seq_1hot = next_datum['sequence'].reshape((-1,4))
        seq_1hot_dna = hot1_dna(seq_1hot)
        seq_fasta = fasta_open.fetch(chrm, start, end).upper()
        assert(seq_1hot_dna == seq_fasta)

        # read targets
        targets = next_datum['target'].reshape(options.target_length, -1)
        targets_ti = targets[:,options.target_i]

        # set values
        chr_values[chrm][start:end] = np.repeat(targets_ti, target_pool)

        try:
          next_datum = sess.run(next_op)
        except tf.errors.OutOfRangeError:
          next_datum = False

    fasta_open.close()

    # write chr values
    for chrm, _ in header:
        print(chrm)

        out_bw_open.addEntries(chrm, 0, values=chr_values[chrm], span=1, step=1)

    # close files
    out_bw_open.close()


def make_next_op(tfr_pattern):
    # read TF Records
    dataset = tf.data.Dataset.list_files(tfr_pattern)

    def file_to_records(filename):
        return tf.data.TFRecordDataset(filename, compression_type='ZLIB')
    dataset = dataset.flat_map(file_to_records)

    dataset = dataset.batch(1)
    dataset = dataset.map(parse_proto)

    iterator = dataset.make_one_shot_iterator()
    try:
        next_op = iterator.get_next()
    except tf.errors.OutOfRangeError:
        print('TFRecord pattern %s is empty' % self.tfr_pattern, file=sys.stderr)
        exit(1)

    return next_op


def parse_proto(example_protos):
    features = {
        'genome': tf.FixedLenFeature([1], tf.int64),
        'sequence': tf.FixedLenFeature([], tf.string),
        'target': tf.FixedLenFeature([], tf.string)
    }
    parsed_features = tf.parse_example(example_protos, features=features)
    genome = parsed_features['genome']
    seq = tf.decode_raw(parsed_features['sequence'], tf.uint8)
    targets = tf.decode_raw(parsed_features['target'], tf.float16)
    return {'genome': genome, 'sequence': seq, 'target': targets}

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
