#!/usr/bin/env python
from optparse import OptionParser
import pdb
import pyBigWig

################################################################################
# unmappable_bed.py
#
# Produce a BED file describing unmappable regions below a certain threshold
# for a certain length.
################################################################################


################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <map_bw>'
    parser = OptionParser(usage)
    parser.add_option('-l', dest='unmap_len',
            default=500, type='int',
            help='Length above which an unmappable region will be output [Default: %default]')
    parser.add_option('-t', dest='unmap_t',
            default=0.2, type='float',
            help='Threshold below which an unmappable region will be considered [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 1:
        parser.error('Must provide mappability bigwig')
    else:
        map_bw = args[0]

    map_in = pyBigWig.open(map_bw)

    chrom_lens = map_in.chroms()
    for chrom in chrom_lens:
        # read mappability values
        map_vals = map_in.values(chrom, 0, chrom_lens[chrom])

        open_start = None
        for i in range(chrom_lens[chrom]):
            if map_vals[i] > options.unmap_t:
                # above threshold

                # close any open regions
                if open_start is None:
                    pass
                else:
                    if i - open_start >= options.unmap_len:
                        print('%s\t%d\t%d' % (chrom, open_start, i))

                open_start = None

            else:
                # below threshold

                # open/extend a region
                if open_start is None:
                    open_start = i

        # close any open regions
        i = chrom_lens[chrom]
        if open_start is None:
            if i - open_start >= options.unmap_len:
                print('%s\t%d\t%d' % (chrom, open_start, i))

    map_in.close()


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
