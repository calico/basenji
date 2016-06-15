#!/usr/bin/env python
from optparse import OptionParser
import gzip
import os
import re
import subprocess
import sys

import h5py
import numpy as np

################################################################################
# merge_sites.py
#
# Preprocess a set of BED files for NN analysis, potentially adding them to an
# existing database of sites, specified as a BED file with the target activities
# comma-separated in column 4 and a full activity table file.
################################################################################

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <target_beds_file>'
    parser = OptionParser(usage)
    parser.add_option('-a', dest='db_act_file', default=None, help='Existing database activity table.')
    parser.add_option('-b', dest='db_bed', default=None, help='Existing database BED file.')
    parser.add_option('-c', dest='chrom_lengths_file', help='Table of chromosome lengths')
    parser.add_option('-i', dest='ignore_auxiliary', default=False, action='store_true', help='Ignore auxiliary chromosomes that don\'t match "chr\d+ or chrX" [Default: %default]')
    parser.add_option('-m', dest='merge_overlap', default=200, type='int', help='Overlap length (after extension to site_size) above which to merge sites [Default: %default]')
    parser.add_option('-n', dest='no_db_activity', default=False, action='store_true', help='Do not pass along the activities of the database sequences [Default: %default]')
    parser.add_option('-o', dest='out_prefix', default='merged', help='Output file prefix [Default: %default]')
    parser.add_option('-s', dest='site_size', default=600, type='int', help='Extend sites to this size [Default: %default]')
    parser.add_option('-y', dest='ignore_y', default=False, action='store_true', help='Ignore Y chromsosome sites [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 1:
    	parser.error('Must provide file labeling the targets and providing BED file paths.')
    else:
    	target_beds_file = args[0]

    # determine whether we'll add to an existing DB
    db_targets = []
    db_add = False
    if options.db_bed is not None:
        db_add = True
        if not options.no_db_activity:
            if options.db_act_file is None:
                parser.error('Must provide both activity table or specify -n if you want to add to an existing database')
            else:
                # read db target names
                db_act_in = open(options.db_act_file)
                db_targets = db_act_in.readline().strip().split('\t')
                db_act_in.close()

    # read in targets and assign them indexes into the db
    target_beds = []
    target_dbi = []
    for line in open(target_beds_file):
    	a = line.split()
        if len(a) != 2:
            print a
            print >> sys.stderr, 'Each row of the target BEDS file must contain a label and BED file separated by whitespace'
            exit(1)
    	target_dbi.append(len(db_targets))
    	db_targets.append(a[0])
    	target_beds.append(a[1])

    # read in chromosome lengths
    chrom_lengths = {}
    if options.chrom_lengths_file:
        chrom_lengths = {}
        for line in open(options.chrom_lengths_file):
            a = line.split()
            chrom_lengths[a[0]] = int(a[1])
    else:
        print >> sys.stderr, 'Warning: chromosome lengths not provided, so regions near ends may be incorrect.'

    #################################################################
    # print peaks to chromosome-specific files
    #################################################################
    chrom_files = {}
    chrom_outs = {}

    peak_beds = target_beds
    if db_add:
        peak_beds.append(options.db_bed)

    for bi in range(len(peak_beds)):
        if peak_beds[bi][-3:] == '.gz':
            peak_bed_in = gzip.open(peak_beds[bi])
        else:
            peak_bed_in = open(peak_beds[bi])

        for line in peak_bed_in:
            a = line.split('\t')
            a[-1] = a[-1].rstrip()

            # hash by chrom/strand
            chrom = a[0]
            strand = '+'
            if len(a) > 5 and a[5] in '+-':
                strand = a[5]
            chrom_key = (chrom,strand)

            # adjust coordinates to midpoint
            start = int(a[1])
            end = int(a[2])
            mid = find_midpoint(start, end)
            a[1] = str(mid)
            a[2] = str(mid + 1)

            # open chromosome file
            if chrom_key not in chrom_outs:
                chrom_files[chrom_key] = '%s_%s_%s.bed' % (options.out_prefix, chrom, strand)
                chrom_outs[chrom_key] = open(chrom_files[chrom_key], 'w')

            # if it's the db bed
            if db_add and bi == len(peak_beds)-1:
                if options.no_db_activity:
                    # set activity to null
                    a[6] = '.'
                    print >> chrom_outs[chrom_key], '\t'.join(a[:7])
                else:
                    print >> chrom_outs[chrom_key], line,

            # if it's a new bed
            else:
                # specify the target index
                while len(a) < 7:
                    a.append('')
                a[5] = strand
                a[6] = str(target_dbi[bi])
                print >> chrom_outs[chrom_key], '\t'.join(a[:7])

        peak_bed_in.close()

    # close chromosome-specific files
    for chrom_key in chrom_outs:
        chrom_outs[chrom_key].close()

    # ignore Y
    if options.ignore_y:
        for orient in '+-':
            chrom_key = ('chrY',orient)
            if chrom_key in chrom_files:
                print >> sys.stderr, 'Ignoring chrY %s' % orient
                os.remove(chrom_files[chrom_key])
                del chrom_files[chrom_key]

    # ignore auxiliary
    if options.ignore_auxiliary:
        primary_re = re.compile('chr\d+$')
        for chrom_key in chrom_files.keys():
            chrom,strand = chrom_key
            primary_m = primary_re.match(chrom)
            if not primary_m and chrom != 'chrX':
                print >> sys.stderr, 'Ignoring %s %s' % (chrom,strand)
                os.remove(chrom_files[chrom_key])
                del chrom_files[chrom_key]


    #################################################################
    # sort chromosome-specific files
    #################################################################
    for chrom_key in chrom_files:
        chrom,strand = chrom_key
        chrom_sbed = '%s_%s_%s_sort.bed' % (options.out_prefix,chrom,strand)
        sort_cmd = 'bedtools sort -i %s > %s' % (chrom_files[chrom_key], chrom_sbed)
        subprocess.call(sort_cmd, shell=True)
        os.remove(chrom_files[chrom_key])
        chrom_files[chrom_key] = chrom_sbed


    #################################################################
    # parse chromosome-specific files
    #################################################################
    final_bed_out = open('%s.bed' % options.out_prefix, 'w')

    for chrom_key in chrom_files:
        chrom, strand = chrom_key

        open_peaks = []
        for line in open(chrom_files[chrom_key]):
            a = line.split('\t')
            a[-1] = a[-1].rstrip()

            # construct Peak
            peak_start = int(a[1])
            peak_end = int(a[2])
            peak_act = activity_set(a[6])
            peak = Peak(peak_start, peak_end, peak_act)
            peak.extend(options.site_size, chrom_lengths.get(chrom,None))

            if len(open_peaks) == 0:
                # initialize open peak
                open_end = peak.end
                open_peaks = [peak]

            else:
                # operate on exiting open peak

                # if beyond existing open peak
                if open_end - options.merge_overlap <= peak.start:
                    # close open peak
                    mpeaks = merge_peaks(open_peaks, options.site_size, options.merge_overlap, chrom_lengths.get(chrom,None))

                    # print to file
                    for mpeak in mpeaks:
                        print >> final_bed_out, mpeak.bed_str(chrom, strand)

                    # initialize open peak
                    open_end = peak.end
                    open_peaks = [peak]

                else:
                    # extend open peak
                    open_peaks.append(peak)
                    open_end = max(open_end, peak.end)

        if len(open_peaks) > 0:
            # close open peak
            mpeaks = merge_peaks(open_peaks, options.site_size, options.merge_overlap, chrom_lengths.get(chrom,None))

            # print to file
            for mpeak in mpeaks:
                print >> final_bed_out, mpeak.bed_str(chrom, strand)

    final_bed_out.close()

    # clean
    for chrom_key in chrom_files:
        os.remove(chrom_files[chrom_key])


    #################################################################
    # construct/update activity table
    #################################################################
    final_act_out = open('%s_act.txt' % options.out_prefix, 'w')

    # print header
    cols = [''] + db_targets
    print >> final_act_out, '\t'.join(cols)

    # print sequences
    for line in open('%s.bed' % options.out_prefix):
        a = line.rstrip().split('\t')
        # index peak
        peak_id = '%s:%s-%s(%s)' % (a[0], a[1], a[2], a[5])

        # construct full activity vector
        peak_act = [0]*len(db_targets)
        for ai in a[6].split(','):
            if ai != '.':
                peak_act[int(ai)] = 1

        # print line
        cols = [peak_id] + peak_act
        print >> final_act_out, '\t'.join([str(c) for c in cols])

    final_act_out.close()


def activity_set(act_cs):
    ''' Return a set of ints from a comma-separated list of int strings.

    Attributes:
        act_cs (str) : comma-separated list of int strings

    Returns:
        set (int) : int's in the original string
    '''
    ai_strs = [ai for ai in act_cs.split(',')]

    if ai_strs[-1] == '':
        ai_strs = ai_strs[:-1]

    if ai_strs[0] == '.':
        aset = set()
    else:
        aset = set([int(ai) for ai in ai_strs])

    return aset


def find_midpoint(start, end):
    ''' Find the midpoint coordinate between start and end '''
    mid = (start + end)/2
    return mid


def merge_peaks(peaks, peak_size, merge_overlap, chrom_len):
    ''' Merge and the list of Peaks.

    Repeatedly find the closest adjacent peaks and consider
    merging them together, until there are no more peaks
    we want to merge.

    Attributes:
        peaks (list[Peak]) : list of Peaks
        peak_size (int) : desired peak extension size
        chrom_len (int) : chromsome length

    Returns:
        Peak representing the merger
    '''
    max_overlap = merge_overlap
    while len(peaks) > 1 and max_overlap >= merge_overlap:
        # find largest overlap
        max_i = 0
        max_overlap = peaks[0].end - peaks[1].start
        for i in range(1,len(peaks)-1):
            peaks_overlap = peaks[i].end - peaks[i+1].start
            if peaks_overlap > max_overlap:
                max_i = i
                max_overlap = peaks_overlap

        if max_overlap >= merge_overlap:
            # merge peaks
            peaks[max_i].merge(peaks[max_i+1], peak_size, chrom_len)

            # remove merged peak
            peaks = peaks[:max_i+1] + peaks[max_i+2:]

    return peaks


def merge_peaks_dist(peaks, peak_size, chrom_len):
    ''' Merge and grow the Peaks in the given list.

    Obsolete

    Attributes:
        peaks (list[Peak]) : list of Peaks
        peak_size (int) : desired peak extension size
        chrom_len (int) : chromsome length

    Returns:
        Peak representing the merger
    '''
    # determine peak midpoints
    peak_mids = []
    peak_weights = []
    for p in peaks:
        mid = (p.start + p.end - 1) / 2.0
        peak_mids.append(mid)
        peak_weights.append(1+len(p.act))

    # take the mean
    merge_mid = int(0.5+np.average(peak_mids, weights=peak_weights))

    # extend to the full size
    merge_start = max(0, merge_mid - peak_size/2)
    merge_end = merge_start + peak_size
    if chrom_len and merge_end > chrom_len:
        merge_end = chrom_len
        merge_start = merge_end - peak_size

    # merge activities
    merge_act = set()
    for p in peaks:
        merge_act |= p.act

    return Peak(merge_start, merge_end, merge_act)


class Peak:
    ''' Peak representation

    Attributes:
        start (int) : peak start
        end   (int) : peak end
        act   (set[int]) : set of target indexes where this peak is active.
    '''
    def __init__(self, start, end, act):
        self.start = start
        self.end = end
        self.act = act

    def extend(self, ext_len, chrom_len):
        ''' Extend the peak to the given length

        Args:
            ext_len (int) : length to extend the peak to
            chrom_len (int) : chromosome length to cap the peak at
        '''
        mid = find_midpoint(self.start, self.end)
        self.start = max(0, mid - ext_len/2)
        self.end = self.start + ext_len
        if chrom_len and self.end > chrom_len:
            self.end = chrom_len
            self.start = self.end - ext_len

    def bed_str(self, chrom, strand):
        ''' Return a BED-style line

        Args:
            chrom (str)
            strand (str)
        '''
        if len(self.act) == 0:
            act_str = '.'
        else:
            act_str = ','.join([str(ai) for ai in sorted(list(self.act))])
        cols = (chrom, str(self.start), str(self.end), '.', '1', strand, act_str)
        return '\t'.join(cols)

    def merge(self, peak2, ext_len, chrom_len):
        ''' Merge the given peak2 into this peak

        Args:
            peak2 (Peak)
            ext_len (int) : length to extend the merged peak to
            chrom_len (int) : chromosome length to cap the peak at
        '''
        # find peak midpoints
        peak_mids = [find_midpoint(self.start,self.end)]
        peak_mids.append(find_midpoint(peak2.start,peak2.end))

        # weight peaks
        peak_weights = [1+len(self.act)]
        peak_weights.append(1+len(peak2.act))

        # compute a weighted average
        merge_mid = int(0.5+np.average(peak_mids, weights=peak_weights))

        # extend to the full size
        merge_start = max(0, merge_mid - ext_len/2)
        merge_end = merge_start + ext_len
        if chrom_len and merge_end > chrom_len:
            merge_end = chrom_len
            merge_start = merge_end - ext_len

        # merge activities
        merge_act = self.act | peak2.act

        # set merge to this peak
        self.start = merge_start
        self.end = merge_end
        self.act = merge_act


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
