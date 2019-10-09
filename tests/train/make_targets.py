#!/usr/bin/env python
from optparse import OptionParser
import glob
import os
import subprocess
import sys

import pandas as pd

'''
make_targets.py

Make targets table for generating TF Records.
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] arg'
    parser = OptionParser(usage)
    #parser.add_option()
    (options,args) = parser.parse_args()

    species = ['human']
    assays = ['DNASE','ATAC','CAGE']

    # sources = ['encode', 'fantom', 'geo', 'uw-atlas']
    sources = ['encode', 'fantom']
    source_clip = {'encode':32, 'fantom':384, 'geo':64, 'uw-atlas':32}
    source_scale = {'encode':2, 'fantom':1, 'geo':1, 'uw-atlas':4}
    source_sum = {'encode':'mean', 'fantom':'sum', 'geo':'sum', 'uw-atlas':'mean'}

    targets_file = 'targets.txt'
    targets_out = open(targets_file, 'w')
    print('\t'.join(['index', 'genome', 'identifier', 'file', 'clip', 'scale', 'sum_stat', 'description']), file=targets_out)

    ti = 0

    for si in range(len(species)):
        for assay in assays:
            for source in sources:
                # collect w5 files
                w5_files = sorted(glob.glob('%s/datasets/%s/%s/%s/*/summary/*.w5' % (os.environ['TILLAGE'], species[si], assay.lower(), source)))
                if len(w5_files) > 0:
                    print('%s %s %s %d datasets' % (species[si], assay, source, len(w5_files)))

                # parse and write each w5 file
                for w5_file in w5_files:
                    w5_dir = os.path.split(w5_file)[0]
                    meta_file = '%s/metadata.txt' % w5_dir
                    # source = meta_file.split('/')[-4]

                    # read meta dict
                    meta_dict = read_meta(meta_file)

                    # check retirement
                    if meta_dict.get('status','active') != 'retired':
                        # augment description
                        assay = assay_succinct(meta_dict['assay'])
                        if assay == 'CHIP':
                            desc = '%s:%s:%s' % (assay, meta_dict['target'], meta_dict['description'])
                        else:
                            desc = '%s:%s' % (assay, meta_dict['description'])

                        cols = [str(ti), str(si), meta_dict['identifier'], w5_file, str(source_clip[source]), str(source_scale[source]), source_sum[source], desc]
                        print('\t'.join(cols), file=targets_out)

                        ti += 1

    targets_out.close()


    ##################################################
    # tests

    targets_df = pd.read_table(targets_file, index_col=0)
    unique_ids = set(targets_df.identifier)
    assert(len(unique_ids) == targets_df.shape[0])


def assay_succinct(assay):
    assay = assay.replace('-seq', '')
    return assay.upper()

def read_meta(meta_file):
    meta_dict = {}
    for line in open(meta_file):
        a = line.strip().split('\t')
        if len(a) > 1:
            meta_dict[a[0]] = a[1]
    return meta_dict


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
