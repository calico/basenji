#!/usr/bin/env python
from optparse import OptionParser

import h5py
import numpy as np

'''
basenji_sample.py

Sample from an HDF5 file.
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <sample_pct> <in_hdf5_file> <out_hdf5_file>'
    parser = OptionParser(usage)
    # parser.add_option('-b', dest='batch_size', default=None, type='int', help='Batch size [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 3:
        parser.error('Must provide sample %, input HDF5 file, and output HDF5 file')
    else:
        sample_pct = float(args[0])
        in_h5_file = args[1]
        out_h5_file = args[2]

    #######################################
    # open HDF5

    in_h5_open = h5py.File(in_h5_file)
    out_h5_open = h5py.File(out_h5_file, 'w')

    #######################################
    # copy over similar info

    # store pooling
    out_h5_open.create_dataset('pool_width', data=in_h5_open['pool_width'], dtype='int')

    # store targets
    out_h5_open.create_dataset('target_labels', data=in_h5_open['target_labels'])

    #######################################
    # sample sequences and targets

    for dataset in ['train','valid','test']:
        sample_num = int(np.round(sample_pct*in_h5_open['%s_in'%dataset].shape[0]))
        out_h5_open.create_dataset('%s_in'%dataset, data=in_h5_open['%s_in'%dataset][:sample_num])
        out_h5_open.create_dataset('%s_out'%dataset, data=in_h5_open['%s_out'%dataset][:sample_num])
        out_h5_open.create_dataset('%s_na'%dataset, data=in_h5_open['%s_na'%dataset][:sample_num])

    #######################################
    # close HDF5

    in_h5_open.close()
    out_h5_open.close()


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
