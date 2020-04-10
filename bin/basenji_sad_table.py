#!/usr/bin/env python
from optparse import OptionParser

import h5py
import numpy as np
import pandas as pd
from tabulate import tabulate

'''
Name

Description...
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <sad_h5_file>'
    parser = OptionParser(usage)
    parser.add_option('-q', dest='quant_h5_file',
    		default=None, help='Compute quantiles using separated SAD HDF5.')
    (options,args) = parser.parse_args()

    if len(args) != 1:
    	parser.error('Must provide SAD HDF5 output file.')
    else:
    	sad_h5_file = args[0]

    sad_df = sad_h5_df(sad_h5_file, options.quant_h5_file)
    print(tabulate(sad_df, headers='keys', tablefmt='plain', showindex=False))


def sad_h5_df(sad_h5_file, quant_h5_file=None):
    sad_h5_open = h5py.File(sad_h5_file)
    sad = sad_h5_open['SAD'][:]
    num_snps, num_targets = sad.shape

    if quant_h5_file is None:
    	quant_h5_open = sad_h5_open
    else:
    	quant_h5_open = h5py.File(quant_h5_file)

    percentiles = quant_h5_open['percentiles']
    percentiles = np.append(percentiles, percentiles[-1])

    target_pct = quant_h5_open['SAD_pct'][:]
    sad_pct = []
    for ti in range(num_targets):
    	sad_ti = sad[:,ti]
    	sad_qi = np.searchsorted(target_pct[ti], sad_ti)
    	sad_pct.append(percentiles[sad_qi])
    sad_pct = np.array(sad_pct).T.flatten()
    # sad_pct = np.around(sad_pct, 4)

    sad = sad.flatten()
    # sad = np.around(sad, 3)

    snps = [snp.decode('UTF-8') for snp in sad_h5_open['snp']]
    snps = np.repeat(snps, num_targets)
    
    target_ids = [ti.decode('UTF-8') for ti in sad_h5_open['target_ids']]
    target_ids = np.tile(target_ids, num_snps)
    
    target_labels = [ti.decode('UTF-8') for ti in sad_h5_open['target_labels']]
    target_labels = np.tile(target_labels, num_snps)
    
    df = pd.DataFrame({'SNP':snps, 'SAD':sad, 'Quantile':sad_pct, 'TargetID':target_ids, 'TargetLabel':target_labels})
    return df


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
