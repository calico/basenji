# Copyright 2017 Calico LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from __future__ import print_function

import pdb
import os
import subprocess

import numpy as np
import pandas as pd
from pysam import VariantFile

'''
emerald.py

Methods to query LD using emeraLD.
'''

class EmeraldVCF:
  def __init__(self, pop_vcf_stem):
    self.pop_vcf_stem = pop_vcf_stem

  def fetch(self, chrm, pos_start, pos_end, return_samples=False):
    vcf_file = '%s.%s.vcf.gz' % (self.pop_vcf_stem, chrm)
    vcf_open = VariantFile(vcf_file, drop_samples=(not return_samples))
    return vcf_open.fetch(chrm, pos_start, pos_end)


  def query_ld(self, snp_id, chrm, pos,
               ld_t=0.1, return_pos=False,
               max_ld_distance=1000000):
    """Retrieve SNPs in LD with the given SNP."""

    chr_vcf_file = '%s.%s.vcf.gz' % (self.pop_vcf_stem, chrm)
    if not os.path.isfile(chr_vcf_file):
        print('WARNING: %s VCF not found.' % chrm)
        ld_df = pd.DataFrame()

    else:
        # determine search region
        ld_region_start = max(0, pos - max_ld_distance)
        ld_region_end = pos + max_ld_distance
        region_str = '--region %s:%d-%d' % (chrm, ld_region_start, ld_region_end)

        # construct emerald command
        cmd  = 'emeraLD'
        cmd += ' -i %s' % chr_vcf_file
        cmd += ' --rsid %s' % snp_id
        cmd += ' %s' % region_str
        cmd += ' --threshold %f' % ld_t
        cmd += ' --no-phase'
        cmd += ' --extra'
        cmd += ' --stdout'
        cmd += ' 2> /dev/null'

        ld_snps = [snp_id]
        ld_r = [1.0]
        ld_pos = [pos]

        # parse returned lines
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        for line in proc.stdout:
            line = line.decode('UTF-8')
            if not line.startswith('#'):
                a = line.split()
                ld_pos.append(int(a[4]))
                ld_snps.append(a[5])
                ld_r.append(float(a[7]))
        proc.communicate()

        # sort by position
        sort_indexes = np.argsort(ld_pos)
        ld_snps = np.array(ld_snps)[sort_indexes]
        ld_r = np.array(ld_r)[sort_indexes]

        # construct data frame
        ld_df = pd.DataFrame()
        ld_df['snp'] = ld_snps
        ld_df['r'] = ld_r

        if return_pos:
            ld_df['chr'] = chrm
            ld_df['pos'] = np.array(ld_pos)[sort_indexes]

    return ld_df
