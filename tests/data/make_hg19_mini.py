#!/usr/bin/env python
from optparse import OptionParser
import os
import subprocess

'''
Name

Description...
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] arg'
    parser = OptionParser(usage)
    #parser.add_option()
    (options,args) = parser.parse_args()

    mini_chroms = ['chr%d'%ci for ci in range(18,23)]

    fasta_file = '%s/assembly/hg19.fa' % os.environ['HG19']
    gaps_file = '%s/assembly/hg19_gaps.bed' % os.environ['HG19']

    # fasta
    fasta_out = open('hg19_mini.fa', 'w')

    print_line = False
    for line in open(fasta_file):
        if line[0] == '>':
            chrom = line[1:].rstrip()
            if chrom in mini_chroms:
                print_line = True
            else:
                print_line = False

        if print_line:
            print(line, end='', file=fasta_out)

    fasta_out.close()

    # index
    subprocess.call('samtools faidx hg19_mini.fa', shell=True)

    # gaps
    gaps_out = open('hg19_mini_gaps.bed', 'w')
    
    for line in open(gaps_file):
        a = line.split()
        if a[0] in mini_chroms:
            print(line, end='', file=gaps_out)

    gaps_out.close()
        

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
