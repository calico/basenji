#!/usr/bin/env python
from optparse import OptionParser
import glob
import os
import pickle
import shutil
import subprocess

import slurm

'''
basenji_sed_multi.py

Compute SNP expression difference scores for variants in a VCF file,
using multiple processes.
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <params_file> <model_file> <genes_hdf5_file> <vcf_file>'
    parser = OptionParser(usage)
    parser.add_option('-a', dest='all_sed', default=False, action='store_true', help='Print all variant-gene pairs, as opposed to only nonzero [Default: %default]')
    parser.add_option('-b', dest='batch_size', default=None, type='int', help='Batch size [Default: %default]')
    parser.add_option('-c', dest='csv', default=False, action='store_true', help='Print table as CSV [Default: %default]')
    parser.add_option('-g', dest='genome_file', default='%s/assembly/human.hg19.genome'%os.environ['HG19'], help='Chromosome lengths file [Default: %default]')
    parser.add_option('-i', dest='index_snp', default=False, action='store_true', help='SNPs are labeled with their index SNP as column 6 [Default: %default]')
    parser.add_option('-o', dest='out_dir', default='sed', help='Output directory for tables and plots [Default: %default]')
    parser.add_option('-p', dest='processes', default=2, type='int', help='Number of parallel processes to run [Default: %default]')
    parser.add_option('-s', dest='score', default=False, action='store_true', help='SNPs are labeled with scores as column 7 [Default: %default]')
    parser.add_option('-t', dest='target_wigs_file', default=None, help='Store target values, extracted from this list of WIG files')
    parser.add_option('--ti', dest='track_indexes', help='Comma-separated list of target indexes to output BigWig tracks')
    parser.add_option('-x', dest='transcript_table', default=False, action='store_true', help='Print transcript table in addition to gene [Default: %default]')
    parser.add_option('-w', dest='tss_width', default=1, type='int', help='Width of bins considered to quantify TSS transcription [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 4:
        parser.error('Must provide parameters and model files, genes HDF5 file, and QTL VCF file')
    else:
        params_file = args[0]
        model_file = args[1]
        genes_hdf5_file = args[2]
        vcf_file = args[3]

    #######################################################
    # prep work

    # output directory
    if os.path.isdir(options.out_dir):
        shutil.rmtree(options.out_dir)
    os.mkdir(options.out_dir)

    # pickle options
    options_pkl_file = '%s/options.pkl' % options.out_dir
    options_pkl = open(options_pkl_file, 'wb')
    pickle.dump(options, options_pkl)
    options_pkl.close()

    #######################################################
    # launch worker threads
    jobs = []
    for pi in range(options.processes):
        cmd = 'source activate py3_gpu; basenji_sed.py %s %s %d' % (options_pkl_file, ' '.join(args), pi)
        name = 'sed_p%d'%pi
        outf = '%s/job%d.out' % (options.out_dir,pi)
        errf = '%s/job%d.err' % (options.out_dir,pi)
        j = slurm.Job(cmd, name, outf, errf, queue='gpu', mem=32000, time='7-0:0:0', gpu=1)
        jobs.append(j)

    slurm.multi_run(jobs, max_proc=options.processes, verbose=True, sleep_time=60)

    #######################################################
    # collect output

    collect_table_multi('sed_gene.txt', options.out_dir, options.processes)
    if options.transcript_table:
        collect_table('sed_tx.txt', options.out_dir, options.processes)

    if options.track_indexes is not None:
        if not os.path.isdir('%s/tracks' % options.out_dir):
            os.mkdir('%s/tracks' % options.out_dir)

        for track_file in glob.glob('%s/job*/tracks/*'):
            track_base = os.path.split(track_file)[1]
            os.rename(track_file, '%s/tracks/%s' % (options.out_dir, track_base))

    for pi in range(options.processes):
        shutil.rmtree('%s/job%d' % (options.out_dir,pi))


def collect_table(file_name, out_dir, num_procs):
    os.rename('%s/job0/%s' % (out_dir, file_name), '%s/%s' % (out_dir, file_name))
    for pi in range(1, num_procs):
        subprocess.call('tail -n +2 %s/job%d/%s >> %s/%s' % (out_dir, pi, file_name, out_dir, file_name), shell=True)

def collect_table_multi(file_name, out_dir, num_procs):
    collect_out = open('%s/%s' % (out_dir, file_name), 'w')

    header = open('%s/job0/%s' % (out_dir, file_name)).readline().rstrip()
    print(header, file=collect_out)

    multi_lines = []

    for pi in range(num_procs):
        table_in = open('%s/job%d/%s' % (out_dir, pi, file_name), 'w')
        table_in.readline()

        for line in table_in:
            a = line.split()
            if a[3][-6:] == '_multi':
                multi_key = (a[0], a[3][:-6], a[5])
                if multi_key in multi_lines:
                    multi_lines[multi_key].add(a)
                else:
                    multi_lines[multi_key] = MultiLine(a)
            else:
                print(line, end='', file=collect_out)

        table_in.close()

    for multi_key in multi_lines:
        print(multi_lines[multi_key], file=collect_out)

    collect_out.close()


class MultiLine:
    def __init__(self, a):
        self.rsid = a[0]
        self.a1 = a[1]
        self.a2 = a[2]
        self.gene = a[3][:-6]
        self.snp_dist_gene = int(a[4])
        self.target = a[5]
        self.ref_pred = [float(a[6])]
        self.alt_pred = [float(a[7])]

    def add(self, a):
        self.snp_dist_gene = min(self.snp_dist_gene, int(a[4]))
        self.ref_pred.append(float(a[6]))
        self.alt_pred.append(float(a[7]))

    def __str__(self):
        ref_pred = np.sum(self.ref_pred)
        alt_pred = np.sum(self.alt_pred)
        sed = ref_pred - alt_pred
        ser = np.log2(ref_pred+1) - np.log2(alt_pred+1)
        cols = (self.rsid, self.a1, self.a2, self.gene, self.snp_dist_gene, self.target, ref_pred, alt_pred, sed, ser)
        return '%-13s %s %5s %12s %5d %12s %6.4f %6.4f %7.4f %7.4f' % tuple(cols)


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
