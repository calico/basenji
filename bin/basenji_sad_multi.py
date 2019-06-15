#!/usr/bin/env python
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

from optparse import OptionParser
import glob
import os
import pickle
import shutil
import subprocess
import sys

import h5py
import numpy as np

try:
    import zarr
except ImportError:
    pass

import slurm

"""
basenji_sad_multi.py

Compute SNP expression difference scores for variants in a VCF file,
using multiple processes.
"""

################################################################################
# main
################################################################################
def main():
    usage = "usage: %prog [options] <params_file> <model_file> <vcf_file>"
    parser = OptionParser(usage)

    # sad
    parser.add_option(
        "-c",
        dest="csv",
        default=False,
        action="store_true",
        help="Print table as CSV [Default: %default]",
    )
    parser.add_option(
        "-f",
        dest="genome_fasta",
        default="%s/data/hg19.fa" % os.environ["BASENJIDIR"],
        help="Genome FASTA for sequences [Default: %default]",
    )
    parser.add_option(
        "-g",
        dest="genome_file",
        default="%s/data/human.hg19.genome" % os.environ["BASENJIDIR"],
        help="Chromosome lengths file [Default: %default]",
    )
    parser.add_option(
        "--local",
        dest="local",
        default=1024,
        type="int",
        help="Local SAD score [Default: %default]",
    )
    parser.add_option("-n", dest="norm_file", default=None, help="Normalize SAD scores")
    parser.add_option(
        "-o",
        dest="out_dir",
        default="sad",
        help="Output directory for tables and plots [Default: %default]",
    )
    parser.add_option(
        "--pseudo",
        dest="log_pseudo",
        default=1,
        type="float",
        help="Log2 pseudocount [Default: %default]",
    )
    parser.add_option(
        "--rc",
        dest="rc",
        default=False,
        action="store_true",
        help="Average forward and reverse complement predictions [Default: %default]",
    )
    parser.add_option(
        "--shifts",
        dest="shifts",
        default="0",
        type="str",
        help="Ensemble prediction shifts [Default: %default]",
    )
    parser.add_option(
        "--stats",
        dest="sad_stats",
        default="SAD",
        help="Comma-separated list of stats to save. [Default: %default]",
    )
    parser.add_option(
        "-t",
        dest="targets_file",
        default=None,
        type="str",
        help="File specifying target indexes and labels in table format",
    )
    parser.add_option(
        "--ti",
        dest="track_indexes",
        default=None,
        type="str",
        help="Comma-separated list of target indexes to output BigWig tracks",
    )
    parser.add_option(
        "--txt",
        dest="out_txt",
        default=False,
        action="store_true",
        help="Output stats to text table [Default: %default]",
    )
    parser.add_option(
        "-u",
        dest="penultimate",
        default=False,
        action="store_true",
        help="Compute SED in the penultimate layer [Default: %default]",
    )
    parser.add_option(
        "-z",
        dest="out_zarr",
        default=False,
        action="store_true",
        help="Output stats to sad.zarr [Default: %default]",
    )

    # multi
    parser.add_option(
        "--cpu",
        dest="cpu",
        default=False,
        action="store_true",
        help="Run without a GPU [Default: %default]",
    )
    parser.add_option(
        "--name",
        dest="name",
        default="sad",
        help="SLURM name prefix [Default: %default]",
    )
    parser.add_option(
        "-p",
        dest="processes",
        default=None,
        type="int",
        help="Number of processes, passed by multi script",
    )
    parser.add_option(
        "-q",
        dest="queue",
        default="k80",
        help="SLURM queue on which to run the jobs [Default: %default]",
    )
    parser.add_option(
        "-r",
        dest="restart",
        default=False,
        action="store_true",
        help="Restart a partially completed job [Default: %default]",
    )
    (options, args) = parser.parse_args()

    if len(args) != 3:
        parser.error("Must provide parameters and model files and VCF file")
    else:
        params_file = args[0]
        model_file = args[1]
        vcf_file = args[2]

    #######################################################
    # prep work

    # output directory
    if not options.restart:
        if os.path.isdir(options.out_dir):
            print("Please remove %s" % options.out_dir, file=sys.stderr)
            exit(1)
        os.mkdir(options.out_dir)

    # pickle options
    options_pkl_file = "%s/options.pkl" % options.out_dir
    options_pkl = open(options_pkl_file, "wb")
    pickle.dump(options, options_pkl)
    options_pkl.close()

    #######################################################
    # launch worker threads
    jobs = []
    for pi in range(options.processes):
        if not options.restart or not job_completed(options, pi):
            if options.cpu:
                cmd = ""
            else:
                cmd = "source activate tf1.13-gpu;"

            cmd += " basenji_sad.py %s %s %d" % (options_pkl_file, " ".join(args), pi)

            name = "%s_p%d" % (options.name, pi)
            outf = "%s/job%d.out" % (options.out_dir, pi)
            errf = "%s/job%d.err" % (options.out_dir, pi)

            num_gpu = 1 * (not options.cpu)

            j = slurm.Job(
                cmd,
                name,
                outf,
                errf,
                queue=options.queue,
                gpu=num_gpu,
                mem=15000,
                time="7-0:0:0",
            )
            jobs.append(j)

    slurm.multi_run(
        jobs, max_proc=options.processes, verbose=True, launch_sleep=10, update_sleep=60
    )

    #######################################################
    # collect output

    if options.out_txt:
        collect_table("sad_table.txt", options.out_dir, options.processes)

    elif options.out_zarr:
        collect_zarr("sad.zarr", options.out_dir, options.processes)

    else:
        collect_h5("sad.h5", options.out_dir, options.processes)

    # for pi in range(options.processes):
    #     shutil.rmtree('%s/job%d' % (options.out_dir,pi))


def collect_table(file_name, out_dir, num_procs):
    os.rename("%s/job0/%s" % (out_dir, file_name), "%s/%s" % (out_dir, file_name))
    for pi in range(1, num_procs):
        subprocess.call(
            "tail -n +2 %s/job%d/%s >> %s/%s"
            % (out_dir, pi, file_name, out_dir, file_name),
            shell=True,
        )


def collect_h5(file_name, out_dir, num_procs):
    # count variants
    num_variants = 0
    for pi in range(num_procs):
        # open job
        job_h5_file = "%s/job%d/%s" % (out_dir, pi, file_name)
        job_h5_open = h5py.File(job_h5_file, "r")
        num_variants += len(job_h5_open["snp"])
        job_h5_open.close()

    # initialize final h5
    final_h5_file = "%s/%s" % (out_dir, file_name)
    final_h5_open = h5py.File(final_h5_file, "w")

    # keep dict for string values
    final_strings = {}

    job0_h5_file = "%s/job0/%s" % (out_dir, file_name)
    job0_h5_open = h5py.File(job0_h5_file, "r")
    for key in job0_h5_open.keys():
        if key in ["percentiles", "target_ids", "target_labels"]:
            # copy
            final_h5_open.create_dataset(key, data=job0_h5_open[key])

        elif key[-4:] == "_pct":
            values = np.zeros(job0_h5_open[key].shape)
            final_h5_open.create_dataset(key, data=values)

        elif job0_h5_open[key].dtype.char == "S":
            final_strings[key] = []

        elif job0_h5_open[key].ndim == 1:
            final_h5_open.create_dataset(
                key, shape=(num_variants,), dtype=job0_h5_open[key].dtype
            )

        else:
            num_targets = job0_h5_open[key].shape[1]
            final_h5_open.create_dataset(
                key, shape=(num_variants, num_targets), dtype=job0_h5_open[key].dtype
            )

    job0_h5_open.close()

    # set values
    vi = 0
    for pi in range(num_procs):
        # open job
        job_h5_file = "%s/job%d/%s" % (out_dir, pi, file_name)
        job_h5_open = h5py.File(job_h5_file, "r")

        # append to final
        for key in job_h5_open.keys():
            if key in ["percentiles", "target_ids", "target_labels"]:
                # once is enough
                pass

            elif key[-4:] == "_pct":
                # average
                u_k1 = np.array(final_h5_open[key])
                x_k = np.array(job_h5_open[key])
                final_h5_open[key][:] = u_k1 + (x_k - u_k1) / (pi + 1)

            else:
                if job_h5_open[key].dtype.char == "S":
                    final_strings[key] += list(job_h5_open[key])
                else:
                    job_variants = job_h5_open[key].shape[0]
                    final_h5_open[key][vi : vi + job_variants] = job_h5_open[key]

        vi += job_variants
        job_h5_open.close()

    # create final string datasets
    for key in final_strings:
        final_h5_open.create_dataset(key, data=np.array(final_strings[key], dtype="S"))

    final_h5_open.close()


def collect_zarr(file_name, out_dir, num_procs):
    final_zarr_file = "%s/%s" % (out_dir, file_name)

    # seed w/ job0
    job_zarr_file = "%s/job0/%s" % (out_dir, file_name)
    shutil.copytree(job_zarr_file, final_zarr_file)

    # open final
    final_zarr_open = zarr.open_group(final_zarr_file)

    for pi in range(1, num_procs):
        # open job
        job_zarr_file = "%s/job%d/%s" % (out_dir, pi, file_name)
        job_zarr_open = zarr.open_group(job_zarr_file, "r")

        # append to final
        for key in final_zarr_open.keys():
            if key in ["percentiles", "target_ids", "target_labels"]:
                # once is enough
                pass

            elif key[-4:] == "_pct":
                # average
                u_k1 = np.array(final_zarr_open[key])
                x_k = np.array(job_zarr_open[key])
                final_zarr_open[key] = u_k1 + (x_k - u_k1) / (pi + 1)

            else:
                # append
                final_zarr_open[key].append(job_zarr_open[key])


def job_completed(options, pi):
    """Check whether a specific job has generated its
     output file."""
    if options.out_txt:
        out_file = "%s/job%d/sad_table.txt" % (options.out_dir, pi)
    elif options.out_zarr:
        out_file = "%s/job%d/sad.zarr" % (options.out_dir, pi)
    elif options.csv:
        out_file = "%s/job%d/sad_table.csv" % (options.out_dir, pi)
    else:
        out_file = "%s/job%d/sad.h5" % (options.out_dir, pi)
    return os.path.isfile(out_file) or os.path.isdir(out_file)


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
