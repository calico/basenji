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
from collections import OrderedDict
import os
import re

import matplotlib

matplotlib.use("PDF")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import basenji

"""basenji_hypers.py

Study the results of a hyper-parameter exploration.
"""


################################################################################
# main
################################################################################
def main():
    usage = "usage: %prog [options] args"
    parser = OptionParser(usage)
    parser.add_option(
        "-o",
        dest="out_dir",
        default="plots",
        help="Output directory [Default: %default]",
    )
    (options, args) = parser.parse_args()

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    ################################################################
    # read parameters

    hypers = OrderedDict()
    for line in open("hypers.txt"):
        hp = basenji.hyper.param()
        hp.init_line(line)
        hypers[hp.name] = hp

    ################################################################
    # read job accuracy

    acc_re = re.compile("Valid R2: (-?\d*\.\d*),")

    jobs = []

    ji = 0
    while os.path.isfile("jobs/%d.txt" % ji):
        # initialize job
        jobs.append(Job())

        # read accuracy by epoch
        for line in open("jobs/%d.txt" % ji):
            if line.startswith("Epoch"):
                acc_m = acc_re.search(line)
                jobs[ji].epoch_acc.append(float(acc_m.group(1)))
        ji += 1

    ################################################################
    # read job parameters

    for ji in range(len(jobs)):
        for line in open("hypers/%d.txt" % ji):
            a = line.split()
            jobs[ji].hypers[a[0]] = basenji.hyper.to_num(a[1])

    ################################################################
    # plot parameters versus accuracy

    for hp_name in hypers:
        hp_values = []
        hp_acc = []

        # extract hyper-parameter values and ultimate accuracies
        for ji in range(len(jobs)):
            if len(jobs[ji].epoch_acc) > 0:
                hp_values.append(jobs[ji].hypers[hp_name])
                hp_acc.append(np.max(jobs[ji].epoch_acc))

        hp_values = np.array(hp_values)
        hp_acc = np.array(hp_acc)

        # compute expected accuracy based on the other hyper-parameters

        hp_exp_acc = model.predict()

        jointplot(hp_values, hp_acc, "%s/%s.pdf" % (options.out_dir, hp_name))


def jointplot(vals1, vals2, out_pdf):
    plt.figure()
    g = sns.jointplot(vals1, vals2, alpha=0.8, color="black")
    ax = g.ax_joint
    xmin, xmax = scatter_lims(vals1)
    ymin, ymax = scatter_lims(vals2)
    ax.plot([xmin, xmax], [ymin, ymax], linestyle="--", color="black")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, linestyle=":")
    plt.tight_layout(w_pad=0, h_pad=0)
    plt.savefig(out_pdf)
    plt.close()


def scatter_lims(vals, buffer=0.05):
    vmin = np.nanmin(vals)
    vmax = np.nanmax(vals)

    buf = 0.05 * (vmax - vmin)

    vmin -= buf
    vmax += buf

    return vmin, vmax


class Job:
    def __init__(self):
        self.epoch_acc = []
        self.hypers = {}


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
