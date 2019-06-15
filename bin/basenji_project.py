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

import os
import pdb
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor

import basenji

"""basenji_project.py

Project the outcome of a set of experiments.
"""


################################################################################
# main
################################################################################
def main():
    usage = "usage: %prog [options] args"
    parser = OptionParser(usage)
    parser.add_option(
        "-d",
        dest="exp_dir",
        default=".",
        help="Experiment directory [Default: %default]",
    )
    parser.add_option(
        "-m",
        dest="memory",
        default=6,
        type="int",
        help="Memory to look back at [Default: %default]",
    )
    parser.add_option(
        "-o",
        dest="out_dir",
        default="proj",
        help="Output directory [Default: %default]",
    )
    (options, args) = parser.parse_args()

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    ################################################################
    # model % decrease as a function of several prior % decreases

    training_epochs = []
    prior_decreases = []
    target_decreases = []

    job_losses = []

    ji = 0
    while os.path.isfile("jobs/%d.txt" % ji):
        training_file = "jobs/%d.txt" % ji

        # read validation losses
        valid_losses = read_job_loss(training_file)

        # transform to training data
        for ei in range(options.memory + 1, len(valid_losses)):
            training_epochs.append(ei)

            target_decreases.append(
                (valid_losses[ei - 1] - valid_losses[ei]) / valid_losses[ei - 1]
            )

            prior_decreases.append([])
            for mi in range(options.memory):
                prior_decreases[-1].append(
                    (valid_losses[ei - mi - 2] - valid_losses[ei - mi - 1])
                    / valid_losses[ei - mi - 2]
                )

        # save losses
        job_losses.append(np.array(valid_losses))

        ji += 1

    num_jobs = ji

    # convert to array
    prior_decreases = np.array(prior_decreases)
    target_decreases = np.array(target_decreases)

    # model
    model = Ridge(alpha=1e-3, fit_intercept=False)
    # model = GaussianProcessRegressor()
    model.fit(prior_decreases, target_decreases)

    # validate model
    epoch_cmap = plt.get_cmap("viridis_r")
    plt.figure()
    pred_decreases = model.predict(prior_decreases)
    # plt.scatter(target_decreases, pred_decreases, color=[epoch_cmap(ei/np.max(training_epochs)) for ei in training_epochs])
    sns.jointplot(target_decreases, pred_decreases)

    plt.savefig("%s/pred_true.pdf" % options.out_dir)
    plt.close()

    ################################################################
    # project stable losses for existing runs

    cols = ("job", "iter", "loss", "proj_mean", "proj_sd", "p_best")
    print("%3s  %4s  %7s  %7s  %7s  %7s" % cols)

    for ji in range(num_jobs):
        if len(job_losses[ji] > 0):
            job_proj_loss = 0
            if len(job_losses[ji]) > 4:
                job_proj_loss = project_loss(job_losses[ji], model, options.memory)

            cols = (ji, len(job_losses[ji]), job_losses[ji].min(), job_proj_loss, 0, 0)
            print("%-3d  %4d  %7.5f  %7.5f  %7.5f  %7.5f" % cols)


def project_loss(job_losses, model, memory):
    """ Project the stable loss using a model for iteration
         to iteration decreases. """

    loss = job_losses[-1]
    decrease = 1

    prior_decrease = []
    for mi in range(memory):
        prior_decrease.append(
            (job_losses[-2 - mi] - job_losses[-1 - mi]) / job_losses[-2 - mi]
        )

    while decrease > 1e-3:
        # predict decrease
        decrease = model.predict(np.array([prior_decrease]))[0]
        # print(decrease)

        # update loss
        prev_loss = loss
        loss *= 1.0 - decrease

        # update prior decrease
        prior_decrease = [decrease] + prior_decrease[:-1]

    return loss


def read_job_loss(training_file):
    """ Read in validation loss updates """
    valid_losses = []
    for line in open(training_file):
        if line.startswith("Epoch"):
            a = line.split(",")
            valid_str = a[1]
            valid_loss = float(valid_str.split()[-1])
            valid_losses.append(valid_loss)
    return valid_losses


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
    # pdb.runcall(main)
