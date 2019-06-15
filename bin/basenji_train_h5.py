#!/usr/bin/env python
# Copyright 2017 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
from __future__ import print_function

import sys
import time

import h5py
import numpy as np
import tensorflow as tf

from basenji import batcher
from basenji import params
from basenji import seqnn
from basenji import shared_flags

FLAGS = tf.app.flags.FLAGS

################################################################################
# main
################################################################################
def main(_):
    np.random.seed(FLAGS.seed)

    run(
        params_file=FLAGS.params,
        data_file=FLAGS.data,
        train_epochs=FLAGS.train_epochs,
        train_epoch_batches=FLAGS.train_epoch_batches,
        test_epoch_batches=FLAGS.test_epoch_batches,
    )


def run(params_file, data_file, train_epochs, train_epoch_batches, test_epoch_batches):

    #######################################################
    # load data
    #######################################################
    data_open = h5py.File(data_file)

    train_seqs = data_open["train_in"]
    train_targets = data_open["train_out"]
    train_na = None
    if "train_na" in data_open:
        train_na = data_open["train_na"]

    valid_seqs = data_open["valid_in"]
    valid_targets = data_open["valid_out"]
    valid_na = None
    if "valid_na" in data_open:
        valid_na = data_open["valid_na"]

    #######################################################
    # model parameters and placeholders
    #######################################################
    job = params.read_job_params(params_file)

    job["seq_length"] = train_seqs.shape[1]
    job["seq_depth"] = train_seqs.shape[2]
    job["num_targets"] = train_targets.shape[2]
    job["target_pool"] = int(np.array(data_open.get("pool_width", 1)))

    augment_shifts = [int(shift) for shift in FLAGS.augment_shifts.split(",")]
    ensemble_shifts = [int(shift) for shift in FLAGS.ensemble_shifts.split(",")]

    t0 = time.time()
    model = seqnn.SeqNN()
    model.build_feed(
        job,
        augment_rc=FLAGS.augment_rc,
        augment_shifts=augment_shifts,
        ensemble_rc=FLAGS.ensemble_rc,
        ensemble_shifts=ensemble_shifts,
    )

    print("Model building time %f" % (time.time() - t0))

    # adjust for fourier
    job["fourier"] = "train_out_imag" in data_open
    if job["fourier"]:
        train_targets_imag = data_open["train_out_imag"]
        valid_targets_imag = data_open["valid_out_imag"]

    #######################################################
    # prepare batcher
    #######################################################
    if job["fourier"]:
        batcher_train = batcher.BatcherF(
            train_seqs,
            train_targets,
            train_targets_imag,
            train_na,
            model.hp.batch_size,
            model.hp.target_pool,
            shuffle=True,
        )
        batcher_valid = batcher.BatcherF(
            valid_seqs,
            valid_targets,
            valid_targets_imag,
            valid_na,
            model.batch_size,
            model.target_pool,
        )
    else:
        batcher_train = batcher.Batcher(
            train_seqs,
            train_targets,
            train_na,
            model.hp.batch_size,
            model.hp.target_pool,
            shuffle=True,
        )
        batcher_valid = batcher.Batcher(
            valid_seqs,
            valid_targets,
            valid_na,
            model.hp.batch_size,
            model.hp.target_pool,
        )
    print("Batcher initialized")

    #######################################################
    # train
    #######################################################

    # checkpoints
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    if FLAGS.log_device_placement:
        config.log_device_placement = True
    with tf.Session(config=config) as sess:
        t0 = time.time()

        # set seed
        tf.set_random_seed(FLAGS.seed)

        if FLAGS.logdir:
            train_writer = tf.summary.FileWriter(FLAGS.logdir + "/train", sess.graph)
        else:
            train_writer = None

        if FLAGS.restart:
            # load variables into session
            saver.restore(sess, FLAGS.restart)
        else:
            # initialize variables
            print("Initializing...")
            sess.run(tf.global_variables_initializer())
            print("Initialization time %f" % (time.time() - t0))

        train_loss = None
        best_loss = None
        early_stop_i = 0

        epoch = 0
        while (
            train_epochs is None or epoch < train_epochs
        ) and early_stop_i < FLAGS.early_stop:
            t0 = time.time()

            # train
            train_loss, steps = model.train_epoch_h5(
                sess,
                batcher_train,
                sum_writer=train_writer,
                epoch_batches=train_epoch_batches,
                no_steps=FLAGS.no_steps,
            )

            # validate
            valid_acc = model.test_h5(
                sess, batcher_valid, test_batches=test_epoch_batches
            )
            valid_loss = valid_acc.loss
            valid_r2 = valid_acc.r2().mean()
            del valid_acc

            best_str = ""
            if best_loss is None or valid_loss < best_loss:
                best_loss = valid_loss
                best_str = ", best!"
                early_stop_i = 0
                saver.save(sess, "%s/model_best.tf" % FLAGS.logdir)
            else:
                early_stop_i += 1

            # measure time
            et = time.time() - t0
            if et < 600:
                time_str = "%3ds" % et
            elif et < 6000:
                time_str = "%3dm" % (et / 60)
            else:
                time_str = "%3.1fh" % (et / 3600)

            # print update
            print(
                "Epoch: %3d,  Steps: %7d,  Train loss: %7.5f,  Valid loss: %7.5f,  Valid R2: %7.5f,  Time: %s%s"
                % (
                    epoch + 1,
                    steps,
                    train_loss,
                    valid_loss,
                    valid_r2,
                    time_str,
                    best_str,
                )
            )
            sys.stdout.flush()

            if FLAGS.check_all:
                saver.save(sess, "%s/model_check%d.tf" % (FLAGS.logdir, epoch))

            # update epoch
            epoch += 1

        if FLAGS.logdir:
            train_writer.close()


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    tf.app.run(main)
