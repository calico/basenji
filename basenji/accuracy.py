# Copyright 2017 Calico LLC

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

import pdb
import numpy as np
import scipy.stats as stats
import sklearn.metrics as metrics

"""
accuracy.py

Accuracy class to more succinctly store predictions/targets and
compute accuracy statistics.
"""


class Accuracy:
    def __init__(self, targets, preds, targets_na=None, loss=None, target_losses=None):
        self.targets = targets
        self.preds = preds
        self.targets_na = targets_na
        self.loss = loss
        self.target_losses = target_losses

        self.num_targets = len(self.target_losses)

    def pearsonr(self, log=False, pseudocount=1, clip=None):
        """ Compute target PearsonR vector. """

        pcor = np.zeros(self.num_targets)

        for ti in range(self.num_targets):
            if self.targets_na is not None:
                preds_ti = self.preds[~self.targets_na, ti].astype("float64")
                targets_ti = self.targets[~self.targets_na, ti].astype("float64")
            else:
                preds_ti = self.preds[:, :, ti].flatten().astype("float64")
                targets_ti = self.targets[:, :, ti].flatten().astype("float64")

            if clip is not None:
                preds_ti = np.clip(preds_ti, 0, clip)
                targets_ti = np.clip(targets_ti, 0, clip)

            if log:
                preds_ti = np.log2(preds_ti + pseudocount)
                targets_ti = np.log2(targets_ti + pseudocount)

            pc, _ = stats.pearsonr(targets_ti, preds_ti)
            pcor[ti] = pc

        return pcor

    def r2(self, log=False, pseudocount=1, clip=None):
        """ Compute target R2 vector. """
        r2_vec = np.zeros(self.num_targets)

        for ti in range(self.num_targets):
            if self.targets_na is not None:
                preds_ti = self.preds[~self.targets_na, ti].astype("float64")
                targets_ti = self.targets[~self.targets_na, ti].astype("float64")
            else:
                preds_ti = self.preds[:, :, ti].flatten().astype("float64")
                targets_ti = self.targets[:, :, ti].flatten().astype("float64")

            if clip is not None:
                preds_ti = np.clip(preds_ti, 0, clip)
                targets_ti = np.clip(targets_ti, 0, clip)

            if log:
                preds_ti = np.log2(preds_ti + pseudocount)
                targets_ti = np.log2(targets_ti + pseudocount)

            r2_vec[ti] = metrics.r2_score(targets_ti, preds_ti)

        return r2_vec

    def spearmanr(self):
        """ Compute target SpearmanR vector. """

        scor = np.zeros(self.num_targets)

        for ti in range(self.num_targets):
            if self.targets_na is not None:
                preds_ti = self.preds[~self.targets_na, ti]
                targets_ti = self.targets[~self.targets_na, ti]
            else:
                preds_ti = self.preds[:, :, ti].flatten()
                targets_ti = self.targets[:, :, ti].flatten()

            sc, _ = stats.spearmanr(targets_ti, preds_ti)
            scor[ti] = sc

        return scor


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
