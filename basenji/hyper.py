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

import random

import numpy as np

####################################################################
# param

class param:
    def __init__(self):
        pass

    def init_line(self, line):
        a = line.split()

        self.name = a[0]

        self.range_min = to_num(a[1])
        self.range_max = to_num(a[2])

        self.logscale = False
        if len(a) > 3 and a[3] == 'log':
            self.logscale = True

        self.type = float
        if type(self.range_min) == int and type(self.range_max) == int:
            self.type = int

        self.value = None

    def rand(self):
        if self.type == int:
            self.value = random.randint(self.range_min, self.range_max)
        else:
            if self.logscale:
                self.value = np.power(2, random.uniform(np.log2(self.range_min), np.log2(self.range_max)))
            else:
                self.value = random.uniform(self.range_min, self.range_max)

        return self.value


def to_num(x):
    if x.find('.') != -1 or x.find('e') != -1:
        return float(x)
    else:
        return int(x)