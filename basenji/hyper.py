#!/usr/bin/env python

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

    def rand(self):
        if self.type == int:
            rval = random.randint(self.range_min, self.range_max)
        else:
            if self.logscale:
                rval = np.power(2, random.uniform(np.log2(self.range_min), np.log2(self.range_max)))
            else:
                rval = random.uniform(self.range_min, self.range_max)

        return rval


def to_num(x):
    if x.find('.') != -1 or x.find('e') != -1:
        return float(x)
    else:
        return int(x)