import numpy as np


class Median(object):

    def __init__(self):
        self.median = None
        self.vals = []

    def step(self, value):
        self.vals.append(value)

    def finalize(self):
        self.median = np.median(self.vals)
        return self.median


class Std(object):

    def __init__(self):
        self.std = None
        self.vals = []

    def step(self, value):
        self.vals.append(value)

    def finalize(self):
        self.std = np.std(self.vals, ddof=1)
        return self.std
