"""
EW - Equally Weighted benchmark
===============================

The 1/N portfolio. It ignores all sample information and is therefore immune to
estimation error, which is exactly what makes it a demanding benchmark
(DeMiguel, Garlappi and Uppal, 2009). Also the natural reference point for
asset-level concentration measures, being perfectly diversified by construction.
"""

import pandas

class EW():
    def __init__(self):
        pass

    def run(self, X):
        self.X = X.copy()
        weights_ew = pandas.DataFrame(1 / self.X.shape[1], index = self.X.columns.tolist(), columns = ["weights"])
        return weights_ew.weights