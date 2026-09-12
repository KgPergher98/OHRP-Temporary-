"""
RP - Risk Parity benchmark
==========================

Equal-risk-contribution portfolio built on top of riskfolio-lib, using the
Ledoit-Wolf shrinkage covariance estimator by default. Serves as the
low-concentration benchmark in the dissertation experiments.
"""

import pandas
import numpy
import riskfolio as rp

class RP():

    def __init__(self, method_cov:str = 'ledoit') -> None:
        self.method_cov = method_cov

    def run(self, X):
        self.X = X.copy()
        port = rp.Portfolio(returns = X.copy())
        port.assets_stats(method_mu='hist', method_cov=self.method_cov)
        w = port.rp_optimization(
            model = 'Classic',
            rm = 'MV', 
            rf = 0, 
            b = None,
            hist = True
        )
        w = w.weights
        return w