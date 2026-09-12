"""
OHRP - Orthogonal Hierarchical Risk Parity
==========================================

Reference implementation of the allocation method introduced in:

    PERGHER, K. G. R.; SOLDERA, J.; SCHARCANSKI, J.
    "An Orthogonal Hierarchical Risk Parity Allocation Method for Improved
    Portfolio Out-of-Sample Performance". IEEE Access, v. 14, 2026.
    DOI: 10.1109/ACCESS.2026.3656702

Idea in one line
----------------
Rather than changing the HRP machinery, OHRP changes *what HRP sees*. Asset
returns are projected onto a lower-dimensional orthogonal subspace that
preserves data locality (PCA followed by OLPP), and only then fed to the three
canonical HRP stages: hierarchical clustering, quasi-diagonalization and
recursive bisection.

The projection is not fixed. At every rebalancing date the triple

    k -> number of nearest neighbours in the locality graph
    d -> dimensionality of the projected subspace
    r -> fraction of the variance retained by the PCA stage

is re-selected by exhaustive grid search against an in-sample objective
(volatility by default), which turns a static preprocessing step into an
adaptive component of the strategy.

Usage
-----
    from OHRP import OHRP

    model   = OHRP(codep="pearson", linkage="ward")
    weights = model.run(X=returns, k=[1, 3, 5], d=[5, 10, 15], r=[0.94, 0.98])
    model._bestHyperparams   # -> {"k": ..., "d": ..., "r": ...}

`X` is a DataFrame of *returns* (rows = dates, columns = tickers).
"""

import pandas
import numpy
import traceback

from HRP import HRP
from COLPP import COLPP

class OHRP():

    def __init__(self, codep:str = "pearson", linkage:str = "ward") -> None:
        self._options = {
            "gnd"         : None,
            "NeighborMode": "NonSupervised",
            "WeightMode"  : "HeatKernel",
            "bNormalized" : 0,
            "bLDA"        : 0,
            "codependence": codep,
            "linkage"     : linkage
        }

    def _convertLabel(lbs):
        conv = pandas.DataFrame(lbs.label.unique().tolist(), columns = ["setor"])
        conv["idx"] = conv.index
        conv.index = conv.setor
        lbs2 = lbs.copy()
        for t in range(lbs2.shape[0]):
            lbs2.iloc[t, 0] = conv.loc[lbs2.iloc[t, 0], "idx"]
        return conv, lbs2

    def run(self, X, k:list = [5], d:list = [10], r:list = [0.9], L = None, optimize_metric:str = "volatility"):

        self.X = X.copy()
        self._options["t"] = COLPP.find_optimal(df = self.X)
        self.X = self.X.transpose()

        # Adjust Labels
        if L is None:
            L = pandas.DataFrame("X", index = X.columns.tolist(), columns = ["label"])
        if (L.shape[1] != 1) or (L.columns[0] != "label"):
            raise ValueError("Labels L must have a single column named 'label'.")
        conv, self.L = OHRP._convertLabel(lbs = L)

        # Adjust iterables
        if type(k) == int  : k = [k]
        if type(d) == int  : d = [d]
        if type(r) == float: r = [r]

        best_metric  = - numpy.inf
        best_combo   = {"d": 0, "k": 0, "r": 0}
        best_weights = pandas.Series(numpy.nan, index = self.X.columns.tolist(), name = "weights")

        for ki in k:
            for di in d:
                for ri in r:
                    try:
                        weights, inSampleRet = self.OrthoRedDim(
                            X = self.X.copy(), L = self.L.copy(), k = ki, d = di, r = ri
                        )

                        if optimize_metric == "volatility":
                            metric = 1/numpy.std(inSampleRet)
                        elif optimize_metric == "sharpe":
                            metric = numpy.sqrt(252) * numpy.mean(inSampleRet) / numpy.std(inSampleRet)
                        elif optimize_metric == "return":
                            metric = numpy.mean(inSampleRet)
                    except Exception as exc:
                        traceback.print_exc()
                        continue
                    if optimize_metric in ['volatility', 'sharpe', 'return']:
                        if numpy.isfinite(metric) and metric > best_metric:
                            best_metric  = metric
                            best_combo   = {"d": di, "k": ki, "r": ri}
                            best_weights = weights.copy()

        self._bestHyperparams = best_combo

        if best_weights.isna().all():
            raise RuntimeError(
                "OHRP.run: no (k, d, r) combination produced a valid portfolio "
                "(all attempts failed or yielded non-finite metrics)."
            )

        return best_weights

    def OrthoRedDim(self, X, L, k:int, d:int, r:float):

        self._options["k"]          = k
        self._options["ReducedDim"] = d
        self._options["PCARatio"]   = r

        W = COLPP.affinity_matrix(df = X, labels = L, ops = self._options, self_connection = False, olpp = True)

        D = COLPP.diagonal(affinity = W)

        X_std = (X.copy() - X.copy().mean(axis = 0))
        U, S, V = COLPP.SVD(X = X_std)

        U, S, V = COLPP.cut_on_ratio(U = U, V = V, S = S, pca_ratio = self._options["PCARatio"])

        eigen_vector = COLPP.build_weights(
            U = U, S = S, V = V, D = D, W = W, reduced_dim = self._options["ReducedDim"], bd = True
        )

        red_space = COLPP.project_data(X = X, P = eigen_vector, L = L)
        red_space = red_space[red_space.drop(["label"], axis = 1).std(axis = 1) != 0]

        portfolio = HRP(
            Y = red_space.drop(["label"], axis = 1).transpose().copy(),
            correlation_method = self._options["codependence"],
            linkage_method     = self._options["linkage"]
        )
        portfolio.run()
        weights_ohrp = portfolio.weights

        weights_ohrp /= numpy.sum(weights_ohrp)

        inp = X.transpose().copy() * weights_ohrp.transpose()
        inp.dropna(axis = 1, how = "all", inplace = True)
        inp = inp.sum(axis = 1)

        return weights_ohrp, inp
