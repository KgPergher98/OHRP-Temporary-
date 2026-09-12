"""
S-OHRP - Sectoral Orthogonal Hierarchical Risk Parity
=====================================================

Strict generalization of OHRP introduced in:

    PERGHER, K. G. R.; SOLDERA, J.; SCHARCANSKI, J.
    "Enhancing Orthogonal Hierarchical Risk Parity with Sectoral Information
    for Out-of-Sample Portfolio Allocation". Manuscript under review, 2026.

What it adds to OHRP
--------------------
1. Sector-constrained locality graph. The k-nearest-neighbour search runs only
   among assets of the same B3 economic sector, which injects an economic prior
   into the learned representation and blocks spurious cross-sector edges -
   the ones that hurt most in short estimation windows. Setting the sector
   partition to the whole universe recovers OHRP exactly, hence "strict
   generalization".

2. Unified delta*F selection. Any scalar portfolio objective F may drive the
   in-sample grid search, minimised or maximised. Two instances are used in the
   dissertation:

       optimize_metric="sharpe"     -> S-OHRP        (maximises in-sample Sharpe)
       optimize_metric="volatility" -> S-OHRP (Vol)  (minimises in-sample variance)

   Contrasting S-OHRP (Vol) against OHRP isolates the effect of the sectoral
   constraint; contrasting the two instances isolates the effect of the
   objective function. That is the ablation reported in the paper.

Usage
-----
    from SOHRP import SOHRP

    model   = SOHRP(codep="pearson", linkage="ward")
    weights = model.run(X=returns, L=labels, k=[1, 3, 5], d=[5, 10, 15],
                        r=[0.94, 0.98], optimize_metric="sharpe")

`L` is a single-column DataFrame named "label", indexed by ticker, holding the
sector of each asset.
"""

import pandas
import numpy
import traceback

from COLPP import COLPP
from HRP import HRP

class SOHRP():

    def __init__(self, codep:str = "pearson", linkage:str = "ward") -> None:
        self._options = { # COMO QUEREMOS CLUSTERIZAR
            "gnd"         : None,
            "NeighborMode": "Supervised", # SUPERVISIONADO (COLPP)
            "WeightMode"  : "HeatKernel",   # HEAT KERNEL
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
    
    def run(self, X, L, k:list = [5], d:list = [10], r:list = [0.9], optimize_metric:str = "volatility"):

        self.X = X.copy()
        self._options["t"] = COLPP.find_optimal(df = self.X)
        self.X = self.X.transpose()

        # Adjust Labels
        if type(L) != pandas.DataFrame:
            raise ValueError("Labels L must be a pandas DataFrame.")
        if (L.shape[1] != 1) or (L.columns[0] != "label"):
            raise ValueError("Labels L must have a single column named 'label'.")

        conv, self.L = SOHRP._convertLabel(lbs = L)

        # Adjust iterables
        if type(k) == int  : k = [k]
        if type(d) == int  : d = [d]
        if type(r) == float: r = [r]

        best_metric     = - numpy.inf
        best_combo      = {"d": 0, "k": 0, "r": 0}
        best_weights    = pandas.Series(numpy.nan, index = self.X.columns.tolist(), name = "weights")

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
                        continue
                    if optimize_metric in ['volatility', 'sharpe', 'return']:
                        if numpy.isfinite(metric) and metric > best_metric:
                            best_metric  = metric
                            best_combo   = {"d": di, "k": ki, "r": ri}
                            best_weights = weights.copy()

        self._bestHyperparams = best_combo

        if best_weights.isna().all():
            raise RuntimeError(
                "SOHRP.run: no (k, d, r) combination produced a valid portfolio "
                "(all attempts failed or yielded non-finite metrics)."
            )

        return best_weights

    def OrthoRedDim(self, X, L, k:int, d:int, r:float):

        self._options["k"]          = k
        self._options["ReducedDim"] = d
        self._options["PCARatio"]   = r

        # AFINIDADE
        W = COLPP.affinity_matrix(df = X, labels = L, ops = self._options, self_connection = False, olpp = False)

        D = COLPP.diagonal(affinity = W)

        X_std = (X.copy() - X.copy().mean(axis = 0))
        # DECOMPOSICAO SVD
        U, S, V = COLPP.SVD(X = X_std)

        U, S, V = COLPP.cut_on_ratio(U = U, V = V, S = S, pca_ratio = self._options["PCARatio"])

        # ESPAÇO ORTOGONAL
        eigen_vector = COLPP.build_weights(
            U = U, S = S, V = V, D = D, W = W, reduced_dim = self._options["ReducedDim"], bd = True
        )

        # PROJETA OS RETORNOS
        red_space = COLPP.project_data(X = X, P = eigen_vector, L = L)
        red_space = red_space[red_space.drop(["label"], axis = 1).std(axis = 1) != 0]

        portfolio = HRP(
            Y = red_space.drop(["label"], axis = 1).transpose().copy(),
            correlation_method = self._options["codependence"],
            linkage_method    = self._options["linkage"]
        )
        portfolio.run()
        weights_sohrp = portfolio.weights

        # NORMALIZA PARA PESO 1 (DESCONSIDERA PESOS MUITO PEQUENOS)
        weights_sohrp /= numpy.sum(weights_sohrp)
        inp = X.transpose().copy() * weights_sohrp.transpose()
        inp.dropna(axis = 1, how = "all", inplace = True)
        inp = inp.sum(axis = 1)

        return weights_sohrp, inp