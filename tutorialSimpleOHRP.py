"""
tutorialSimpleOHRP.py
=====================

Quick start: build ONE allocation, on ONE estimation window, with every strategy
compared in the dissertation, and inspect the resulting weight vectors.

This is the cheapest way to check that the environment is working before
committing to the full backtest of `tutorialSimulation.py`, which takes hours.

Run it from the repository root, so that the relative paths to `datasets/`
resolve:

    python tutorialSimpleOHRP.py

The file is organised in `#%%` cells: VS Code, Spyder and PyCharm all render
them as an interactive notebook, so you can also step through it cell by cell.
"""

#%%
# -----------------------------------------------------------------------------
# 1. Imports and data loading
# -----------------------------------------------------------------------------
import warnings

import numpy
import pandas

from OHRP import OHRP
from SOHRP import SOHRP
from HRP import HRP
from EW import EW
from Measures import Measures

warnings.filterwarnings("ignore")

PRICES_CSV = "datasets/BR_equity_closing_prices.csv"
SECTORS_XLSX = "datasets/SectoralBrazilianClassification.xlsx"

# Adjusted closing prices. The file also carries two benchmark columns (IBOV and
# SELIC) that are not investable assets, so they are dropped here.
prices = pandas.read_csv(PRICES_CSV, index_col=0, parse_dates=True)
prices.drop(columns=[c for c in ("IBOV", "SELIC") if c in prices.columns], inplace=True)

print(f"Price matrix: {prices.shape[0]} days x {prices.shape[1]} tickers")
print(f"From {prices.index[0].date()} to {prices.index[-1].date()}")


#%%
# -----------------------------------------------------------------------------
# 2. Pick one estimation window
# -----------------------------------------------------------------------------
# The dissertation starts the rolling window at index 2280 (2011-01-03) and uses
# WL * 252 trading days of history. Here we take a single one-year window so the
# example runs in seconds rather than hours.
INITIAL_TIME = 2280          # first rebalancing date used in the dissertation
WINDOW_LENGTH = int(1.0 * 252)   # WL = 1.0 year

window = prices.iloc[INITIAL_TIME - WINDOW_LENGTH:INITIAL_TIME, :].copy()

# Keep only tickers that are actually trading at the end of the window and that
# have at least 90% of valid observations inside it. This is the same liquidity
# filter applied in simulationFund.py.
active = ~pandas.isna(window.iloc[-1, :])
window = window.loc[:, active[active].index]
returns = window.pct_change()
returns[returns == 0] = numpy.nan                       # frozen quotes -> missing
returns.dropna(axis=0, how="all", inplace=True)
returns.dropna(axis=1, thresh=int(0.9 * WINDOW_LENGTH), inplace=True)
returns.fillna(0, inplace=True)
returns = returns.iloc[1:, :]

# Drop anything with zero volatility (Yahoo occasionally carries flat series).
volatility = returns.std(axis=0)
returns = returns[volatility[volatility > 0].index.tolist()]

print(f"Estimation window: {returns.shape[0]} days x {returns.shape[1]} assets")
print(f"Reference date (allocation is made here): {returns.index[-1].date()}")


#%%
# -----------------------------------------------------------------------------
# 3. Sector labels, required by S-OHRP
# -----------------------------------------------------------------------------
# S-OHRP restricts the locality graph to intra-sector edges, so it needs a
# single-column DataFrame named "label", indexed by ticker. Tickers are matched
# by their reduced (4-letter) form, e.g. PETR4 -> PETR.
classification = pandas.read_excel(SECTORS_XLSX, index_col=0)

labels = pandas.DataFrame("Unknown", index=returns.columns, columns=["label"])
for reduced in classification.REDUCED_TICKER.unique().tolist():
    matches = [t for t in returns.columns if t.startswith(reduced)]
    if matches:
        labels.loc[matches, "label"] = classification.loc[
            classification.REDUCED_TICKER == reduced, "SECTOR"
        ].values[0]

print(f"{labels.label.nunique()} sectors present in this window:")
print(labels.label.value_counts().to_string())


#%%
# -----------------------------------------------------------------------------
# 4. Hyperparameter grid
# -----------------------------------------------------------------------------
# The dissertation searches the full grid below at EVERY rebalancing. That is
# 7 x 7 x 5 = 245 projections per date per method, which is why the complete
# backtest is expensive. For this tutorial we use a coarse subset.
GRID_K = [1, 3, 5]        # nearest neighbours in the locality graph
GRID_D = [5, 10, 15]      # dimensionality of the projected subspace
GRID_R = [0.94, 0.98]     # variance retained by the PCA stage

# Full grid used in the dissertation, for reference:
#   list_of_k = [1, 3, 5, 7, 9, 11, 13]
#   list_of_d = [5, 10, 15, 20, 25, 30, 35]
#   list_of_r = [0.92, 0.94, 0.96, 0.98, 1.00]


#%%
# -----------------------------------------------------------------------------
# 5. Run every strategy on this window
# -----------------------------------------------------------------------------
weights = {}

# --- OHRP: unconstrained locality graph, in-sample volatility minimisation ----
ohrp = OHRP(codep="pearson", linkage="ward")
weights["OHRP"] = ohrp.run(X=returns.copy(), k=GRID_K, d=GRID_D, r=GRID_R)
print(f"OHRP selected (k, d, r) = {ohrp._bestHyperparams}")

# --- S-OHRP: sector-constrained graph, in-sample Sharpe maximisation ----------
sohrp = SOHRP(codep="pearson", linkage="ward")
weights["S-OHRP"] = sohrp.run(
    X=returns.copy(), L=labels.copy(),
    k=GRID_K, d=GRID_D, r=GRID_R,
    optimize_metric="sharpe",
)
print(f"S-OHRP selected (k, d, r) = {sohrp._bestHyperparams}")

# --- S-OHRP (Vol): same graph, variance objective ----------------------------
sohrp_vol = SOHRP(codep="pearson", linkage="ward")
weights["S-OHRP (Vol)"] = sohrp_vol.run(
    X=returns.copy(), L=labels.copy(),
    k=GRID_K, d=GRID_D, r=GRID_R,
    optimize_metric="volatility",
)
print(f"S-OHRP (Vol) selected (k, d, r) = {sohrp_vol._bestHyperparams}")

# --- HRP: the canonical benchmark, no projection -----------------------------
hrp = HRP(Y=returns.copy(), correlation_method="pearson", linkage_method="ward")
hrp.run()
weights["HRP"] = hrp.weights

# --- EW: 1/N ------------------------------------------------------------------
weights["EW"] = EW().run(X=returns.copy())

# RP (risk parity) is omitted here because it pulls in riskfolio-lib; see
# simulationFund.py for how it enters the full comparison.


#%%
# -----------------------------------------------------------------------------
# 6. Compare the resulting portfolios
# -----------------------------------------------------------------------------
# Two concentration measures are reported side by side, and they disagree by
# design. The asset-level Gini sees how spread the weights are across tickers;
# the Sectoral Gini (SGI) sees how spread they are across economic sectors. A
# portfolio can look diversified under the first and concentrated under the
# second, which is precisely the blind spot the SGI was proposed to cover.
summary = []
for name, w in weights.items():
    w = w / w.sum()
    summary.append({
        "Strategy": name,
        "Assets > 1%": int((w > 0.01).sum()),
        "Max weight": w.max(),
        "Gini": Measures.gini_index(series=w),
        "SGI": Measures.sectoral_gini_index(weights=w, labels=labels),
    })

summary = pandas.DataFrame(summary).set_index("Strategy")
print("\nIn-sample portfolio structure at "
      f"{returns.index[-1].date()}:\n")
print(summary.round(4).to_string())

#%%
# -----------------------------------------------------------------------------
# 7. Sector allocation of each strategy
# -----------------------------------------------------------------------------
# Aggregating the weights by sector makes the SGI ranking above concrete.
sector_mix = pandas.DataFrame({
    name: Measures.to_sectoral_weights(weights=w / w.sum(), labels=labels)
    for name, w in weights.items()
})
print("\nAggregate weight by sector:\n")
print((100 * sector_mix).round(1).to_string())

# %%
