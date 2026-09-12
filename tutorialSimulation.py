"""
tutorialSimulation.py
=====================

How the out-of-sample backtest works, and how to reproduce the dissertation
results.

The production driver is `simulationFund.py`; this tutorial explains what it
does, then runs a deliberately small version of the same loop so you can watch
the mechanics end to end in a couple of minutes instead of a couple of days.

    python tutorialSimulation.py

--------------------------------------------------------------------------
Reproducing the dissertation in full
--------------------------------------------------------------------------
Edit nothing and simply run:

    python simulationFund.py

That regenerates every CSV under `results/`, for both experiments:

    ExperimentoSOHRP   S-OHRP, S-OHRP (Vol), OHRP, HRP, EW, RP
                       WL = 0.20 ... 2.00 years, step 0.20   (short windows)

    ExperimentoOHRP    OHRP, HRP, EW, RP
                       WL = 1.0 ... 5.0 years, step 0.5      (long windows)

Two files are written per window length:

    <experiment>_Ms_WL_<WL>.csv   one row per rebalanced portfolio, with all
                                  nine performance and concentration measures
    <experiment>_Ts_WL_<WL>.csv   daily out-of-sample return series, gross and
                                  net of transaction costs

`simulationFund.py` skips any window length whose `_Ms_` file already exists,
so an interrupted run can simply be relaunched.

Be warned: the full grid is 7 x 7 x 5 = 245 projections per method per
rebalancing date, across 89 rebalancings and 19 window lengths. Expect hours,
not minutes.

--------------------------------------------------------------------------
Protocol (identical in both experiments)
--------------------------------------------------------------------------
    Universe          158 Brazilian stocks (IBOV + SMLL), 10 B3 sectors
    Period            January 2011 to February 2026
    Rolling window    WL * 252 trading days
    Holding period    42 trading days (~2 months) -> 89 out-of-sample portfolios
    Liquidity filter  >= 90% valid observations inside the window
    Transaction cost  3 basis points on traded volume
    Risk-free rate    SELIC
"""

#%%
# -----------------------------------------------------------------------------
# 1. Setup: same data preparation used by simulationFund.py
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

prices = pandas.read_csv(
    "datasets/BR_equity_closing_prices.csv", index_col=0, parse_dates=True
)

# SELIC is carried inside the price file and serves as the risk-free rate.
benchmarks = ["IBOV", "SELIC"]
indices = prices[benchmarks].copy()
stocks = prices.drop(columns=benchmarks)
stocks.dropna(how="all", axis=0, inplace=True)
indices = indices.loc[stocks.index, :]

classification = pandas.read_excel(
    "datasets/SectoralBrazilianClassification.xlsx", index_col=0
)
sector_labels = pandas.DataFrame(
    "Unknown", index=stocks.columns, columns=["label"]
)
for reduced in classification.REDUCED_TICKER.unique().tolist():
    matches = [t for t in stocks.columns if t.startswith(reduced)]
    if matches:
        sector_labels.loc[matches, "label"] = classification.loc[
            classification.REDUCED_TICKER == reduced, "SECTOR"
        ].values[0]


#%%
# -----------------------------------------------------------------------------
# 2. Reduced-scale backtest
# -----------------------------------------------------------------------------
# Same loop as simulationFund.py, but with a coarse hyperparameter grid and only
# a handful of rebalancings, so that it finishes quickly.
INITIAL_TIME = 2280            # 2011-01-03
WINDOW_LENGTH = int(1.0 * 252)  # WL = 1.0 year
HOLDING_PERIOD = 42            # trading days between rebalancings
N_REBALANCINGS = 4             # dissertation uses 89
TRANSACTION_COST = 0.03 / 100  # 3 basis points

GRID_K, GRID_D, GRID_R = [3, 5], [10, 15], [0.98]

STRATEGIES = ["S-OHRP", "OHRP", "HRP", "EW"]

returns_by_strategy = {s: pandas.Series(dtype=float) for s in STRATEGIES}
measures_rows = []
previous_final_weights = {s: None for s in STRATEGIES}

t = INITIAL_TIME
for step in range(N_REBALANCINGS):

    # --- in-sample window and out-of-sample holding period -------------------
    Xn = stocks.iloc[t - WINDOW_LENGTH:t, :].copy()
    Yn = stocks.iloc[t - 1:t + HOLDING_PERIOD - 1, :].copy()

    active = ~pandas.isna(Xn.iloc[-1, :])
    Xn = Xn[active[active].index.tolist()]
    Xn = Xn.pct_change()
    Xn[Xn == 0] = numpy.nan
    Xn.dropna(axis=0, how="all", inplace=True)
    Xn.dropna(axis=1, thresh=int(0.9 * WINDOW_LENGTH), inplace=True)
    Xn.fillna(0, inplace=True)

    volatility = Xn.std(axis=0)
    Xn = Xn[volatility[volatility > 0].index.tolist()]

    Yn = Yn[Xn.columns.tolist()].pct_change().fillna(0)
    selic = indices.loc[Yn.index, "SELIC"].pct_change().fillna(0)

    Xn, Yn, selic = Xn.iloc[1:, :], Yn.iloc[1:, :], selic.iloc[1:]
    reference_date = Xn.index[-1]
    print(f"[{step + 1}/{N_REBALANCINGS}] {reference_date.date()} "
          f"- {Xn.shape[1]} assets")

    # --- allocate ------------------------------------------------------------
    labels_now = sector_labels.loc[Xn.columns.tolist(), :].copy()
    w = {}
    w["S-OHRP"] = SOHRP().run(X=Xn.copy(), L=labels_now.copy(), k=GRID_K,
                              d=GRID_D, r=GRID_R, optimize_metric="sharpe")
    w["OHRP"] = OHRP().run(X=Xn.copy(), k=GRID_K, d=GRID_D, r=GRID_R)
    hrp = HRP(Y=Xn.copy(), correlation_method="pearson", linkage_method="ward")
    hrp.run()
    w["HRP"] = hrp.weights
    w["EW"] = EW().run(X=Xn.copy())

    # --- evaluate out of sample ---------------------------------------------
    for strategy in STRATEGIES:
        series = (1 + Yn.copy()).cumprod() * w[strategy]
        final_weights = series.iloc[-1, :] / series.iloc[-1, :].sum()
        series = series.sum(axis=1)

        period_returns = series.pct_change()
        period_returns.iloc[0] = series.iloc[0] - 1

        pre_trade = previous_final_weights[strategy]
        if pre_trade is None:
            pre_trade = pandas.Series(0, index=w[strategy].index)

        turnover = Measures.turnover_ratio(
            pre_trade_weights=pre_trade, post_trade_weights=w[strategy]
        )
        # Costs are charged on the first day of the holding period. The very
        # first rebalancing is skipped because every strategy starts from cash.
        if step > 0:
            period_returns.iloc[0] -= 2 * turnover * TRANSACTION_COST

        measures_rows.append({
            "Date": reference_date,
            "Strategy": strategy,
            "Sharpe": Measures.sharpe_ratio(series=period_returns,
                                            risk_free_rate=selic),
            "Volatility": Measures.standard_deviation(series=period_returns),
            "Drawdown": Measures.drawdown(series=period_returns),
            "Gini": Measures.gini_index(series=w[strategy]),
            "SGI": Measures.sectoral_gini_index(weights=w[strategy],
                                                labels=labels_now),
            "Turnover": numpy.nan if step == 0 else turnover,
        })

        returns_by_strategy[strategy] = pandas.concat(
            [returns_by_strategy[strategy], period_returns]
        )
        previous_final_weights[strategy] = final_weights

    t += HOLDING_PERIOD


#%%
# -----------------------------------------------------------------------------
# 3. Results of the reduced run
# -----------------------------------------------------------------------------
measures = pandas.DataFrame(measures_rows)

print("\nAverage across the simulated holding periods:\n")
print(measures.groupby("Strategy")[
    ["Sharpe", "Volatility", "Drawdown", "Gini", "SGI", "Turnover"]
].mean().round(4).to_string())

print("\nCumulative out-of-sample return (net of costs):\n")
for strategy in STRATEGIES:
    total = (1 + returns_by_strategy[strategy]).prod() - 1
    print(f"    {strategy:<8} {100 * total:7.2f}%")

print(
    "\nThese numbers come from 4 rebalancings and a coarse grid, so they are a "
    "smoke test, not a result.\nThe figures reported in the dissertation come "
    "from the full run of simulationFund.py."
)

# %%
