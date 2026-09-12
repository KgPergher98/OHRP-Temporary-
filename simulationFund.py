import pandas
import datetime
import numpy
import warnings
import os

warnings.filterwarnings("ignore")

from OHRP import OHRP
from SOHRP import SOHRP
from HRP import HRP
from Measures import Measures
from RP import RP
from EW import EW

# Available hyperparameters configurations
list_of_d = [5,10,15,20,25,30,35]
list_of_k = [1,3,5,7,9,11,13]
list_of_r = [0.92, 0.94, 0.96, 0.98, 1.00]

def getLabels(df, rule = 'SECTOR'):
    exchangeLabels = pandas.read_excel('datasets/SectoralBrazilianClassification.xlsx', index_col = 0)
    print(exchangeLabels)
    tradableTickers = df.columns.tolist()
    labels = pandas.DataFrame(numpy.nan, index = tradableTickers, columns = ['label'])
    for red_ticker in exchangeLabels.REDUCED_TICKER.unique().tolist():
        fillList = [t for t in tradableTickers if t.startswith(red_ticker)]
        labels.loc[fillList,'label'] = exchangeLabels.loc[exchangeLabels.REDUCED_TICKER == red_ticker, rule].values[0]

    verifyLabels = labels.dropna(axis = 0).copy()
    if verifyLabels.shape[0] < df.shape[1]:
        missingTickers = [t for t in tradableTickers if t not in verifyLabels.index.tolist()]
        print(f'Warning: The following tickers are missing labels and will be assigned as "Unknown": {missingTickers}')
        for t in missingTickers:
            labels.loc[t,'label'] = 'Unknown'

    # Create Numerical Labels
    #labels['num_label'] = pandas.factorize(labels['label'])[0]

    return labels

# Extract the Dataset
X = pandas.read_csv('datasets/BR_equity_closing_prices.csv', index_col = 0, parse_dates=True)
X.index = pandas.to_datetime(X.index, format='%Y-%m-%d')

# Indices/Benchmarks
benchmarks = ['IBOV', 'SELIC']

Xindices = X[benchmarks].copy()

X.drop(columns=benchmarks, inplace = True)

# Separate Indices and Stocks
Xstocks = X.copy()
Xstocks.dropna(how='all', axis=0, inplace=True)
Xindices = Xindices.loc[Xstocks.index,:].copy()

# Secotral Data
sectoralLabels = getLabels(df = Xstocks)
#raise ValueError("The simulation is not ready yet. Please check the code and try again.")

def calculateData(series:pandas.Series, weights:pandas.Series, labels:pandas.DataFrame, risk_free_rate:pandas.Series, pre_trade_weights:pandas.Series) -> pandas.DataFrame:
    df = []
    df.append(Measures.sharpe_ratio(series = series, risk_free_rate = risk_free_rate))
    df.append(Measures.sortino_ratio(series = series, risk_free_rate = risk_free_rate))
    df.append(Measures.standard_deviation(series = series))
    df.append(Measures.mean_return(series = series))
    #df.append(Measures.cagr_return(series = series))
    df.append(Measures.drawdown(series = series))
    df.append(Measures.pain_index(series = series))
    df.append(Measures.sectoral_gini_index(weights = weights, labels = labels))
    df.append(Measures.gini_index(series = weights))
    df.append(Measures.turnover_ratio(pre_trade_weights = pre_trade_weights, post_trade_weights = weights))
    df = pandas.DataFrame(df, index = ['Sharpe Ratio', 'Sortino Ratio', 'Standard Deviation', 'Mean Return', 'Drawdown', 'Pain Index', 'Sectoral Gini Index', 'Gini Index', 'Turnover Ratio'], columns = ['Value'])
    return df.transpose()

# Time we want to start the rolling window
# Vary as you want or need
#initialTime = 1536 # starts at 2008-01-02
initialTime = 2280 # starts at 2011-01-03

#list_of_d = [20]
#list_of_k = [2]
#list_of_r = [0.98]

transaction_cost = 0.03/100

results_dir = 'results/'

# -----------------------------------------------------------------------------
# Variantes S-OHRP: nome da coluna de saída -> métrica otimizada em SOHRP.run
# -----------------------------------------------------------------------------
SOHRP_VARIANTS = {
    "SOHRP":    "sharpe",        # S-OHRP "puro"
    "SOHRPvol": "volatility",    # S-OHRP com vol-target -> S-OHRP (Vol) no paper
}

# -----------------------------------------------------------------------------
# Definição dos experimentos. Saídas por experimento (em results/):
#   <name>_Ms_WL_<WL>.csv  -> medidas por carteira (por rebalanceamento)
#   <name>_Ts_WL_<WL>.csv  -> séries temporais de retorno diário
# -----------------------------------------------------------------------------
EXPERIMENTS = {
    1: {  # com S-OHRP (SOHRP + SOHRPvol), janelas curtas
        "name":           "ExperimentoSOHRP",
        "strategies":     ["SOHRP", "SOHRPvol", "OHRP", "HRP", "EW", "RP"],
        "window_lengths": [round(0.2 * i, 2) for i in range(1, 11)],   # 0.20 .. 2.00
    },
    2: {  # sem S-OHRP (apenas referências), janelas longas
        "name":           "ExperimentoOHRP",
        "strategies":     ["OHRP", "HRP", "EW", "RP"],
        "window_lengths": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    },
}

# >>> ESCOLHA AQUI: 1, 2 ou "both" (gera ambos os experimentos) <<<
EXPERIMENT_TO_RUN = "both"


def run_experiment(name, strategies, window_lengths):
    """Roda a simulação de um experimento e grava <name>_Ms/Ts_WL_<WL>.csv."""
    measures_prefix   = f'{name}_Ms'
    timeseries_prefix = f'{name}_Ts'

    print(f"\n===== {name}: {len(strategies)} estratégias, {len(window_lengths)} WL =====")

    # Windows we want to test
    for WL in window_lengths:

        if f'{measures_prefix}_WL_{WL:.2f}.csv' in os.listdir(results_dir):
            print(f"Results for WL = {WL} already exist. Skipping this configuration.")
            continue

        try:

            # Out of Sample Portfolio Returns
            portfolios = {
                "weights": {key: pandas.DataFrame() for key in strategies},
                "final_weights": {key: pandas.DataFrame() for key in strategies},
                "out_sample_returns": {key: pandas.Series() for key in strategies},
                "out_sample_returns_with_cost": {key: pandas.Series() for key in strategies},
                "out_sample_measures": {key: pandas.DataFrame() for key in strategies}
            }

            print(f"Processing WL = {WL}")

            t  = initialTime

            # Convert window length in days
            windowLength = int(WL * 252)

            # How much day to hold the portfolio
            #holdingPeriod = 21
            holdingPeriod = 42

            while t < (Xstocks.shape[0] - holdingPeriod):

                # In-sample Data
                Xn = Xstocks.iloc[(t-windowLength):t,:].copy()
                # Out-of-sample Data
                Yn = Xstocks.iloc[t-1:t+holdingPeriod-1,:].copy()

                # Tickers with data on t
                activeFilter = ~pandas.isna(Xn.iloc[-1,:])
                activeFilter = activeFilter[activeFilter].index.tolist()

                Xn = Xn[activeFilter]
                Xn = Xn.pct_change()
                Xn[Xn == 0] = numpy.nan
                Xn.dropna(axis = 0, how = 'all', inplace = True)
                Xn.dropna(axis = 1, thresh = int(0.9 * windowLength), inplace = True)
                Xn.fillna(0, inplace = True)

                # Volatility filter - Some prices are manteined constant at YaHoo
                volatility = Xn.std(axis = 0)

                volatility = volatility[volatility > 0]
                Xn = Xn[volatility.index.tolist()]

                # Out-of-sample data
                Yn = Yn[Xn.columns.tolist()]
                Yn = Yn.pct_change().fillna(0)
                # Benchmark
                SELIC = Xindices.loc[Yn.index, 'SELIC'].copy()
                SELIC = SELIC.pct_change().fillna(0)
                # Drop first row (only zeros)
                Xn = Xn.iloc[1:,:].copy()
                Yn = Yn.iloc[1:,:].copy()
                SELIC = SELIC.iloc[1:].copy()

                # Reference date - time stamp of operation
                referenceDate = Xn.index[-1]
                print(f"Processing t = {t} / {Xstocks.shape[0]} - Date: {referenceDate.date()}")

                temporaryData = {
                    "weights": {key: None for key in strategies},
                    "out_sample_returns": {key: None for key in strategies},
                    "out_sample_returns_with_cost": {key: None for key in strategies},
                    "measurements": {key: None for key in strategies},
                    "final_weights": {key: None for key in strategies}
                }

                #sectoralLabels.label = 'any'
                #raise ValueError("The simulation is not ready yet. Please check the code and try again.")

                # Perform S-OHRP variants (SOHRP / SOHRPvol) present in this experiment
                for sohrp_name, optimize_metric in SOHRP_VARIANTS.items():
                    if sohrp_name in strategies:
                        sohrp = SOHRP(codep = "pearson", linkage = "ward")
                        temporaryData['weights'][sohrp_name] = sohrp.run(
                            X = Xn.copy(),
                            L = sectoralLabels.loc[Xn.columns.tolist(), :].copy(),
                            k = list_of_k,
                            d = list_of_d,
                            r = list_of_r,
                            optimize_metric = optimize_metric
                        )

                if "OHRP" in strategies:
                    ohrp = OHRP(codep = "pearson", linkage = "ward")
                    temporaryData['weights']['OHRP'] = ohrp.run(
                        X = Xn.copy(),
                        k = list_of_k,
                        d = list_of_d,
                        r = list_of_r
                    )

                # Perform HRP
                if "HRP" in strategies:
                    hrp = HRP(
                        Y = Xn.copy(),
                        correlation_method = "pearson",
                        linkage_method = "ward"
                    )
                    hrp.run()
                    temporaryData['weights']['HRP'] = hrp.weights

                # Perform RP
                if "RP" in strategies:
                    rp = RP(method_cov="ledoit")
                    temporaryData['weights']['RP'] = rp.run(
                        X = Xn.copy()
                    )

                # Perform EW
                if "EW" in strategies:
                    ew = EW()
                    temporaryData['weights']['EW'] = ew.run(
                        X = Xn.copy()
                    )

                for strategy in strategies:

                    # Out-of-sample Portfolio Returns - Individual composition returns
                    timeSeries = (1 + Yn.copy()).cumprod()
                    # Apply weights
                    timeSeries *= temporaryData['weights'][strategy]
                    # Final Weights - Distribution of weights by ticker
                    finalWeights = timeSeries.iloc[-1,:].copy()
                    finalWeights = finalWeights / finalWeights.sum()
                    temporaryData['final_weights'][strategy] = finalWeights
                    # Portfolio Returns
                    timeSeries = timeSeries.sum(axis = 1)
                    returns = timeSeries.pct_change()
                    # Assume the sum of the weights (i.e sum_weights = 1) is invested at t-1, so the return at t is 1 - return of the portfolio
                    returns.iloc[0] = timeSeries.iloc[0] - 1

                    try:
                        lastFinalWeights = portfolios['final_weights'][strategy].iloc[:,-1].copy()
                    except IndexError:
                        # First Iteration - starting the portfolio with initia capital = 1
                        lastFinalWeights = pandas.Series(0, index = temporaryData['weights'][strategy].index)

                    temporaryData["out_sample_returns"][strategy] = returns

                    # Necessary to apply costs - the more a strategy gains the more it costs will be relevant
                    if portfolios['out_sample_returns_with_cost'][strategy].empty:
                        accumulatedReturns = 1
                    else:
                        accumulatedReturns = (1 + portfolios['out_sample_returns_with_cost'][strategy]).cumprod().iloc[-1]

                    measures = calculateData(
                        series = temporaryData['out_sample_returns'][strategy],
                        weights = temporaryData['weights'][strategy],
                        pre_trade_weights = lastFinalWeights,
                        labels = sectoralLabels,
                        risk_free_rate = SELIC
                    )

                    # Unbiasing the initial turnover - all portfolios start from zero in the same condition, so the first turnover is not relevant for comparison
                    cost = 2 * measures['Turnover Ratio'].values[0] * transaction_cost * accumulatedReturns
                    if t == initialTime:
                        measures['Turnover Ratio'] = numpy.nan

                    temporaryData["out_sample_returns_with_cost"][strategy] = temporaryData['out_sample_returns'][strategy].copy()
                    temporaryData["out_sample_returns_with_cost"][strategy].iloc[0] -= cost

                    measures['EntryDateRef'] = referenceDate
                    measures['SortieDateRef'] = Yn.index[-1]
                    measures['WindowLength'] = WL
                    measures['RiskFreeRate'] = SELIC.mean() * 252
                    measures['Strategy'] = strategy
                    temporaryData['measurements'][strategy] = measures

                    portfolios['out_sample_returns'][strategy] = pandas.concat(
                        [portfolios['out_sample_returns'][strategy], temporaryData["out_sample_returns"][strategy]], axis = 0
                    )
                    portfolios['out_sample_returns_with_cost'][strategy] = pandas.concat(
                        [portfolios['out_sample_returns_with_cost'][strategy], temporaryData["out_sample_returns_with_cost"][strategy]], axis = 0
                    )
                    portfolios['out_sample_measures'][strategy] = pandas.concat(
                        [portfolios['out_sample_measures'][strategy], temporaryData["measurements"][strategy]], axis = 0, ignore_index = True
                    )

                    wgs = temporaryData['weights'][strategy].copy()
                    wgs.name = str(referenceDate)
                    portfolios['weights'][strategy] = pandas.concat(
                        [portfolios['weights'][strategy], wgs], axis = 1
                    )

                    portfolios['final_weights'][strategy] = pandas.concat(
                        [portfolios['final_weights'][strategy], temporaryData['final_weights'][strategy]], axis = 1
                    )

                t += holdingPeriod

            datasetMeasures = pandas.DataFrame()
            datasetTimeSeries = pandas.DataFrame()

            for strategy in strategies:
                auxM = portfolios['out_sample_measures'][strategy].copy()
                auxP = pandas.concat(
                    [portfolios['out_sample_returns'][strategy], portfolios['out_sample_returns_with_cost'][strategy]], axis=1
                )
                auxP.columns = [f'Returns {strategy}', f'Returns with Cost {strategy}']

                datasetMeasures = pandas.concat([datasetMeasures, auxM], axis = 0, ignore_index = True)
                datasetTimeSeries = pandas.concat([datasetTimeSeries, auxP], axis = 1)

            datasetMeasures.to_csv(f'{results_dir}{measures_prefix}_WL_{WL:.2f}.csv', index = False)
            datasetTimeSeries.to_csv(f'{results_dir}{timeseries_prefix}_WL_{WL:.2f}.csv', index = True)
        except Exception as e:
            print(f"An error occurred for WL = {WL}. Error: {str(e)}")


# -----------------------------------------------------------------------------
# Dispatcher: roda o(s) experimento(s) selecionado(s) em EXPERIMENT_TO_RUN
# -----------------------------------------------------------------------------
if EXPERIMENT_TO_RUN == "both":
    experiments_to_run = [1, 2]
else:
    experiments_to_run = [EXPERIMENT_TO_RUN]

for _exp_id in experiments_to_run:
    _cfg = EXPERIMENTS[_exp_id]
    run_experiment(
        name           = _cfg["name"],
        strategies     = _cfg["strategies"],
        window_lengths = _cfg["window_lengths"],
    )
