"""
Measures - out-of-sample performance and concentration metrics
==============================================================

All metrics operate on daily returns with an annualization factor of 252.

Performance
    sharpe_ratio, sortino_ratio, mean_return, standard_deviation
Drawdown
    drawdown (maximum), pain_index (mean depth of the drawdown series)
Concentration and activity
    gini_index           -> Gini over individual asset weights
    sectoral_gini_index  -> Gini over sector-level aggregate weights (SGI)
    turnover_ratio       -> 0.5 * sum |w_post - w_pre|

The Sectoral Gini Index is a contribution of this research. It captures
cross-sector concentration that the asset-level Gini cannot see: a portfolio may
be spread over many tickers and still have its weight clustered in a single
economic sector.
"""

import pandas
import numpy

class Measures():

    def turnover_ratio(pre_trade_weights: pandas.DataFrame, post_trade_weights: pandas.DataFrame) -> float:
        """
        Calculate the Turnover Ratio between pre-trade and post-trade weights.
        
        Parameters:
        -----------
        pre_trade_weights : pd.DataFrame
            A DataFrame where each column represents the weights of assets before trading.
        post_trade_weights : pd.DataFrame
            A DataFrame where each column represents the weights of assets after trading.
        
        Returns:
        --------
        float
            The Turnover Ratio
        """

        if type(pre_trade_weights) is not pandas.Series:
            raise ValueError("pre_trade_weights must be a pandas Series.")
        if type(post_trade_weights) is not pandas.Series:
            raise ValueError("post_trade_weights must be a pandas Series.")
        
        aux = pandas.concat([pre_trade_weights, post_trade_weights], axis=1).fillna(0)
        aux.columns = ['pre_trade', 'post_trade']
        
        # Calculate absolute differences
        abs_diff = (aux.post_trade - aux.pre_trade).abs()
        
        # Sum of absolute differences for each time period
        sum_abs_diff = abs_diff.sum(axis=0)

        # Turnover ratio is half the sum of absolute differences
        turnover_ratio = 0.5 * sum_abs_diff
        
        return float(turnover_ratio)

    def to_sectoral_weights(weights: pandas.Series, labels: pandas.DataFrame) -> pandas.Series:
        df = pandas.DataFrame(weights.copy())
        df.columns = ['weights']
        df['labels'] = labels.loc[df.index, 'label'].values

        df = pandas.pivot_table(
            data    = df,
            index   = 'labels',
            values  = 'weights',
            aggfunc = 'sum'
        )
        df.index = df.index.tolist()
        return df['weights']

    def gini_index(series: pandas.Series) -> float:
        """
        Calculate the Gini index for a pandas Series.
        
        The series is normalized to sum to 1 before calculating the Gini index.
        Returns a value between 0 (perfect equality) and 1 (perfect inequality).
        
        Parameters:
        -----------
        series : pd.Series
            A pandas Series with numeric values (should be non-negative for meaningful results)
        
        Returns:
        --------
        float
            The Gini index, a value between 0 and 1
        """
        # Remove NaN values and convert to numpy array
        values = series.dropna().values
        
        # Handle edge cases
        if len(values) == 0:
            return 0.0
        if len(values) == 1:
            return 0.0
        
        # Normalize to sum to 1
        total = values.sum()
        if total == 0:
            return 0.0
        
        proportions = values / total
        
        # Sort the proportions
        sorted_proportions = numpy.sort(proportions)
        
        # Calculate Gini index using the standard formula
        n = len(sorted_proportions)
        index = numpy.arange(1, n + 1)
        gini = (2 * numpy.sum(index * sorted_proportions)) / (n * numpy.sum(sorted_proportions)) - (n + 1) / n
        
        return float(gini)

    def sectoral_gini_index(weights: pandas.Series, labels: pandas.DataFrame) -> float:
        sectoral_weights = Measures.to_sectoral_weights(weights = weights, labels = labels)
        return Measures.gini_index(series = sectoral_weights)

    def sharpe_ratio(series: pandas.Series, risk_free_rate: float = 0.0) -> float:
        """
        Calculate the annualized Sharpe ratio for a pandas Series of returns.
        
        Parameters:
        -----------
        series : pd.Series
            A pandas Series with periodic returns (e.g., daily returns)
        risk_free_rate : float
            The risk-free rate as a decimal (default is 0.0)
        
        Returns:
        --------
        float
            The annualized Sharpe ratio
        """
        # Remove NaN values
        returns = series.dropna()
        
        # Calculate excess returns
        excess_returns = returns - risk_free_rate  # Assuming daily returns
        
        # Calculate mean and standard deviation of excess returns
        mean_excess_return = excess_returns.mean()
        std_excess_return = returns.std()
        
        # Handle edge case where std is zero
        if std_excess_return == 0:
            return 0.0
        
        # Calculate annualized Sharpe ratio
        sharpe_ratio = (mean_excess_return / std_excess_return) * numpy.sqrt(252)  # Annualization factor for daily returns
        
        return float(sharpe_ratio)

    def sortino_ratio(series: pandas.Series, risk_free_rate: float = 0.0) -> float:
        """
        Calculate the annualized Sortino ratio for a pandas Series of returns.
        
        Parameters:
        -----------
        series : pd.Series
            A pandas Series with periodic returns (e.g., daily returns)
        risk_free_rate : float
            The risk-free rate as a decimal (default is 0.0)
        
        Returns:
        --------
        float
            The annualized Sortino ratio
        """
        # Remove NaN values
        returns = series.dropna()
        
        # Calculate excess returns
        excess_returns = returns - risk_free_rate  # Assuming daily returns
        
        # Calculate mean of excess returns
        mean_excess_return = excess_returns.mean()
        
        # Calculate downside deviation
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return 0.0
        downside_deviation = numpy.sqrt((downside_returns ** 2).mean())
        
        # Handle edge case where downside deviation is zero
        if downside_deviation == 0:
            return 0.0
        
        # Calculate annualized Sortino ratio
        sortino_ratio = (mean_excess_return / downside_deviation) * numpy.sqrt(252)  # Annualization factor for daily returns
        
        return float(sortino_ratio)

    def standard_deviation(series: pandas.Series) -> float:
        """
        Calculate the annualized standard deviation for a pandas Series of returns.
        
        Parameters:
        -----------
        series : pd.Series
            A pandas Series with periodic returns (e.g., daily returns)
        
        Returns:
        --------
        float
            The annualized standard deviation
        """
        # Remove NaN values
        returns = series.dropna()
        
        # Calculate standard deviation
        std_dev = returns.std()
        
        # Annualize the standard deviation
        annualized_std_dev = std_dev * numpy.sqrt(252)  # Annualization factor for daily returns
        
        return float(annualized_std_dev)

    def mean_return(series: pandas.Series) -> float:
        """
        Calculate the annualized mean return for a pandas Series of returns.
        
        Parameters:
        -----------
        series : pd.Series
            A pandas Series with periodic returns (e.g., daily returns)
        
        Returns:
        --------
        float
            The annualized mean return
        """
        # Remove NaN values
        returns = series.dropna()
        
        # Calculate mean return
        mean_ret = returns.mean()
        
        # Annualize the mean return
        annualized_mean_ret = mean_ret * 252  # Annualization factor for daily returns
        
        return float(annualized_mean_ret)

    def cagr_return(series: pandas.Series) -> float:
        """
        Calculate the Compound Annual Growth Rate (CAGR) for a pandas Series of returns.
        
        Parameters:
        -----------
        series : pd.Series
            A pandas Series with periodic returns (e.g., daily returns)
        
        Returns:
        --------
        float
            The CAGR
        """
        # Remove NaN values
        returns = series.dropna()
        
        # Calculate total number of periods
        n_periods = len(returns)
        
        # Calculate cumulative return
        cumulative_return = (1 + returns).prod() - 1
        
        # Calculate CAGR
        cagr = (1 + cumulative_return) ** (252 / n_periods) - 1  # Annualization factor for daily returns
        
        return float(cagr)

    def drawdown(series: pandas.Series) -> float:
        """
        Calculate the maximum drawdown for a pandas Series of returns.
        
        Parameters:
        -----------
        series : pd.Series
            A pandas Series with periodic returns (e.g., daily returns)
        
        Returns:
        --------
        float
            The maximum drawdown
        """
        # Remove NaN values
        returns = series.dropna()
        
        # Calculate cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cumulative_returns.cummax()
        
        # Calculate drawdowns
        drawdowns = (cumulative_returns - running_max) / running_max
        
        # Get maximum drawdown
        max_drawdown = drawdowns.min()
        
        return float(max_drawdown)

    def pain_index(series: pandas.Series) -> float:
        """
        Calculate the Pain Index for a pandas Series of returns.
        
        Parameters:
        -----------
        series : pd.Series
            A pandas Series with periodic returns (e.g., daily returns)
        
        Returns:
        --------
        float
            The Pain Index
        """
        # Remove NaN values
        returns = series.dropna()
        
        # Calculate cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cumulative_returns.cummax()
        
        # Calculate drawdowns
        drawdowns = (cumulative_returns - running_max) / running_max
        
        # Calculate Pain Index as the average of absolute drawdowns
        pain_index = -drawdowns[drawdowns < 0].mean()
        
        return float(pain_index)
