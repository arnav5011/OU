from calibrate.baskets import Basket
from utils.price_data_utils import get_time_series
import pandas as pd
import numpy as np
import statsmodels.api as sm

class BackTest:
    def __init__(self, trading_results: dict, start_date, end_date, baskets: list[Basket], 
                 initial_capital=1e7, benchmark="SPY", risk_free_rate=0.5, year_convention=252):
        self.trading_results = trading_results
        self.start_date = start_date
        self.end_date = end_date
        
        self.baskets = baskets
        self.initial_capital = float(initial_capital)
        self.year_convention = year_convention
        
        self._daily_returns = self.trading_results["portfolio_returns"]

        self._total_baskets = len(baskets)
        ticker_price_data_list = {}
        for basket in baskets:
            cluster_id = basket.cluster_id
            coint_type = basket.cointegration_result.type
            if coint_type == "Engle-Granger":
                tickers = [basket.cointegration_result.ticker_1, basket.cointegration_result.ticker_2]
            elif coint_type == "Johansen":
                tickers = basket.cointegration_result.ticker_list
            else:
                raise ValueError("Invalid Parameters")
            ticker_price_data_list[cluster_id] = get_time_series(
                tickers, self.start_date, self.end_date
            )

        self._benchmark_price_data = get_time_series([benchmark], self.start_date, self.end_date)
        self._benchmark_returns = self._benchmark_price_data.pct_change().dropna()

        returns = pd.concat([self._daily_returns, self._benchmark_returns], axis=1, join="inner").dropna()
        returns.columns = ["portfolio", "benchmark"]
        self._daily_returns = returns["portfolio"]
        self._benchmark_returns = returns["benchmark"]

        self.risk_free_rate = risk_free_rate / 100 / year_convention

        self.alpha = None
        self.beta = None
        self.sharpe_ratio = None
        self.sortino_ratio = None
        self.drawdown = None
        self.win_rate = None

    def _CAPM_parameters(self):
        y = self._daily_returns - self.risk_free_rate
        X = self._benchmark_returns - self.risk_free_rate
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        alpha = model.params["const"]
        beta = model.params["benchmark"]
        self.alpha, self.beta = alpha, beta
        return alpha, beta

    def _get_alpha(self):
        if self.alpha is None:
            self._CAPM_parameters()
        return self.alpha * self.year_convention

    def _get_beta(self):
        if self.beta is None:
            self._CAPM_parameters()
        return self.beta

    def _get_drawdown(self):
        cumulative = (1 + self._daily_returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        self.drawdown = drawdown.min()
        return self.drawdown

    def _get_sharpe_ratio(self):
        excess_returns = self._daily_returns - self.risk_free_rate
        sr = (excess_returns.mean() / excess_returns.std()) * np.sqrt(self.year_convention) if excess_returns.std() != 0 else 0
        self.sharpe_ratio = sr
        return sr

    def _get_sortino_ratio(self):
        excess_returns = self._daily_returns - self.risk_free_rate
        negative_returns = excess_returns[excess_returns < 0]
        downside_deviation = negative_returns.std()
        sr = (excess_returns.mean() / downside_deviation) * np.sqrt(self.year_convention) if downside_deviation != 0 else 0
        self.sortino_ratio = sr
        return sr

    def _get_win_rate(self):
        wins = (self._daily_returns > 0).sum()
        total = len(self._daily_returns)
        self.win_rate = wins / total if total > 0 else np.nan
        return self.win_rate
    
    def summary(self):
        return {
            "alpha": self._get_alpha(),
            "beta": self._get_beta(),
            "sharpe_ratio": self._get_sharpe_ratio(),
            "sortino_ratio": self._get_sortino_ratio(),
            "drawdown": self._get_drawdown(),
            "win_rate": self._get_win_rate(),
        }

