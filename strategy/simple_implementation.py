import numpy as np
import pandas as pd
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from calibrate.baskets import Basket
from calibrate.cointegration import PriceData
from utils import ou_trading_utils

class OUTrader:
    def __init__(self, start_date, end_date, baskets: list[Basket], initial_capital = 1e7):
        self.start_date = start_date
        self.end_date = end_date

        ticker_list = []
        for basket in baskets:
            cointegrated_results = basket.cointegration_result
            if cointegrated_results.type == "Engle-Granger":
                ticker_list.append(cointegrated_results.ticker_1)
                ticker_list.append(cointegrated_results.ticker_2)
            elif cointegrated_results.type == "Johansen":
                for ticker in cointegrated_results.ticker_list:
                    ticker_list.append(ticker)
            else:
                raise ValueError("Invalid integration test")
        self.price_data = PriceData.get_time_series(ticker_list, start_date, end_date)
        self.log_price_data = np.log(self.price_data)
        self.baskets = baskets
        self.initial_capital = float(initial_capital)
    
    def run(self, entry_z=3.0, exit_z=0.1, use_ou_params=True, rolling_window=20):
        prices = self.price_data
        n_baskets = len(self.baskets)
        capital_per_basket = self.initial_capital / n_baskets

        portfolio_pnl = pd.Series(0.0, index=prices.index)
        basket_results = {}

        for basket in self.baskets:
    
            spread = self._compute_spread(prices, basket)

            if use_ou_params:
                ou = basket.params
                zscore = (spread - ou.mu) / ou.sigma
            else:
                zscore = ou_trading_utils.calculate_zscore(spread, window=rolling_window)

            # --- Generate signals
            signals = ou_trading_utils.generate_ou_signals(zscore, entry_threshold=entry_z, exit_threshold=exit_z)

            # --- Spread returns
            spread_ret = spread.diff().fillna(0)

            # --- PnL
            pnl = signals.shift(1).fillna(0) * spread_ret * capital_per_basket
            cum_pnl = pnl.cumsum()

            basket_results[basket.cluster_id] = {
                "spread": spread,
                "zscore": zscore,
                "signals": signals,
                "pnl": pnl,
                "cum_pnl": cum_pnl
            }

            portfolio_pnl += pnl

        results = {
            "portfolio_pnl": portfolio_pnl,
            "portfolio_cum_pnl": portfolio_pnl.cumsum(),
            "basket_results": basket_results
        }

        return results

    def _compute_spread(self, prices, basket: Basket):
        coint = basket.cointegration_result
        if coint.type == "Engle-Granger":
            return ou_trading_utils.ou_spread_engle(
                prices[coint.ticker_1],
                prices[coint.ticker_2],
                coint.beta
            )
        elif coint.type == "Johansen":
            return ou_trading_utils.ou_spread_johansen(prices[coint.ticker_list], np.array(coint.chosen_weights))
        else:
            raise ValueError(f"Unknown basket type: {coint.type}")
    
    