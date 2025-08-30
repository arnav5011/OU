import numpy as np
import pandas as pd
from statsmodels.api import OLS, add_constant
from calibrate.cointegration import CointegratedResults

class OUResults:
    def __init__(self, ou_params: dict, ols_params: dict):
        self.theta = ou_params["theta"]
        self.mu =  ou_params["mu"]
        self.sigma = ou_params["sigma"]
        self.half_life = ou_params["half_life"]
        self.phi = ols_params["phi"]
        self.c = ols_params["c"]
        self.sigma_eps2 = ols_params["sigma_eps2"]
        self.ols_summary = ols_params["ols_summary"]


class OUCalibration:
    def __init__(self, price_data, cointegrated_cluster_data: dict[int, CointegratedResults]):
        self._log_price_data = np.log(price_data)
        self.cointegrated_cluster_data = cointegrated_cluster_data
    
    def form_spread(self):
        spreads = {}
        for cluster, results in self.cointegrated_cluster_data.items():
            if results.type == "Engle-Granger":
                t1, t2 = results.ticker_1, results.ticker_2
                beta, alpha = results.beta, results.alpha
                spread = self._log_price_data[t1] - beta * self._log_price_data[t2] + alpha

            elif results.type == "Johansen":
                tickers = results.ticker_list
                price_data_tickers = self._log_price_data[tickers]
                spread = price_data_tickers.values @ results.chosen_weights
                spread = pd.Series(spread, index=price_data_tickers.index, name=f"spread_cluster_{cluster}")

            else:
                raise ValueError("Invalid Parameters")

            spreads[cluster] = spread

        return spreads

    def fit_ou_parameters(self, spreads: dict[int, pd.Series], delta: float = 1.0):
        ou_results = {}

        for cluster, spread in spreads.items():
            X = spread.dropna()
            Xt = X.iloc[:-1].values
            Xnext = X.iloc[1:].values

            # AR(1) with intercept: X_{t+1} = c + phi*X_t + eps
            Xreg = add_constant(Xt)  # intercept + lag
            model = OLS(Xnext, Xreg).fit()
            c, phi = model.params

            # OU parameters
            theta = -np.log(phi) / delta
            mu = c / (1 - phi)
            sigma_eps2 = model.mse_resid
            sigma = np.sqrt((2 * theta * sigma_eps2) / (1 - phi**2))
            half_life = np.log(2) / theta

            ou_params = {
                "theta": theta,
                "mu": mu,
                "sigma": sigma,
                "half_life": half_life
            }

            ols_params = {
                "phi": phi,
                "c": c,
                "sigma_eps2": sigma_eps2,
                "ols_summary": model.summary().as_text()
            }

            ou_results[cluster] = OUResults(ou_params, ols_params)

        return ou_results