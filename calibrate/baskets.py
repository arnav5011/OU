from .cointegration import CointegratedResults
from .OU_calibration import OUResults
class Basket:
    """Class to store relevant information about a basket of assets for running a mean reversion strategy."""
    def __init__(self, cluster_id: int, cointegration_result: CointegratedResults, params: OUResults):
        self.cluster_id = cluster_id
        self.params = params
        self.cointegration_result = cointegration_result
    
    def summary(self):
        return {
            "cluster_id": self.cluster_id,
            "cointegration_type": self.cointegration_result.type,
            "cointegrated": self.cointegration_result.cointegrated,
            "p_value": self.cointegration_result.p_value,
            "alpha": self.cointegration_result.alpha,
            "beta": self.cointegration_result.beta,
            "tickers": (self.cointegration_result.ticker_1, self.cointegration_result.ticker_2),
            "theta": self.params.theta,
            "mu": self.params.mu,
            "sigma": self.params.sigma,
            "half_life": self.params.half_life
        }
    
    def __repr__(self):
        return f"Basket(cluster_id={self.cluster_id}, cointegration_type={self.cointegration_result.type})\ntickers: {self.cointegration_result.ticker_list if self.cointegration_result.type == 'Johansen' else [self.cointegration_result.ticker_1, self.cointegration_result.ticker_2]}\nOU Params: mu={self.params.mu:.4f}, sigma={self.params.sigma:.4f}, half_life={self.params.half_life:.2f}"