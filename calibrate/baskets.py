from .cointegration import CointegratedResults
from .OU_calibration import OUResults
class Basket:
    def __init__(self, cluster_id: int, cointegration_result: CointegratedResults, params: OUResults):
        self.cluster_id = cluster_id
        self.params = params
        self.cointegration_result = cointegration_result
    
    def __repr__(self):
        return f"Basket(cluster_id={self.cluster_id}, cointegration_type={self.cointegration_result.type})\ntickers: {self.cointegration_result.ticker_list if self.cointegration_result.type == 'Johansen' else [self.cointegration_result.ticker_1, self.cointegration_result.ticker_2]}\nOU Params: mu={self.params.mu:.4f}, sigma={self.params.sigma:.4f}, half_life={self.params.half_life:.2f}"