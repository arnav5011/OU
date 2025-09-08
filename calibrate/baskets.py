from .cointegration import CointegratedResults
from .OU_calibration import OUResults
class Basket:
    def __init__(self, cluster_id: int, cointegration_result: CointegratedResults, params: OUResults):
        self.cluster_id = cluster_id
        self.params = params
        self.cointegration_resul = cointegration_result
