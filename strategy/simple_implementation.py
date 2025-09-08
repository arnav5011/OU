import numpy as np
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from calibrate.baskets import Basket

class OUTrader:
    def __init__(self, price_data, baskets: list[Basket]):
        self.price_data = price_data
        self.log_price_data = np.log(price_data)
        self.baskets = Basket
    
    def run_trader(self):
        pass
    
        
