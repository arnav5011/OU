import pandas as pd
import yfinance as yf
from data_process_helper import parse_date
import numpy as np
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
import constants
import API_setup


key = API_setup.API_key
secret = API_setup.API_secret
base_URL = API_setup.base_URL


class RawDataRead:
    @staticmethod
    def read_raw_data():
        path_to_raw = os.path.join(constants.DIRNAME, "data_management", "raw_data")
        path_to_file = os.path.join(path_to_raw, "S&P500.csv")
        raw_data = pd.read_csv(path_to_file, header = 0)
        return raw_data

class DataProcess:
    def __init__(self):
        self.data = RawDataRead.read_raw_data()
    
    def filter_sector(self, sector):
        valid_sector_list = set(list(self.data["Sector"]))
        if sector not in valid_sector_list:
            raise KeyError("Invalid sector key")
        filtered_data = self.data[self.data["Sector"] == sector]
        return filtered_data

class FinancialData(DataProcess):
    def __init__(self, sector, start, end):
        """sectors = ['Utilities', 'Financials', 'Consumer Discretionary', 
                    'Information Technology', 'Materials', 'Industrials', 'Energy',
                    'Consumer Staples', 'Real Estate', 'Telecommunication Services', 'Health Care']
        start and end type: int, format: YYYYMMDD"""
        self.start = parse_date(start)
        self.end = parse_date(end)
        super().__init__()
        self.sector = sector
        filtered_data = self.filter_sector(self.sector)
        self.ticker_list = list(filtered_data["Symbol"])
    
    
    def get_financial_feature_data(self, benchmark = "SPY"):
        """ Features for clustering:
        1. Momentum
        2. Log Market Capitalization
        3. Beta
        4. Volatility
        5. Liquidity
        """
        features = []
        
        # Getting returns for a benchmark
        benchmark_data_returns = yf.download(benchmark, start = self.start, end = self.end)["Adj Close"].pct_change().dropna()

        for symbol in self.ticker_list:
            try:
                ticker_history_data = yf.download(symbol, start = self.start, end = self.end)
                ticker_history_data["returns"] = ticker_history_data["Adj Close"].pct_change().dropna()

                beta = (ticker_history_data["returns"].cov(benchmark_data_returns))/ticker_history_data["returns"].var()
            except:
                print(f"Error for ticker {symbol}")    

        
        

        




a = FinancialData("Information Technology", 20100101, 20170101)
x = a.get_financial_feature_data()