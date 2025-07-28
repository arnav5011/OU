import pandas as pd
import yfinance as yf
from utils.parsers import parse_date
from collections import defaultdict
import argparse
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
import constants

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
        1. Beta
        3. Momentum
        4. Volatility
        5. Liquidity
        """
        features = {}
        
        # Getting returns for a benchmark
        benchmark_data_returns = yf.download(benchmark, start = self.start, end = self.end)["Close"].pct_change().dropna()
        
        for symbol in self.ticker_list:
            try:
                ticker_history_data = yf.download(symbol, start = self.start, end = self.end, group_by='column')
                
                if isinstance(ticker_history_data.columns, pd.MultiIndex):
                    ticker_history_data.columns = ticker_history_data.columns.get_level_values(0)
                
                
                ticker_history_data["returns"] = ticker_history_data["Close"].pct_change().dropna()
                
                returns = pd.concat([ticker_history_data["returns"], benchmark_data_returns], axis=1).dropna()
                returns.columns = ['asset', 'benchmark']
                beta = returns['asset'].cov(returns['benchmark']) / returns['benchmark'].var()
                
                momentum = ticker_history_data["Close"].iloc[-1] / ticker_history_data["Close"].iloc[-21] - 1
                
                volatility = ticker_history_data["returns"].rolling(window = 20).std().median()
                
                avg_volume = ticker_history_data['Volume'].rolling(window = 20).mean().median()
                
                autocorrelation = ticker_history_data["returns"].autocorr(lag = 1)
                
                spread = ticker_history_data["returns"].quantile(0.75) - ticker_history_data["returns"].quantile(0.25)

                features[symbol] = {"beta": beta, 
                                    "momentum": momentum,
                                    "volatility": volatility,
                                    "liquidity": avg_volume,
                                    "autocorrelation": autocorrelation,
                                    "spread": spread}
                                    
            except Exception as e:
                print(f"Error for {symbol}: {type(e).__name__} - {e}")    
        df = pd.DataFrame.from_dict(features, orient = "index")
        return df

class FeatureExport(FinancialData):
    def __init__(self, sector, start, end):
        super().__init__(sector, start, end)
    
    def export_CSV(self):
        features_data = self.get_financial_feature_data()
        dir_path = os.path.join(constants.DIRNAME, "data_management", "processed_data", f"{self.sector}", f"{self.start} - {self.end}")
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)  # Creates all intermediate directories if needed

        features_data.to_csv(os.path.join(dir_path, "features.csv"))

def main():
    parser = argparse.ArgumentParser()
    allowed_sectors = constants.SECTORS
    parser.add_argument("--sector", choices=allowed_sectors, required=True)
    parser.add_argument("--start_date", type=int, required=True, help="Format: YYYYMMDD")
    parser.add_argument("--end_date", type=int, required=True, help="Format: YYYYMMDD")
    args=parser.parse_args()
    sector = args.sector
    start_date = args.start_date
    end_date = args.end_date
    exporter = FeatureExport(sector, start_date, end_date)
    exporter.export_CSV()


if __name__ == "__main__":
    main()