import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import yfinance as yf
import pandas as pd
from utils.parsers import parse_date
from itertools import combinations
from time import sleep

class PriceData:
    @staticmethod
    def get_time_series(data: pd.DataFrame, start_date, end_date):
        start_date = parse_date(start_date)
        end_date = parse_date(end_date)
        ticker_list = data.index.tolist()
        batch_size = 20
        all_prices = []
        failed_tickers = []

        for i in range(0, len(ticker_list), batch_size):
            batch = ticker_list[i:i + batch_size]
            retries = 0
            while retries < 5:
                try:
                    prices = yf.download(batch, start=start_date, end=end_date)['Close']
                    
                    # Detect missing tickers
                    missing = [t for t in batch if t not in prices.columns]
                    if not prices.empty and len(missing) == 0:
                        all_prices.append(prices)
                        break  # batch success
                    else:
                        print(f"Missing tickers {missing}, retrying batch...")
                        batch = missing  # retry only missing tickers
                        retries += 1
                        sleep(10)
                except Exception as e:
                    print(f"Error downloading batch {batch}: {e}")
                    retries += 1
                    sleep(10)

            # Any tickers still missing after retries
            if retries == 5 and len(batch) > 0:
                failed_tickers.extend(batch)

        price_data = pd.concat(all_prices, axis=1)

        if failed_tickers:
            print(f"⚠ Warning: Failed to download data for {failed_tickers}")

        return price_data

class CointegrrationTest:
    def __init__(self, price_data, clustered_data):
        self.price_data = price_data
        self.clustered_data = clustered_data
    

    def _engle_granger_test(self, ticker1, ticker2, confidence_level=0.95):
        series1 = self.price_data[ticker1].dropna()
        series2 = self.price_data[ticker2].dropna()
        if len(series1) < 30 or len(series2) < 30:
            return False, None
        
        score, p_value, _ = coint(series1, series2)
        return p_value < (1-confidence_level), p_value

    def _johansen_test(self, tickers, confidence_level=0.95):
        series = self.price_data[tickers].dropna()
        if len(series) < 30:
            return False, None

        cointegrated_sets = []



        
        result = coint_johansen(series, det_order=0, k_ar_diff=1)
        return result.lr1[0] > result.cvt[0][1], result.lr1[0]

    def run_tests(self):
        results = {}
        for cluster in self.clustered_data['cluster'].unique():
            cluster_tickers = self.clustered_data[self.clustered_data['cluster'] == cluster].index.tolist()
            if len(cluster_tickers) < 2:
                continue
                
            if len(cluster_tickers) == 2:
                ticker1, ticker2 = cluster_tickers
                is_cointegrated, p_value = self._engle_granger_test(ticker1, ticker2)
                results[cluster] = {
                    "test": "Engle-Granger",
                    "tickers": [ticker1, ticker2],
                    "is_cointegrated": is_cointegrated,
                    "p_value": p_value
                }
            else:
                pass


