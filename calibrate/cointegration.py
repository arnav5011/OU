import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import yfinance as yf
import pandas as pd
from utils.parsers import parse_date
import statsmodels.api as sm
from time import sleep
from statsmodels.tsa.stattools import adfuller
import numpy as np

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

        _log_price_data = pd.concat(all_prices, axis=1)

        if failed_tickers:
            print(f"⚠ Warning: Failed to download data for {failed_tickers}")

        return _log_price_data

class CointegratedResults:
    def __init__(self, type, cointegrated, **kwargs):
        self.type = type
        self.cointegrated = cointegrated
        self.p_value = kwargs.get("p_value", None)
        self.rank = kwargs.get("rank", None)
        self.alpha = kwargs.get("alpha", None)
        self.beta = kwargs.get("beta", None)
        self.ticker_1 = kwargs.get("ticker_1", None)
        self.ticker_2 = kwargs.get("ticker_2", None)
        self.ticker_list = kwargs.get("ticker_list", None)
        self.chosen_weights = kwargs.get("chosen_weights", None)
    
    def modify_chosen_weights(self, weights):
        if self.weight_list is not None:
            self.chosen_weights = weights
        else:
            raise ValueError("No weight list available to modify chosen weights.")

class CointegrrationTest:
    def __init__(self, _log_price_data, clustered_data):
        self._log_price_data = np.log10(_log_price_data)
        self.clustered_data = clustered_data
    
    def _engle_granger_test(self, ticker1, ticker2, confidence_level=0.8): # For sake of testing 80% confidence
        series1 = self._log_price_data[ticker1].dropna()
        series2 = self._log_price_data[ticker2].dropna()
            
        print(f"series1: {len(series1)}, series2: {len(series2)}")
        if len(series1) != len(series2):
            min_length = min(len(series1), len(series2))
            series1 = series1[-min_length:]
            series2 = series2[-min_length:]
        
        if len(series1) < 30 or len(series2) < 30:
            return False, None, None, None
        score1, p1, _ = coint(series1, series2)
        score2, p2, _ = coint(series2, series1)
        p_value = min(p1, p2)
        cointegrated = p_value < (1 - confidence_level)

        X = sm.add_constant(series2)
        model = sm.OLS(series1, X).fit()
        alpha, beta = model.params

        return cointegrated, p_value, alpha, beta

    def _johansen_test(self, tickers):
        series = self._log_price_data[tickers].dropna()
        if len(series) < 30:
            return False, None, None
        
        rank = 0
        result = coint_johansen(series, det_order=0, k_ar_diff=1)
        trace_stat = result.lr1
        critical_values = result.cvt[:, 0]  # 90% critical values
        
        for i,res in enumerate(trace_stat):
            if pd.isna(critical_values[i]):
                rank = i + 1
            elif res > critical_values[i]:
                rank = i + 1
            else:
                break
        cointegrated = rank > 0
        weight_vectors = result.evec
        
        if not cointegrated:
            return False, None, None
        
        weight_list = []
        for i in range(rank):
            vec = weight_vectors[:, i]
            vec = vec / vec[0]
            weight_list.append(vec)
        
        adf_results = []
        for vec in weight_list:
            spread = series.dot(vec)
            adf_stat, p_value, _, _, critical_values, _ = adfuller(spread)
        
            adf_results.append({
                'vector': vec,
                'adf_stat': adf_stat,
                'p_value': p_value,
                'critical_values': critical_values
            })

        best_spread = min(adf_results, key=lambda x: x['p_value'])
        best_vector = best_spread['vector']
        return cointegrated, rank, best_vector

    def run_cointegration_tests(self):
        print("Running cointegration tests...")
        results = {}
        for cluster in self.clustered_data['cluster'].unique():
            cluster_tickers = self.clustered_data[self.clustered_data['cluster'] == cluster].index.tolist()
            if len(cluster_tickers) < 2:
                continue
                
            if len(cluster_tickers) == 2:
                ticker1, ticker2 = cluster_tickers
                is_cointegrated, p_value, alpha, beta = self._engle_granger_test(ticker1, ticker2)
                results[cluster] = CointegratedResults(type='Engle-Granger',
                                        cointegrated= is_cointegrated,
                                        p_value= p_value,
                                        alpha= alpha,
                                        beta= beta,
                                        ticker_1= ticker1,
                                        ticker_2= ticker2)
                
            else:
                tickers = cluster_tickers
                is_cointegrated, rank, best_spread = self._johansen_test(tickers)
                results[cluster] = CointegratedResults(type='Johansen',
                                        cointegrated= is_cointegrated,
                                        rank= rank,
                                        chosen_weights= best_spread,
                                        ticker_list= tickers)
                
        
        return results
    
    def get_cointegrated_clusters(self):
        cointegrated_clusters = {}
        results = self.run_cointegration_tests()
        for cluster, result in results.items():
            if result.cointegrated:
                cointegrated_clusters[cluster] = result
        return cointegrated_clusters


