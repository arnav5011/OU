import yfinance as yf
import pandas as pd
from utils.parsers import parse_date
from time import sleep

def get_time_series(data, start_date, end_date):
        if isinstance(data, pd.DataFrame): 
            ticker_list = data.index.tolist()
        elif isinstance(data, list):
            ticker_list = data
        else:
            raise ValueError("Kill yourself retard")
        start_date = parse_date(start_date)
        end_date = parse_date(end_date)
        
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