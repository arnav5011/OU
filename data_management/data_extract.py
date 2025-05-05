import pandas as pd
import yfinance as yf
import numpy as np


def load_excel_data(path):
    # Read data file for market   
    df = pd.read_excel(path)
    # Ensure data file contains the ticker, and the sector
    required_cols = ['Symbol', 'Sector', 'Company']
    if not all(col in df.columns for col in required_cols):
        raise ValueError("Excel missing required columns (Symbol, Sector, Company)")
    return df.dropna(subset=required_cols)

def filter_by_sectors(df, sector):
    # Filter the tickers into the desired sector
    sector = sector.title() 
    valid_sectors = df['Sector'].unique()
    if sector not in valid_sectors:
        available = ", ".join(valid_sectors)
        raise ValueError(f"WTF BRO KYS FUCKING BITCH\nValid Sectors: {available}")
    tickers = df.loc[df['Sector'] == sector, 'Symbol'].tolist()
    return tickers

def get_financial_data(sector):
    ticker_data = {}
    for ticker in sector:
        ticker = yf.Ticker(ticker)
        info = ticker.info
        # print(info.keys())
        market_cap = info.get("marketCap", np.nan)
        history = ticker.history(period='1y', interval='1d')['Close'].dropna()
        if len(history) < 200:
            # not enough data—skip
            continue
        ma50  = history.rolling(50).mean().iloc[-1]
        ma200 = history.rolling(200).mean().iloc[-1]
        mom   = ma50 / ma200 if ma200 else np.nan
        daily_ret = history.pct_change().dropna()
        vol_a = daily_ret.std() * np.sqrt(252)
        ticker_data[str(ticker)] = (info.get("symbol",np.nan), market_cap, mom, vol_a)
    ticker_data = pd.DataFrame.from_dict(ticker_data, orient = "index", columns=["Ticker", "Log Market Cap", "Momentum", "Volatility"]).dropna()
    ticker_data["Log Market Cap"] = np.log(ticker_data["Log Market Cap"])
    return ticker_data

# Debugging
# df = load_excel_data("s&p500.xlsx")
# info_sect = filter_by_sectors(df, "Information Technology")
# print(get_financial_data(info_sect))