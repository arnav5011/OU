import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import os
os.environ["OMP_NUM_THREADS"] = "1"
import yfinance as yf
import data_extract
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from pprint import pprint

def create_basket(df):
    feature_cols = [c for c in df.columns if c != 'Ticker']
    X_raw = df[feature_cols]
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    
    k_ideal = 8
    print(k_ideal)
    kmeanModel = KMeans(n_clusters=k_ideal, random_state=42).fit(X)
    df['Cluster'] = kmeanModel.labels_
    cluster_summary = df.groupby('Cluster')[['Log Market Cap','Momentum','Volatility']].agg(['mean','std'])
    print(cluster_summary)
    return df, k_ideal

def johansen_test(df, n):
    result_summary = {}
    for i in range(n):
        current_tickers = df.loc[df['Cluster'] == i, 'Ticker'].tolist()
        if len(current_tickers)>1:
            data = data_extract.download_log_close(current_tickers)
            result = coint_johansen(data.values, det_order=0,k_ar_diff=1)
            out = {
            'r': np.arange(len(result.lr1)),               
            'trace_stat': result.lr1,                       
            'crit_90': result.cvt[:,0],                     
            'crit_95': result.cvt[:,1],                     
            'crit_99': result.cvt[:,2],                     
            'max_eig_stat': result.lr2,                     
            'max_90': result.cvm[:,0],
            'max_95': result.cvm[:,1],
            'max_99': result.cvm[:,2],
            }
        else:
            out = {}
        result_summary[i] = out
    return result_summary


df = data_extract.load_excel_data("s&p500.xlsx")
info_sect = data_extract.filter_by_sectors(df, "Information Technology")
info_sect_fin_data = data_extract.get_financial_data(info_sect)
basket, clusters = create_basket(info_sect_fin_data)
pprint(johansen_test(basket, clusters))
