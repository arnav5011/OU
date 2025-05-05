import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import os
os.environ["OMP_NUM_THREADS"] = "1"
import yfinance as yf
import data_extract
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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
    for i in range(n):
        current_tickers = df.loc[df['Cluster'] == i, 'Ticker'].tolist()
        print(current_tickers)

df = data_extract.load_excel_data("s&p500.xlsx")
info_sect = data_extract.filter_by_sectors(df, "Information Technology")
info_sect_fin_data = data_extract.get_financial_data(info_sect)
basket, clusters = create_basket(info_sect_fin_data)
johansen_test(basket, clusters)
