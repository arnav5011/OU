import numpy as np
import statsmodels
import yfinance as yf
import data_extract
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def create_basket(df):
    
    feature_cols = [c for c in df.columns if c != 'Ticker']
    X_raw = df[feature_cols]
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
        
    k_ideal = int(len(df)/3)
    kmeanModel = KMeans(n_clusters=k_ideal, random_state=42).fit(X)
    df['Cluster'] = kmeanModel.labels_
    df = df.sort_values(by='Cluster').reset_index(drop=True)
    return df

def johansen_test():
    pass

df = data_extract.load_excel_data("s&p500.xlsx")
info_sect = data_extract.filter_by_sectors(df, "Information Technology")
info_sect_fin_data = data_extract.get_financial_data(info_sect)
print(create_basket(info_sect_fin_data))