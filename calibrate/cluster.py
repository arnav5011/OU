import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
import constants
import pandas as pd
from utils.parsers import parse_date
from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod

class ImportProcessed:
    @staticmethod
    def import_csv(sector, start_date, end_date):
        file_path = os.path.join(constants.DIRNAME, "data_management",
                                 "processed_data", f"{sector}", f"{start_date} - {end_date}", "features.csv")
        feature_data = pd.read_csv(file_path)
        feature_data.columns.values[0] = "symbol"
        return feature_data

class NormalizeData:
    def __init__(self, sector, start_date, end_date):
        parsed_start_date = parse_date(start_date)
        parsed_end_date = parse_date(end_date)
        self.data = ImportProcessed.import_csv(sector, parsed_start_date, parsed_end_date)
        self.data.dropna(inplace= True)
    
    def normalize_features(self):
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(self.data)
        normalized_df = pd.DataFrame(normalized_data, columns=self.data.columns, index=self.data.index)
        return normalized_df

class ClusterModels(NormalizeData, ABC):
    def __init__(self, sector, start_date, end_date):
        super.__init__(sector, start_date, end_date)
    
    @abstractmethod
    def cluster_data(self):
        pass

class kMeanClustering(ClusterModels):
    pass