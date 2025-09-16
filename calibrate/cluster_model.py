import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
import constants
import pandas as pd
from utils.parsers import parse_date
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from abc import ABC, abstractmethod
from itertools import product
import matplotlib.pyplot as plt
import math
import numpy as np

class ImportProcessed:
    @staticmethod
    # Use this function to import the processed CSV data for a given sector and date range after running data_management/data_process.py
    def import_csv(sector, start_date, end_date):
        file_path = os.path.join(constants.DIRNAME, "data_management",
                                 "processed_data", f"{sector}", f"{start_date} - {end_date}", "features.csv")
        feature_data = pd.read_csv(file_path, header=[0], index_col=[0])
        return feature_data

class NormalizeData:
    # Normalize the features using StandardScaler for clustering
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

class ClusterModels(ABC):
    # Base class for clustering models
    def __init__(self, sector, start_date, end_date):
        self.normalized_data = NormalizeData(sector, start_date, end_date).normalize_features()
        self.model = None
        self.labels = None
        self.param_grid = None

    @abstractmethod
    def fit(self):
        pass

    def predict(self):
        if self.model is None or self.labels is None:
            raise ValueError("Model not fit yet.")
        return self.labels

    def get_clustered_data(self):
        df = self.normalized_data.copy()
        df["cluster"] = self.labels
        return df
    
    @abstractmethod
    def modify_parameters(self):
        pass

    def evaluate(self):
        X = self.normalized_data
        return {
            "silhouette": silhouette_score(X, self.labels),
            "calinski": calinski_harabasz_score(X, self.labels),
            "davies": davies_bouldin_score(X, self.labels)
        }
    
    def show_baskets(self):
        if self.labels is None:
            raise ValueError("Model must be fit before showing baskets.")

        df = self.get_clustered_data()
        baskets = {cluster: group.index.tolist() for cluster, group in df.groupby("cluster")}
        for cluster, items in baskets.items():
            print(f"\nCluster {cluster} ({len(items)} items):")
            print(items)
    
class ClusterFactory:
    @staticmethod
    # Make this scaleable for future clustering models by using cls
    def create(model_name, sector, start_date, end_date, **kwargs):
        if model_name == "kmeans":
            return KMeansClustering(sector, start_date, end_date, **kwargs)
        elif model_name == "gmm":
            return GaussianMixtureClustering(sector, start_date, end_date, **kwargs)
        elif model_name == "hierarchical":
            return HierarchicalClustering(sector, start_date, end_date, **kwargs)
        elif model_name == "spectral":
            return SpectralClusteringModel(sector, start_date, end_date, **kwargs)
        else:
            raise ValueError(f"Unknown clustering model: {model_name}")

# Implement different clustering models inheriting from ClusterModels
class KMeansClustering(ClusterModels):
    def __init__(self, sector, start_date, end_date, **kwargs):
        super().__init__(sector, start_date, end_date)
        self.n_clusters = kwargs.get("n_clusters", 5)
        self.param_grid = {"n_clusters": list(range(2, 11))}

    def fit(self):
        self.model = KMeans(n_clusters=self.n_clusters, random_state=0, n_init=10)
        self.labels = self.model.fit_predict(self.normalized_data)
    
    def modify_parameters(self, **kwargs):
        self.n_clusters = kwargs.get("n_clusters", self.n_clusters)

class GaussianMixtureClustering(ClusterModels):
    def __init__(self, sector, start_date, end_date, **kwargs):
        super().__init__(sector, start_date, end_date)
        self.n_components = kwargs.get("n_components", 5)
        self.cov = kwargs.get("covariance_type", "full")
        self.param_grid = {
            "n_components": list(range(2, 11)),
            "covariance_type": ["full", "tied", "diag", "spherical"]
        }

    def fit(self):
        self.model = GaussianMixture(n_components=self.n_components, random_state=0, covariance_type=self.cov)
        self.labels = self.model.fit_predict(self.normalized_data)
    
    def modify_parameters(self, **kwargs):
        self.n_components = kwargs.get("n_components", self.n_components)
        self.cov = kwargs.get("covariance_type", self.cov)

class HierarchicalClustering(ClusterModels):
    def __init__(self, sector, start_date, end_date, **kwargs):
        super().__init__(sector, start_date, end_date)
        self.n_clusters = kwargs.get("n_clusters", 5)
        self.linkage = kwargs.get("linkage", "ward")
        self.param_grid = {
            "n_clusters": list(range(2, 11)),
            "linkage": ["ward", "complete", "average", "single"]
        }

    def fit(self):
        self.model = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage)
        self.labels = self.model.fit_predict(self.normalized_data)
    
    def modify_parameters(self, **kwargs):
        self.n_clusters = kwargs.get("n_clusters", self.n_clusters)
        self.linkage = kwargs.get("linkage", self.linkage)

class SpectralClusteringModel(ClusterModels):
    def __init__(self, sector, start_date, end_date, **kwargs):
        super().__init__(sector, start_date, end_date)
        self.n_clusters = kwargs.get("n_clusters", 5)
        self.affinity = kwargs.get("affinity", "rbf")
        self.param_grid = {
            "n_clusters": list(range(2, 11)),
            "affinity": ["rbf", "nearest_neighbors"]
        }

    def fit(self):
        self.model = SpectralClustering(n_clusters=self.n_clusters, affinity=self.affinity, random_state=0)
        self.labels = self.model.fit_predict(self.normalized_data)
    
    def modify_parameters(self, **kwargs):
        self.n_clusters = kwargs.get("n_clusters", self.n_clusters)
        self.affinity = kwargs.get("affinity", self.affinity)
    
# Method to find the optimum clustering parameters
class ClusteringOptimizer:
    def __init__(self, model: ClusterModels):
        self.model = model

    def _cluster_balance_penalty(self, labels):
        from collections import Counter
        counts = list(Counter(labels).values())
        sizes = np.array(counts) / sum(counts)  # relative sizes
        # Penalize if cluster sizes vary too much, e.g. high std dev
        penalty = np.std(sizes)
        return penalty

    def _custom_score(self, metrics, kwargs, labels, cluster_penalty_weight=25):
        # Get a custom score from the evaluated metrics
        # The cluster penalty weight can be adjusted, and ideally can be done using ML techniques
        n_clusters = kwargs.get("n_clusters") or kwargs.get("n_components", 1)
        silhouette = metrics["silhouette"]
        calinski = metrics["calinski"]
        davies = metrics["davies"]
        if davies <= 0 or silhouette <= 0:
            return -float("inf")
        base_score = (silhouette * math.log(calinski + 1)) / (davies) * math.sqrt(n_clusters)
        penalty = self._cluster_balance_penalty(labels)
        adjusted_score = base_score - cluster_penalty_weight * penalty
        return adjusted_score

    def search(self):
        # Search through the parameter grid and find the best parameters based on custom score
        all_results = []
        best_score = float('-inf')
        best_params = None

        keys, values = zip(*self.model.param_grid.items())
        for combo in product(*values):
            kwargs = dict(zip(keys, combo))
            try:
                self.model.modify_parameters(**kwargs)
                self.model.fit()
                metrics = self.model.evaluate()

                score = self._custom_score(metrics, kwargs, self.model.labels)
                all_results.append((kwargs, metrics, score))

                if score > best_score:
                    best_score = score
                    best_params = kwargs
            except Exception as e:
                print(f"Failed with params {kwargs}: {e}")
                continue

        return best_params, best_score, all_results