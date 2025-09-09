import numpy as np
import pandas as pd

def calculate_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    mean = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    zscore = (series - mean) / std
    return zscore

def generate_ou_signals(zscore: pd.Series, entry_threshold: float = 1.0, exit_threshold: float = 0.0) -> pd.Series:
    signals = pd.Series(index=zscore.index, dtype=int)
    signals[zscore > entry_threshold] = -1  # Short spread
    signals[zscore < -entry_threshold] = 1  # Long spread
    signals[(zscore <= exit_threshold) & (zscore >= -exit_threshold)] = 0  # Exit
    signals = signals.fillna(0)
    return signals

def ou_spread_engle(series1: pd.Series, series2: pd.Series, hedge_ratio: float) -> pd.Series:
    return series1 - hedge_ratio * series2

def ou_spread_johansen(series: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    return series @ weights