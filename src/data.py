"""
data.py — loading and preprocessing for the snowmelt PINN project.
"""

import pandas as pd
import numpy as np
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data"

# Feature columns fed into the neural network
FEATURE_COLS = ["t2m_c", "rad_mj", "swe_mm", "elev_m"]

# Target column
TARGET_COL = "melt_obs_mm"


def load_train_test():
    """
    Load pre-split train and test DataFrames.
    
    Returns
    -------
    train : pd.DataFrame  (2000–2015)
    test  : pd.DataFrame  (2016–2023)
    """
    train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["date"])
    test  = pd.read_csv(DATA_DIR / "test.csv",  parse_dates=["date"])
    return train, test


def get_feature_stats(train_df):
    """
    Compute mean and std of features on the training set only.
    Used for normalisation — NEVER compute on test set.
    
    Returns
    -------
    means : np.ndarray  shape (n_features,)
    stds  : np.ndarray  shape (n_features,)
    """
    X = train_df[FEATURE_COLS].values
    return X.mean(axis=0), X.std(axis=0)


def prepare_arrays(df, means, stds):
    """
    Extract and normalise features, extract target.
    
    Parameters
    ----------
    df    : pd.DataFrame with FEATURE_COLS and TARGET_COL
    means : np.ndarray from get_feature_stats()
    stds  : np.ndarray from get_feature_stats()
    
    Returns
    -------
    X_norm : np.ndarray  shape (N, n_features), float32, normalised
    y      : np.ndarray  shape (N,),            float32
    """
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)
    X_norm = (X - means) / (stds + 1e-8)   # +eps avoids division by zero
    return X_norm, y