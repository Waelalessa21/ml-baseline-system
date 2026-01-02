from sklearn.model_selection import train_test_split
import pandas as pd


def split_random(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def split_time(X, y, test_size=0.2, time_col=None):
    if time_col is None or time_col not in X.columns:
        raise ValueError(f"time_col '{time_col}' must be provided and exist in X")
    
    split_idx = int(len(X) * (1 - test_size))
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    return X_train, X_test, y_train, y_test


def split_group(X, y, test_size=0.2, group_col=None, random_state=42):
    if group_col is None or group_col not in X.columns:
        raise ValueError(f"group_col '{group_col}' must be provided and exist in X")
    
    groups = X[group_col].unique()
    n_test_groups = max(1, int(len(groups) * test_size))
    
    import numpy as np
    np.random.seed(random_state)
    test_groups = np.random.choice(groups, size=n_test_groups, replace=False)
    
    test_mask = X[group_col].isin(test_groups)
    X_train = X[~test_mask]
    X_test = X[test_mask]
    y_train = y[~test_mask]
    y_test = y[test_mask]
    
    return X_train, X_test, y_train, y_test


def get_splitter(strategy):
    splitters = {
        "random": split_random,
        "time": split_time,
        "group": split_group,
    }
    
    if strategy not in splitters:
        raise ValueError(f"Unknown split strategy: {strategy}. Choose from {list(splitters.keys())}")
    
    return splitters[strategy]

