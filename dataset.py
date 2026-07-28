"""
dataset.py
----------
Dataset loader for GNSS spoofing detection.
Handles both tabular features (for XGBoost) and sequences (for LSTM).
"""

import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


FEATURE_COLS = [
    "cn0_mean", "cn0_std", "agc_level", "carrier_phase_noise",
    "signal_lock_time", "cycle_slip_count",
    "pdop", "hdop", "vdop", "doppler_residual",
    "velocity_consistency", "position_jump",
    "pseudorange_residual_mean", "pseudorange_residual_std",
    "toa_spread", "inter_sat_correlation",
    "nav_message_anomaly", "iono_residual"
]


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_features(cfg):
    """Load feature CSV and return X, y arrays."""
    path = os.path.join(cfg["data"]["features_dir"], "features.csv")
    if not os.path.exists(path):
        # Fall back to synthetic data for demo
        print("[Dataset] Feature CSV not found — generating synthetic data.")
        from feature_extractor import generate_synthetic_dataset
        df = generate_synthetic_dataset()
        os.makedirs(cfg["data"]["features_dir"], exist_ok=True)
        df.to_csv(path, index=False)
    else:
        df = pd.read_csv(path)

    # Keep only labeled samples
    df = df[df["label"].isin([0, 1])].copy()
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["label"].values.astype(np.int64)
    print(f"[Dataset] Loaded {len(df)} samples: genuine={sum(y==0)}, spoofed={sum(y==1)}")
    return X, y, df


def get_splits(X, y, cfg):
    """Stratified train/val/test split."""
    seed = cfg["data"]["random_seed"]
    test_size = cfg["data"]["test_split"]
    val_size = cfg["data"]["val_split"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)

    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_ratio, stratify=y_train, random_state=seed)

    print(f"[Dataset] Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_scaler(X_train, save_path="models/scaler.pkl"):
    """Fit StandardScaler on training data only."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(scaler, save_path)
    return scaler


def load_scaler(path="models/scaler.pkl"):
    return joblib.load(path)


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset for LSTM (sequences)
# ──────────────────────────────────────────────────────────────────────────────

class GNSSSequenceDataset(Dataset):
    """
    Sliding window dataset for LSTM Autoencoder.
    Only uses GENUINE samples for autoencoder training.
    """
    def __init__(self, X, y, window_size=30, genuine_only=False):
        if genuine_only:
            X = X[y == 0]
        self.sequences = []
        for i in range(len(X) - window_size + 1):
            self.sequences.append(X[i:i + window_size])
        self.sequences = np.array(self.sequences, dtype=np.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx])
        return seq, seq  # Autoencoder: input == target


class GNSSLabeledDataset(Dataset):
    """Per-epoch dataset with labels (for evaluation only)."""
    def __init__(self, X, y, window_size=30):
        self.window_size = window_size
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        # Build valid indices
        self.indices = list(range(window_size - 1, len(X)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        end = self.indices[idx] + 1
        start = end - self.window_size
        return self.X[start:end], self.y[self.indices[idx]]


def get_loaders(X_train, X_val, X_test, y_train, y_val, y_test, cfg):
    """Build all DataLoaders."""
    window = cfg["data"]["window_size"]
    bs = cfg["lstm_autoencoder"]["batch_size"]

    train_ds = GNSSSequenceDataset(X_train, y_train, window, genuine_only=True)
    val_ds   = GNSSSequenceDataset(X_val,   y_val,   window, genuine_only=False)
    test_ds  = GNSSLabeledDataset(X_test, y_test, window)

    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=0),
        DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=0),
        DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=0),
    )
