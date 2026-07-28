"""
train.py
--------
Trains both models:
  1. LSTM Autoencoder (on genuine signals only)
  2. XGBoost classifier (on all labeled samples)

Usage:
    python src/train.py
    python src/train.py --config config.yaml --synthetic
"""

import argparse
import os
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from feature_extractor import generate_synthetic_dataset
from dataset import load_features, get_splits, fit_scaler, get_loaders, FEATURE_COLS
from lstm_autoencoder import build_lstm_model
from xgboost_classifier import XGBoostDetector


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────────────────────
# LSTM Autoencoder Training
# ──────────────────────────────────────────────────────────────────────────────

def train_lstm(model, train_loader, val_loader, cfg, device):
    lstm_cfg = cfg["lstm_autoencoder"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lstm_cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()
    scaler_amp = GradScaler()

    best_val_loss = float("inf")
    patience = lstm_cfg["early_stopping_patience"]
    no_improve = 0
    best_state = None

    print(f"\n[LSTM] Training for up to {lstm_cfg['epochs']} epochs...")

    for epoch in range(1, lstm_cfg["epochs"] + 1):
        # Train
        model.train()
        train_losses = []
        for x, _ in tqdm(train_loader, desc=f"  Epoch {epoch:03d} Train", leave=False):
            x = x.to(device)
            optimizer.zero_grad()
            with autocast():
                recon = model(x)
                loss = criterion(recon, x)
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                with autocast():
                    recon = model(x)
                    loss = criterion(recon, x)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        print(f"  Epoch {epoch:03d} | Train MSE: {train_loss:.5f} | Val MSE: {val_loss:.5f} | LR: {lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[LSTM] Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model, best_val_loss


def compute_anomaly_threshold(model, X_genuine, cfg, device):
    """
    Compute the 95th percentile MSE on clean validation data.
    Signals above this threshold are flagged as spoofed.
    """
    from dataset import GNSSSequenceDataset
    from torch.utils.data import DataLoader

    window = cfg["data"]["window_size"]
    y_zero = np.zeros(len(X_genuine), dtype=np.int64)
    ds = GNSSSequenceDataset(X_genuine, y_zero, window, genuine_only=False)
    loader = DataLoader(ds, batch_size=256, shuffle=False)

    model.eval()
    all_mse = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            mse = model.reconstruction_error(x)
            all_mse.extend(mse.tolist())

    pct = cfg["lstm_autoencoder"]["anomaly_threshold_percentile"]
    threshold = float(np.percentile(all_mse, pct))
    print(f"[LSTM] Anomaly threshold ({pct}th pct on clean val): {threshold:.6f}")
    return threshold


# ──────────────────────────────────────────────────────────────────────────────
# Main Training Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    os.makedirs(cfg["output"]["models_dir"], exist_ok=True)
    os.makedirs(cfg["output"]["results_dir"], exist_ok=True)

    # ── Load / generate data ──────────────────────────────────────────────────
    if args.synthetic:
        print("[Train] Using synthetic dataset.")
        df = generate_synthetic_dataset(n_genuine=3000, n_spoofed=600)
        import pandas as pd
        os.makedirs(cfg["data"]["features_dir"], exist_ok=True)
        df.to_csv(f"{cfg['data']['features_dir']}/features.csv", index=False)

    X, y, df = load_features(cfg)
    X_train, X_val, X_test, y_train, y_val, y_test = get_splits(X, y, cfg)

    # Normalize
    scaler = fit_scaler(X_train, f"{cfg['output']['models_dir']}/scaler.pkl")
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_val_s   = scaler.transform(X_val).astype(np.float32)
    X_test_s  = scaler.transform(X_test).astype(np.float32)

    # ── Train LSTM Autoencoder ─────────────────────────────────────────────────
    train_loader, val_loader, _ = get_loaders(X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, cfg)
    lstm_model = build_lstm_model(cfg).to(device)
    lstm_model, _ = train_lstm(lstm_model, train_loader, val_loader, cfg, device)

    # Save checkpoint
    ckpt_path = cfg["output"]["lstm_checkpoint"]
    threshold = compute_anomaly_threshold(lstm_model, X_val_s[y_val == 0], cfg, device)
    torch.save({
        "model_state_dict": lstm_model.state_dict(),
        "anomaly_threshold": threshold,
        "config": cfg,
    }, ckpt_path)
    print(f"[LSTM] Saved: {ckpt_path}")

    # ── Train XGBoost ──────────────────────────────────────────────────────────
    print("\n[XGBoost] Training...")
    xgb_detector = XGBoostDetector(cfg)
    xgb_detector.train(X_train_s, y_train, X_val_s, y_val)
    xgb_detector.save(cfg["output"]["xgboost_checkpoint"])
    xgb_detector.plot_feature_importance(f"{cfg['output']['results_dir']}/feature_importance.png")

    # SHAP explanation on one spoofed sample (if any)
    spoofed_idx = np.where(y_val == 1)[0]
    if len(spoofed_idx) > 0:
        import pandas as pd
        X_sample = pd.DataFrame(X_val_s[spoofed_idx[:1]], columns=FEATURE_COLS)
        xgb_detector.explain_prediction(X_sample, f"{cfg['output']['results_dir']}/shap_spoofed.png")

    print("\n[Train] All models trained and saved. Run evaluate.py for full metrics.")


if __name__ == "__main__":
    main()
