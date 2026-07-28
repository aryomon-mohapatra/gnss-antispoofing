"""
ensemble.py
-----------
Weighted ensemble combining LSTM Autoencoder + XGBoost.
Final Score = 0.45 × Normalize(LSTM MSE) + 0.55 × P_xgb(spoofed)
"""

import numpy as np
import torch
import yaml


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


class GNSSEnsembleDetector:
    """
    Combines LSTM Autoencoder anomaly scores with XGBoost probabilities.
    """

    def __init__(self, lstm_model, xgb_detector, lstm_threshold, cfg):
        self.lstm = lstm_model
        self.xgb = xgb_detector
        self.lstm_threshold = lstm_threshold   # 95th pct of clean validation MSE
        self.lstm_weight = cfg["ensemble"]["lstm_weight"]
        self.xgb_weight  = cfg["ensemble"]["xgb_weight"] if "xgb_weight" in cfg["ensemble"] else cfg["ensemble"]["xgboost_weight"]
        self.decision_threshold = cfg["ensemble"]["decision_threshold"]

    def predict(self, X_flat, X_seq, device):
        """
        Args:
            X_flat: (N, 18) numpy array — for XGBoost
            X_seq:  (N, window, 18) tensor  — for LSTM
        Returns:
            predictions (0/1), confidence scores
        """
        # XGBoost probability
        xgb_proba = self.xgb.predict_proba(X_flat)   # (N,)

        # LSTM anomaly score (normalized)
        self.lstm.eval()
        X_seq_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
        lstm_mse = self.lstm.reconstruction_error(X_seq_tensor)
        lstm_score = np.clip(lstm_mse / (self.lstm_threshold * 2), 0, 1)

        # Weighted ensemble
        final_score = self.lstm_weight * lstm_score + self.xgb_weight * xgb_proba
        predictions = (final_score >= self.decision_threshold).astype(int)

        return predictions, final_score, xgb_proba, lstm_score

    def tune_threshold(self, X_flat_val, X_seq_val, y_val, device):
        """
        Find optimal decision threshold on validation set
        that maximizes F1 while keeping FAR <= 2%.
        """
        from sklearn.metrics import f1_score

        _, scores, _, _ = self.predict(X_flat_val, X_seq_val, device)
        best_thresh, best_f1 = 0.5, 0.0

        for t in np.arange(0.3, 0.75, 0.02):
            preds = (scores >= t).astype(int)
            f1 = f1_score(y_val, preds, zero_division=0)
            genuine_mask = y_val == 0
            far = preds[genuine_mask].mean() if genuine_mask.sum() > 0 else 1.0

            if f1 > best_f1 and far <= 0.02:
                best_f1 = f1
                best_thresh = t

        print(f"[Ensemble] Optimal threshold: {best_thresh:.2f}  (Val F1: {best_f1:.4f})")
        self.decision_threshold = best_thresh
        return best_thresh
