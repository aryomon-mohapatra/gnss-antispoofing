"""
xgboost_classifier.py
---------------------
XGBoost classifier for per-epoch spoofing detection.
Includes SHAP-based explainability.
"""

import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import os
import yaml


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


FEATURE_NAMES = [
    "cn0_mean", "cn0_std", "agc_level", "carrier_phase_noise",
    "signal_lock_time", "cycle_slip_count",
    "pdop", "hdop", "vdop", "doppler_residual",
    "velocity_consistency", "position_jump",
    "pseudorange_residual_mean", "pseudorange_residual_std",
    "toa_spread", "inter_sat_correlation",
    "nav_message_anomaly", "iono_residual"
]


class XGBoostDetector:
    def __init__(self, cfg):
        xgb_cfg = cfg["xgboost"]
        self.model = xgb.XGBClassifier(
            n_estimators=xgb_cfg["n_estimators"],
            max_depth=xgb_cfg["max_depth"],
            learning_rate=xgb_cfg["learning_rate"],
            subsample=xgb_cfg["subsample"],
            colsample_bytree=xgb_cfg["colsample_bytree"],
            scale_pos_weight=xgb_cfg["scale_pos_weight"],
            eval_metric=xgb_cfg["eval_metric"],
            early_stopping_rounds=xgb_cfg["early_stopping_rounds"],
            use_label_encoder=False,
            random_state=cfg["data"]["random_seed"],
            tree_method="hist",
        )
        self.explainer = None
        self.feature_names = FEATURE_NAMES

    def train(self, X_train, y_train, X_val, y_val):
        print("[XGBoost] Training...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )
        best_iter = self.model.best_iteration
        print(f"[XGBoost] Best iteration: {best_iter}")

        # Fit SHAP explainer on training data
        self.explainer = shap.TreeExplainer(self.model)
        return self

    def predict_proba(self, X):
        """Returns P(spoofed) for each sample."""
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path="models/xgboost_model.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)
        print(f"[XGBoost] Saved to {path}")

    def load(self, path="models/xgboost_model.json"):
        self.model.load_model(path)
        self.explainer = shap.TreeExplainer(self.model)
        print(f"[XGBoost] Loaded from {path}")
        return self

    def plot_feature_importance(self, save_path="results/feature_importance.png"):
        """Plot SHAP feature importance."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        importance = self.model.feature_importances_
        sorted_idx = np.argsort(importance)[::-1]

        fig, ax = plt.subplots(figsize=(10, 7))
        colors = ["#0D9488" if i < 6 else "#1B3A5C" if i < 12 else "#D97706"
                  for i in range(len(importance))]
        bars = ax.barh(
            [FEATURE_NAMES[i] for i in sorted_idx],
            importance[sorted_idx],
            color=[colors[i] for i in sorted_idx]
        )
        ax.set_xlabel("Feature Importance (XGBoost gain)", fontsize=12)
        ax.set_title("GNSS Spoofing Detection — Feature Importance", fontsize=13, pad=15)
        ax.invert_yaxis()

        # Legend
        from matplotlib.patches import Patch
        legend = [
            Patch(color="#0D9488", label="Signal Quality"),
            Patch(color="#1B3A5C", label="Geometry"),
            Patch(color="#D97706", label="Cross-Satellite"),
        ]
        ax.legend(handles=legend, loc="lower right")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[XGBoost] Feature importance plot saved: {save_path}")

    def explain_prediction(self, X_sample, save_path="results/shap_explanation.png"):
        """SHAP waterfall plot for a single prediction."""
        if self.explainer is None:
            print("[XGBoost] No explainer — call train() first.")
            return
        shap_values = self.explainer(X_sample)
        fig = plt.figure(figsize=(10, 5))
        shap.plots.waterfall(shap_values[0], max_display=14, show=False)
        plt.title("SHAP Explanation — Why This Signal Was Flagged", pad=10)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[XGBoost] SHAP explanation saved: {save_path}")
