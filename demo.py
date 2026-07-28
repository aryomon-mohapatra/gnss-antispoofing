"""
demo.py
-------
Live simulation demo for online presentation.
Replays a GNSS scenario and shows real-time anomaly score visualization.

Usage:
    python src/demo.py --synthetic          # Demo with synthetic data
    python src/demo.py --scenario data/raw/texbat_scenario1.csv
"""

import argparse
import os
import sys
import time
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

sys.path.insert(0, os.path.dirname(__file__))
from feature_extractor import generate_synthetic_dataset, extract_epoch_features
from lstm_autoencoder import build_lstm_model
from xgboost_classifier import XGBoostDetector
from ensemble import GNSSEnsembleDetector
from dataset import load_scaler, FEATURE_COLS


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def run_demo(cfg, device, synthetic=True, scenario_path=None, save_gif=False):
    """
    Replays GNSS epochs one by one, updating anomaly score plot in real time.
    """
    print("[Demo] Loading models...")
    ckpt = torch.load(cfg["output"]["lstm_checkpoint"], map_location=device)
    lstm_model = build_lstm_model(cfg).to(device)
    lstm_model.load_state_dict(ckpt["model_state_dict"])
    lstm_threshold = ckpt["anomaly_threshold"]
    lstm_model.eval()

    xgb_detector = XGBoostDetector(cfg)
    xgb_detector.load(cfg["output"]["xgboost_checkpoint"])
    scaler = load_scaler(f"{cfg['output']['models_dir']}/scaler.pkl")

    ensemble = GNSSEnsembleDetector(lstm_model, xgb_detector, lstm_threshold, cfg)

    # ── Generate / load data ──────────────────────────────────────────────────
    if synthetic:
        print("[Demo] Generating synthetic spoofing scenario...")
        df = generate_synthetic_dataset(n_genuine=150, n_spoofed=50, seed=7)
        df = df.sample(frac=1, random_state=7).reset_index(drop=True)
        # Force genuine first, spoofed in middle
        genuine = df[df.label == 0].head(100)
        spoofed = df[df.label == 1].head(50)
        genuine_end = df[df.label == 0].tail(50)
        scenario_df = pd.concat([genuine, spoofed, genuine_end]).reset_index(drop=True)
    else:
        import pandas as pd
        scenario_df = pd.read_csv(scenario_path)

    import pandas as pd

    X_all = scaler.transform(scenario_df[FEATURE_COLS].fillna(0).values).astype(np.float32)
    y_all = scenario_df["label"].values if "label" in scenario_df.columns else np.zeros(len(scenario_df))
    window = cfg["data"]["window_size"]

    # ── Live Plot Setup ───────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), facecolor="#0D1B2A")
    fig.suptitle("GNSS Anti-Spoofing Live Detection — DEBUG THUGS | IIT Bombay",
                 color="white", fontsize=14, fontweight="bold", y=0.98)

    for ax in [ax1, ax2]:
        ax.set_facecolor("#1B3A5C")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#0D9488")
        ax.spines["left"].set_color("#0D9488")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    MAXLEN = 120
    epochs_buf = deque(maxlen=MAXLEN)
    scores_buf  = deque(maxlen=MAXLEN)
    labels_buf  = deque(maxlen=MAXLEN)
    cn0_buf     = deque(maxlen=MAXLEN)

    line_score, = ax1.plot([], [], color="#0D9488", lw=2, label="Anomaly Score")
    fill_genuine = ax1.fill_between([], [], 0, alpha=0.15, color="#0D9488")
    fill_spoofed = ax1.fill_between([], [], 0, alpha=0.15, color="#DC2626")
    thresh_line  = ax1.axhline(ensemble.decision_threshold, color="#F59E0B",
                               linestyle="--", lw=1.5, label=f"Threshold ({ensemble.decision_threshold:.2f})")
    ax1.set_xlim(0, MAXLEN)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Anomaly Score", color="white", fontsize=11)
    ax1.set_title("Real-Time Anomaly Score", color="#5EEAD4", fontsize=11, pad=6)
    ax1.legend(loc="upper left", facecolor="#0D1B2A", labelcolor="white", fontsize=9)

    line_cn0, = ax2.plot([], [], color="#5EEAD4", lw=1.8, label="C/N₀ mean")
    ax2.set_xlim(0, MAXLEN)
    ax2.set_ylim(25, 55)
    ax2.set_ylabel("C/N₀ (dB-Hz)", color="white", fontsize=11)
    ax2.set_xlabel("Epoch", color="white", fontsize=11)
    ax2.set_title("Signal Carrier-to-Noise Ratio", color="#5EEAD4", fontsize=11, pad=6)
    ax2.legend(loc="upper left", facecolor="#0D1B2A", labelcolor="white", fontsize=9)

    status_text = ax1.text(0.99, 0.92, "MONITORING...", transform=ax1.transAxes,
                           color="#0D9488", fontsize=12, fontweight="bold",
                           ha="right", va="top")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    def update(frame):
        nonlocal fill_genuine, fill_spoofed
        i = frame + window - 1
        if i >= len(X_all):
            return line_score, line_cn0, status_text

        X_seq = X_all[i - window + 1: i + 1][np.newaxis]  # (1, window, 18)
        X_flat = X_all[i:i+1]                               # (1, 18)

        preds, final_scores, _, _ = ensemble.predict(X_flat, X_seq, device)
        score = float(final_scores[0])
        label = int(y_all[i])

        epochs_buf.append(frame)
        scores_buf.append(score)
        labels_buf.append(label)
        cn0_buf.append(float(X_all[i, 0]) * 4 + 38)  # Approximate denormalized C/N₀

        xs = list(range(len(scores_buf)))
        ys = list(scores_buf)
        cs = list(labels_buf)

        line_score.set_data(xs, ys)
        line_cn0.set_data(xs, list(cn0_buf))

        # Recolor fill
        fill_genuine.remove()
        fill_spoofed.remove()
        xs_arr = np.array(xs)
        ys_arr = np.array(ys)
        cs_arr = np.array(cs)
        fill_genuine = ax1.fill_between(xs_arr, ys_arr, 0,
                                        where=(cs_arr == 0), alpha=0.15, color="#0D9488")
        fill_spoofed = ax1.fill_between(xs_arr, ys_arr, 0,
                                        where=(cs_arr == 1), alpha=0.25, color="#DC2626")

        # Status text
        if score >= ensemble.decision_threshold:
            status_text.set_text("⚠ SPOOFING DETECTED")
            status_text.set_color("#DC2626")
        else:
            status_text.set_text("✓  GENUINE SIGNAL")
            status_text.set_color("#0D9488")

        return line_score, line_cn0, status_text, fill_genuine, fill_spoofed

    n_frames = max(1, len(X_all) - window + 1)
    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                  interval=80, blit=False, repeat=False)

    os.makedirs(cfg["output"]["results_dir"], exist_ok=True)
    if save_gif:
        gif_path = f"{cfg['output']['results_dir']}/demo.gif"
        ani.save(gif_path, writer="pillow", fps=12)
        print(f"[Demo] Saved: {gif_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--synthetic", action="store_true", default=True)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--save_gif", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_demo(cfg, device, synthetic=args.synthetic,
             scenario_path=args.scenario, save_gif=args.save_gif)
