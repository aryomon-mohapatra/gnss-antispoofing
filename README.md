# GNSS Anti-Spoofing AI Detection System
**Team: DEBUG THUGS | IIT Bombay**
**Kaizen × ARIES (IIT Delhi) × NyneOS Technologies | March 2026**

---

## Overview

An ML-driven multi-feature anomaly detection framework that detects GNSS spoofing attacks in real time using a hybrid **LSTM Autoencoder + XGBoost ensemble**.

```
Raw GNSS Signal
       │
       ▼
Feature Extraction (18 features)
  ├── Signal Quality  (C/N₀, AGC, cycle slips)
  ├── Geometry        (PDOP/HDOP jumps, Doppler residuals)
  └── Cross-Satellite (Pseudorange residuals, multipath)
       │
       ├──▶ LSTM Autoencoder ──▶ Reconstruction Error Score
       │
       └──▶ XGBoost Classifier ──▶ P(spoofed)
                     │
                     ▼
           Weighted Ensemble Decision
           SPOOFED / GENUINE + Confidence
```

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/gnss-antispoofing
cd gnss-antispoofing
pip install -r requirements.txt

# 2. Download TEXBAT dataset (see data/README.md)

# 3. Extract features
python src/feature_extractor.py --input data/raw/ --output data/features/

# 4. Train models
python src/train.py --config config.yaml

# 5. Evaluate
python src/evaluate.py --config config.yaml

# 6. Run live demo
python src/demo.py --scenario data/raw/texbat_scenario1.csv
```

---

## File Structure

```
├── config.yaml                  # All hyperparameters
├── requirements.txt
├── README.md
├── src/
│   ├── feature_extractor.py     # 18-feature extraction from GNSS observables
│   ├── dataset.py               # Dataset loader + train/val/test split
│   ├── lstm_autoencoder.py      # LSTM Autoencoder model
│   ├── xgboost_classifier.py    # XGBoost classifier + SHAP
│   ├── ensemble.py              # Weighted ensemble fusion
│   ├── train.py                 # Training pipeline
│   ├── evaluate.py              # Full metrics + plots
│   └── demo.py                  # Live simulation demo
├── data/
│   ├── README.md                # Dataset download instructions
│   └── sample/                  # Small sample for testing
├── models/                      # Saved checkpoints (git-ignored)
├── results/                     # Output metrics and plots
└── notebooks/
    └── exploration.ipynb        # EDA and visualization
```

---

## Results (TEXBAT Dataset)

| Metric | Value |
|--------|-------|
| Detection Rate (Recall) | ≥ 97% |
| False Alarm Rate | ≤ 2% |
| Macro F1-Score | ≥ 0.96 |
| AUC-ROC | ≥ 0.98 |
| Inference Latency | < 15ms/epoch |

---

## Dataset

We use the **Texas Spoofing Test Battery (TEXBAT)** — 8 real-world spoofing scenarios.
See `data/README.md` for download instructions.
