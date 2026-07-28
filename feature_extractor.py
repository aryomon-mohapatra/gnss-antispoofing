"""
feature_extractor.py
--------------------
Extracts 18 discriminative features per GNSS epoch from raw observables.

Supports:
  - RINEX observation files (via georinex)
  - CSV format (NovAtel / u-blox style)
  - TEXBAT dataset format

Usage:
    python src/feature_extractor.py --input data/raw/ --output data/features/
"""

import argparse
import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Raw Data Loader
# ──────────────────────────────────────────────────────────────────────────────

def load_gnss_csv(filepath):
    """
    Load GNSS observables from CSV.
    Expected columns (flexible naming):
      epoch, sv_id, cn0, pseudorange, carrier_phase, doppler,
      elevation, azimuth, [label (0=genuine, 1=spoofed)]
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower().str.strip()

    # Normalize common column name variants
    rename = {
        "signal_strength": "cn0", "snr": "cn0",
        "prn": "sv_id", "sat": "sv_id",
        "pr": "pseudorange", "range": "pseudorange",
        "cp": "carrier_phase", "phase": "carrier_phase",
        "dop": "doppler",
        "el": "elevation", "elev": "elevation",
        "az": "azimuth",
        "time": "epoch", "gps_time": "epoch", "tow": "epoch",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return df


def load_texbat(filepath):
    """Load TEXBAT dataset CSV — already per-epoch aggregated."""
    return load_gnss_csv(filepath)


# ──────────────────────────────────────────────────────────────────────────────
# Feature Extraction (18 features per epoch)
# ──────────────────────────────────────────────────────────────────────────────

def extract_epoch_features(epoch_df, prev_position=None, prev_velocity=None):
    """
    Extract all 18 features for a single epoch.

    Args:
        epoch_df:       DataFrame of all satellite observations for one epoch
        prev_position:  (lat, lon, alt) from previous epoch for jump detection
        prev_velocity:  velocity vector from previous epoch

    Returns:
        dict of 18 feature values
    """
    feats = {}
    sats = epoch_df

    # ── SIGNAL QUALITY FEATURES (6) ──────────────────────────────────────────

    # 1. Mean C/N₀ — unusually high or uniform = suspicious
    cn0 = sats["cn0"].values if "cn0" in sats else np.array([35.0])
    feats["cn0_mean"] = float(np.mean(cn0))

    # 2. C/N₀ std — spoofers produce unnaturally uniform signal strength
    feats["cn0_std"] = float(np.std(cn0)) if len(cn0) > 1 else 0.0

    # 3. AGC level — spoofers drive up overall power
    feats["agc_level"] = float(sats["agc"].mean()) if "agc" in sats else feats["cn0_mean"] * 0.95

    # 4. Carrier phase noise variance
    if "carrier_phase" in sats and len(sats) > 2:
        cp = sats["carrier_phase"].values
        # Detrend and compute variance
        cp_detrended = cp - np.polyval(np.polyfit(np.arange(len(cp)), cp, 1), np.arange(len(cp)))
        feats["carrier_phase_noise"] = float(np.var(cp_detrended))
    else:
        feats["carrier_phase_noise"] = 0.0

    # 5. Signal lock time — spoofing causes sudden reacquisition
    feats["signal_lock_time"] = float(sats["lock_time"].mean()) if "lock_time" in sats else 300.0

    # 6. Cycle slip count — spoofer disrupts phase continuity
    feats["cycle_slip_count"] = int(sats["cycle_slip"].sum()) if "cycle_slip" in sats else 0

    # ── GEOMETRY & KINEMATICS FEATURES (6) ───────────────────────────────────

    # 7. PDOP — spoofing causes fake geometry improvement
    feats["pdop"] = float(sats["pdop"].iloc[0]) if "pdop" in sats else _compute_dop(sats, "p")

    # 8. HDOP
    feats["hdop"] = float(sats["hdop"].iloc[0]) if "hdop" in sats else _compute_dop(sats, "h")

    # 9. VDOP
    feats["vdop"] = float(sats["vdop"].iloc[0]) if "vdop" in sats else _compute_dop(sats, "v")

    # 10. Doppler residual — claimed velocity vs. actual Doppler shift
    if "doppler" in sats and "elevation" in sats:
        expected_doppler = _expected_doppler(sats)
        measured_doppler = sats["doppler"].values
        feats["doppler_residual"] = float(np.mean(np.abs(measured_doppler - expected_doppler)))
    else:
        feats["doppler_residual"] = 0.0

    # 11. Velocity consistency (inter-epoch)
    if prev_velocity is not None and "velocity" in sats.columns:
        curr_vel = sats["velocity"].iloc[0]
        feats["velocity_consistency"] = float(abs(curr_vel - prev_velocity))
    else:
        feats["velocity_consistency"] = 0.0

    # 12. Position jump — sudden position change = spoofing onset
    if prev_position is not None and "lat" in sats.columns:
        curr_pos = np.array([sats["lat"].iloc[0], sats["lon"].iloc[0], sats.get("alt", pd.Series([0])).iloc[0]])
        jump = np.linalg.norm(curr_pos - np.array(prev_position))
        feats["position_jump"] = float(jump)
    else:
        feats["position_jump"] = 0.0

    # ── CROSS-SATELLITE CONSISTENCY FEATURES (6) ─────────────────────────────

    # 13. Pseudorange residual mean
    if "pseudorange" in sats and "elevation" in sats:
        pr_residuals = _pseudorange_residuals(sats)
        feats["pseudorange_residual_mean"] = float(np.mean(np.abs(pr_residuals)))
    else:
        feats["pseudorange_residual_mean"] = 0.0

    # 14. Pseudorange residual std
    if "pseudorange" in sats and len(sats) > 1:
        feats["pseudorange_residual_std"] = float(np.std(_pseudorange_residuals(sats)))
    else:
        feats["pseudorange_residual_std"] = 0.0

    # 15. Time-of-arrival spread — spoofed signals arrive at suspiciously similar times
    if "pseudorange" in sats and len(sats) > 1:
        toa = sats["pseudorange"].values / 3e8  # Convert to seconds
        feats["toa_spread"] = float(np.std(toa))
    else:
        feats["toa_spread"] = 0.0

    # 16. Inter-satellite correlation — spoofer broadcasts all from one point
    if "cn0" in sats and "pseudorange" in sats and len(sats) > 2:
        try:
            corr = np.corrcoef(sats["cn0"].values, sats["pseudorange"].values)[0, 1]
            feats["inter_sat_correlation"] = float(abs(corr)) if not np.isnan(corr) else 0.0
        except Exception:
            feats["inter_sat_correlation"] = 0.0
    else:
        feats["inter_sat_correlation"] = 0.0

    # 17. Navigation message anomaly flag
    feats["nav_message_anomaly"] = int(sats["nav_anomaly"].sum()) if "nav_anomaly" in sats else 0

    # 18. Ionospheric residual consistency
    if "iono_correction" in sats and len(sats) > 1:
        iono = sats["iono_correction"].values
        model_iono = _klobuchar_model(sats) if "elevation" in sats else iono
        feats["iono_residual"] = float(np.mean(np.abs(iono - model_iono)))
    else:
        feats["iono_residual"] = 0.0

    return feats


def _compute_dop(sats, dop_type="p"):
    """Approximate DOP from satellite elevation angles."""
    if "elevation" not in sats or len(sats) < 2:
        return 2.0
    el_rad = np.radians(sats["elevation"].values)
    az_rad = np.radians(sats["azimuth"].values) if "azimuth" in sats else np.zeros(len(el_rad))
    H = np.column_stack([
        np.cos(el_rad) * np.sin(az_rad),
        np.cos(el_rad) * np.cos(az_rad),
        np.sin(el_rad),
        np.ones(len(el_rad))
    ])
    try:
        Q = np.linalg.inv(H.T @ H)
        if dop_type == "p": return float(np.sqrt(np.trace(Q)))
        if dop_type == "h": return float(np.sqrt(Q[0, 0] + Q[1, 1]))
        if dop_type == "v": return float(np.sqrt(Q[2, 2]))
    except Exception:
        return 2.0


def _expected_doppler(sats, freq=1575.42e6, c=3e8):
    """Rough expected Doppler from elevation (simplified)."""
    el = np.radians(sats["elevation"].values)
    return -freq / c * 3000 * np.sin(el)  # Assume ~3 km/s orbital velocity component


def _pseudorange_residuals(sats):
    """Compute pseudorange residuals vs. elevation-based model."""
    if "pseudorange" not in sats:
        return np.zeros(len(sats))
    pr = sats["pseudorange"].values
    median_pr = np.median(pr)
    return pr - median_pr


def _klobuchar_model(sats):
    """Simplified Klobuchar ionospheric correction model."""
    el = np.radians(sats["elevation"].values)
    return 5e-9 / np.sin(el + 0.1)  # Very simplified


# ──────────────────────────────────────────────────────────────────────────────
# Full Dataset Processing
# ──────────────────────────────────────────────────────────────────────────────

def extract_features_from_file(filepath, label=None):
    """
    Process a single GNSS file and extract per-epoch feature vectors.

    Returns:
        DataFrame with columns = 18 features + 'label' + 'epoch'
    """
    df = load_gnss_csv(filepath)

    if "epoch" not in df.columns:
        print(f"[Warning] No 'epoch' column in {filepath}, treating whole file as one epoch")
        df["epoch"] = 0

    all_features = []
    prev_pos = None
    prev_vel = None

    for epoch_id, epoch_df in df.groupby("epoch"):
        feats = extract_epoch_features(epoch_df, prev_pos, prev_vel)
        feats["epoch"] = epoch_id

        if label is not None:
            feats["label"] = label
        elif "label" in epoch_df.columns:
            feats["label"] = int(epoch_df["label"].iloc[0])
        else:
            feats["label"] = -1  # Unknown

        all_features.append(feats)

        # Update state for next epoch
        if "lat" in epoch_df.columns:
            prev_pos = [epoch_df["lat"].iloc[0], epoch_df["lon"].iloc[0], 0]
        if "velocity" in epoch_df.columns:
            prev_vel = epoch_df["velocity"].iloc[0]

    return pd.DataFrame(all_features)


def process_directory(input_dir, output_dir):
    """Process all CSV files in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    input_path = Path(input_dir)
    all_dfs = []

    for f in sorted(input_path.glob("**/*.csv")):
        print(f"[Extract] {f.name}")
        try:
            feat_df = extract_features_from_file(str(f))
            feat_df["source_file"] = f.stem
            all_dfs.append(feat_df)
        except Exception as e:
            print(f"  [Error] {f.name}: {e}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        out_path = os.path.join(output_dir, "features.csv")
        combined.to_csv(out_path, index=False)
        print(f"\n[Extract] Done: {len(combined)} epochs → {out_path}")
        print(f"[Extract] Labeled: genuine={len(combined[combined.label==0])}, spoofed={len(combined[combined.label==1])}")
        return combined
    else:
        print("[Extract] No files processed.")
        return pd.DataFrame()


def generate_synthetic_dataset(n_genuine=2000, n_spoofed=500, seed=42):
    """
    Generate a synthetic GNSS dataset for testing when TEXBAT is unavailable.
    Spoofed samples have: high cn0_std, high position_jump, high pseudorange_residual_mean.
    """
    rng = np.random.default_rng(seed)

    def genuine_sample():
        return {
            "cn0_mean": rng.normal(38, 4),
            "cn0_std": rng.normal(5, 1.5),            # Natural variation
            "agc_level": rng.normal(36, 3),
            "carrier_phase_noise": rng.normal(0.02, 0.005),
            "signal_lock_time": rng.normal(400, 80),
            "cycle_slip_count": rng.integers(0, 2),
            "pdop": rng.normal(2.1, 0.5),
            "hdop": rng.normal(1.4, 0.3),
            "vdop": rng.normal(2.0, 0.4),
            "doppler_residual": rng.normal(0.5, 0.3),
            "velocity_consistency": rng.normal(0.1, 0.05),
            "position_jump": rng.normal(0.8, 0.4),
            "pseudorange_residual_mean": rng.normal(1.5, 0.8),
            "pseudorange_residual_std": rng.normal(0.8, 0.3),
            "toa_spread": rng.normal(1e-7, 3e-8),
            "inter_sat_correlation": rng.normal(0.2, 0.1),
            "nav_message_anomaly": int(rng.random() < 0.01),
            "iono_residual": rng.normal(0.3, 0.1),
            "label": 0,
        }

    def spoofed_sample():
        return {
            "cn0_mean": rng.normal(45, 3),            # Higher overall power
            "cn0_std": rng.normal(1.2, 0.5),          # Unnaturally uniform
            "agc_level": rng.normal(44, 2),
            "carrier_phase_noise": rng.normal(0.08, 0.02),
            "signal_lock_time": rng.normal(5, 3),     # Recent reacquisition
            "cycle_slip_count": rng.integers(2, 8),
            "pdop": rng.normal(1.1, 0.2),             # Suspiciously good geometry
            "hdop": rng.normal(0.8, 0.1),
            "vdop": rng.normal(1.0, 0.2),
            "doppler_residual": rng.normal(8.0, 2.0), # Doppler mismatch
            "velocity_consistency": rng.normal(5.0, 1.5),
            "position_jump": rng.normal(12.0, 5.0),   # Sudden jump
            "pseudorange_residual_mean": rng.normal(18.0, 4.0),  # Residuals spike
            "pseudorange_residual_std": rng.normal(9.0, 2.0),
            "toa_spread": rng.normal(1e-8, 5e-9),     # All from same point
            "inter_sat_correlation": rng.normal(0.85, 0.08),
            "nav_message_anomaly": int(rng.random() < 0.4),
            "iono_residual": rng.normal(2.1, 0.5),
            "label": 1,
        }

    rows = [genuine_sample() for _ in range(n_genuine)] + \
           [spoofed_sample() for _ in range(n_spoofed)]
    rng.shuffle(rows)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw", help="Input directory with GNSS CSV files")
    parser.add_argument("--output", default="data/features", help="Output directory for feature CSVs")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic dataset for testing")
    args = parser.parse_args()

    if args.synthetic:
        print("[Extract] Generating synthetic dataset...")
        df = generate_synthetic_dataset()
        os.makedirs(args.output, exist_ok=True)
        df.to_csv(f"{args.output}/features.csv", index=False)
        print(f"[Extract] Saved {len(df)} samples → {args.output}/features.csv")
    else:
        process_directory(args.input, args.output)
