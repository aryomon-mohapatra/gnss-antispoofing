# Dataset Instructions

## Primary Dataset: TEXBAT (Texas Spoofing Test Battery)

TEXBAT contains 8 real-world GNSS spoofing scenarios recorded at the University of Texas.

### Download

1. Visit: https://radionavlab.ae.utexas.edu/texbat/
2. Fill the access request form (free for research/education)
3. Download all 8 scenario files

### File Placement

```
data/
  raw/
    ds1_clean.csv          # Scenario 1: Clean (no spoofing)
    ds2_spoofing.csv       # Scenario 2: Static spoofing
    ds3_spoofing.csv       # Scenario 3: Dynamic spoofing
    ...
```

### Column Format Expected

| Column | Description |
|--------|-------------|
| `epoch` | GPS time of week (seconds) |
| `sv_id` | Satellite vehicle ID (e.g., G01, G15) |
| `cn0` | Carrier-to-noise ratio (dB-Hz) |
| `pseudorange` | Code pseudorange (meters) |
| `carrier_phase` | Carrier phase (cycles) |
| `doppler` | Doppler shift (Hz) |
| `elevation` | Satellite elevation angle (degrees) |
| `azimuth` | Satellite azimuth angle (degrees) |
| `label` | 0 = genuine, 1 = spoofed |

---

## Alternative: GPSpy Dataset

Available at: https://github.com/stanford-gps-lab/gnss-sdr-sim

---

## Synthetic Data (No Download Required)

To test the pipeline without downloading any dataset:

```bash
python src/train.py --synthetic
python src/evaluate.py --synthetic
python src/demo.py --synthetic
```

This generates a realistic synthetic dataset with 3,000 genuine and 600 spoofed samples based on known signal characteristics of each class.
