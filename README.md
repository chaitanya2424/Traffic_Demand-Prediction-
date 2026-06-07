<div align="center">

# 🚦 Traffic Demand Prediction
### End-to-End Machine Learning on Geospatial Time-Series Data

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-2ca02c?style=for-the-badge)](https://lightgbm.readthedocs.io)
[![CatBoost](https://img.shields.io/badge/CatBoost-Yandex-FF6600?style=for-the-badge)](https://catboost.ai)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Toolkit-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Score](https://img.shields.io/badge/Competition%20Score-97.53%20%2F%20100-brightgreen?style=for-the-badge)](.)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

<br/>

> **Predicting normalised traffic demand across 1,249 geospatial road segments using temporal lag features, target encoding, and gradient boosting — achieving a final R² score of 97.53/100.**

<br/>

[📖 Read the Full Report](#-documentation) · [🚀 Quick Start](#-quick-start) · [🔬 Methodology](#-methodology) · [📊 Results](#-results)

</div>

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Key Findings](#-key-findings)
- [Methodology](#-methodology)
- [Feature Engineering](#-feature-engineering)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Repository Structure](#-repository-structure)
- [Quick Start](#-quick-start)
- [Documentation](#-documentation)
- [Lessons Learned](#-lessons-learned)

---

## 🗺 Project Overview

This project solves a **real-world traffic demand forecasting problem**: given historical observations of road segments at 15-minute intervals, predict the normalised traffic demand (0–1) for future time slots across 1,249 geographically encoded locations.

The core challenge is not building a model — it is building the **right features**. Initial models scored ~79/100. After root-cause analysis revealed the model was catastrophically failing on high-demand events (R² = −8.35 on the top demand bucket), a momentum-based lag feature architecture was designed that lifted the score to **97.53/100**.

| Attribute | Detail |
|---|---|
| **Problem Type** | Supervised Regression — Time-Series Forecasting |
| **Evaluation Metric** | `max(0, 100 × R²(actual, predicted))` |
| **Training Rows** | 77,299 |
| **Test Rows** | 41,778 |
| **Unique Locations** | 1,249 geohash-encoded road segments |
| **Temporal Resolution** | 15-minute intervals |
| **Final Score** | **97.53 / 100** |

---

## 📦 Dataset

The dataset captures traffic observations across a city grid encoded with [Geohash](https://en.wikipedia.org/wiki/Geohash) (precision 6 ≈ 1.2km × 0.6km cells).

### Column Reference

| Column | Type | Description | Notes |
|---|---|---|---|
| `geohash` | `string` | Geohash-encoded lat/lon location | 1,249 unique values |
| `day` | `int` | Day number | Only days 48 & 49 in dataset |
| `timestamp` | `string` | `H:MM` format, 15-min intervals | e.g. `"10:15"`, `"23:45"` |
| `demand` | `float [0,1]` | **TARGET** — normalised traffic demand | Skew=3.73, Kurtosis=17.33 |
| `RoadType` | `categorical` | `Highway / Street / Residential` | 0.78% missing |
| `NumberofLanes` | `int (1–5)` | Physical road capacity | Sharp jump in demand at 4+ lanes |
| `LargeVehicles` | `categorical` | `Allowed / Not Allowed` | — |
| `Landmarks` | `categorical` | `Yes / No` | — |
| `Temperature` | `float (°C)` | Ambient temperature | 3.23% missing |
| `Weather` | `categorical` | `Sunny / Rainy / Foggy / Snowy` | 1.03% missing |

### Data Split Structure

```
Day 48  ──────────────────────────────────────────────────── (slots 0–95, all 24 hours)
           Training Base (69,427 rows)

Day 49  ─────── (slots 0–8, hours 0–2)    ┊ ───────────────── (slots 9–55, hours 2–13)
           Holdout Validation (7,872 rows) ┊  Test Set (41,778 rows) ← PREDICT THESE
                                           ┊
                             Boundary: slot 9 (02:15)
```

> ⚠️ **Critical insight**: Day 49 train rows (hours 0–2) are not just a validation set — they are the **momentum source** for test predictions. Test slot 9's `lag_1` feature retrieves Day49 slot 8 actual demand.

---

## 💡 Key Findings

### 1. The Failure Was Invisible in Aggregate Metrics

Initial submission scored **78.99/100** — seemingly reasonable. But demand-bucket error analysis revealed:

| Demand Bucket | Range | N (rows) | v1 R² | v2 R² |
|---|---|---|---|---|
| Very Low | 0–10% | 5,370 | **−0.246** | 0.997 |
| Low-Mid | 10–30% | 1,922 | **−2.167** | 0.953 |
| High | 30–60% | 419 | **−5.791** | 0.342 |
| Very High | 60–100% | 161 | **−8.346** | 0.385 |

> The model was performing **8× worse than predicting the mean** on the most important demand events.

### 2. Lag Features Are the Dominant Signal

```python
lag_corr = {
    'lag_1  (15 min ago)': 0.972,   # ← Strongest single feature
    'lag_2  (30 min ago)': 0.955,
    'lag_4  (60 min ago)': 0.916,
    'lag_8  (2 hrs ago)' : 0.809,
    'lag_96 (24 hrs ago)': 0.792,
}
```

### 3. Root Cause of High-Demand Failures

Certain geohashes showed `demand = 1.0` on Day49 hours 0–2 (an anomalous event), but their Day48 pattern at the same hours was only `0.2–0.4`. Without lag features, the model predicted history. With lag features, it correctly reads: *"demand was 1.0 here 15 minutes ago → predict 1.0 now."*

---

## 🔬 Methodology

```
Raw Data
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  1. TIMESTAMP PARSING                                               │
│     H:MM → hour (0-23), minute (0,15,30,45), slot (0-95)           │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  2. LOOKUP TABLE CONSTRUCTION                                       │
│     • geo_slot_d48:      geohash × slot → demand  [Day 48]         │
│     • geo_slot_d49:      geohash × slot → demand  [Day 49 train]   │
│     • geo_slot_combined: Day49 overrides Day48 (no leakage)        │
│     • geo_hour_stats:    geohash × hour → mean/std/max  [Day 48]   │
│     • geo_evening:       geohash hrs 20-23 → mean/max  [Day 48]    │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  3. FEATURE ENGINEERING                                             │
│     Lag Features  │ Target Encoding │ Cyclical Time │ Road Features │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4. VALIDATION STRATEGY                                             │
│     Phase 1: Train on Day48 → Evaluate on Day49 (time-split)       │
│     Phase 2: Retrain on ALL data (Day48 + Day49) → Predict test    │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  5. MODELLING                                                       │
│     Ridge (baseline) → LightGBM → CatBoost → Weighted Ensemble     │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
  submission_v2.csv
```

---

## 🛠 Feature Engineering

37 features across 6 tiers, ordered by predictive power:

### Tier 1 — Momentum (Lag Features)

The single most impactful feature group. Built from a **combined Day49+Day48 lookup** to propagate real-time state into test predictions.

```python
for lag in [1, 2, 4, 8, 12, 96]:
    lag_slot = (df['slot'] - lag) % 96
    keys     = zip(df['geohash'], lag_slot)
    df[f'lag_{lag}'] = [combined_lookup.get(k, fallback) for k in keys]

# Derived momentum signals
df['lag_ewm']   = 0.50*df['lag_1'] + 0.25*df['lag_2'] + 0.15*df['lag_4'] + 0.10*df['lag_8']
df['lag_trend'] = df['lag_1'] - df['lag_4']   # Rising (+) or falling (-)
df['lag_max_4'] = df[['lag_1','lag_2','lag_4','lag_8']].max(axis=1)
```

### Tier 2 — Historical Profile (Target Encoding)

```python
# Computed on Day 48 only — no leakage into validation/test
geo_hour_stats = d48.groupby(['geohash', 'hour'])['demand'].agg(['mean', 'std', 'max'])

# Applied via lookup (not join) to prevent cardinality explosion
df['geo_hour_mean'] = [geo_hour_stats['mean'].get((g, h)) for g, h in zip(df['geohash'], df['hour'])]
```

### Tier 3 — Cyclical Time Encoding

```python
# Trees don't know hour 23 ≈ hour 0 without this
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['slot_sin'] = np.sin(2 * np.pi * df['slot'] / 96)
df['slot_cos'] = np.cos(2 * np.pi * df['slot'] / 96)
```

### Tier 4 — Road Features

```python
df['is_highway'] = (df['RoadType'] == 'Highway').astype(int)
df['is_street']  = (df['RoadType'] == 'Street').astype(int)
df['wide_road']  = (df['NumberofLanes'] >= 4).astype(int)  # Threshold effect at 4+ lanes

# Safe interaction: 3 road types × 24 hours = 72 combinations
# (Geohash × Hour = 1,249 × 24 = 29,976 combos → AVOIDED: memorisation risk)
df['roadtype_hour'] = df['RoadType'].fillna('Unknown') + '_' + df['hour'].astype(str)
```

### Tier 5 — Geo Summary Statistics

```python
df['geo_d48_mean']  = df['geohash'].map(geo_all['geo_d48_mean'])
df['geo_eve_mean']  = df['geohash'].map(geo_evening['mean'])   # Hours 20-23 = most recent
df['geo_frequency'] = df['geohash'].map(train['geohash'].value_counts())
```

### Tier 6 — Missing Value Imputation

```python
# Temperature: per-weather-type median (preserves physical relationship)
temp_med = train.groupby('Weather')['Temperature'].median()
df.loc[df['Temperature'].isna(), 'Temperature'] = \
    df.loc[df['Temperature'].isna(), 'Weather'].map(temp_med)

# Categorical NaNs: labelled as '__MISSING__' — tree models learn from missingness
df['RoadType'].fillna('__MISSING__', inplace=True)
```

---

## 🤖 Model Architecture

### LightGBM — Primary Model

```python
params = {
    'objective'        : 'regression_l1',  # MAE: robust to right-skewed demand
    'metric'           : 'mae',
    'learning_rate'    : 0.03,
    'n_estimators'     : 5000,             # With early stopping
    'max_depth'        : 8,                # Increased from 6 — lag features justify depth
    'num_leaves'       : 127,              # 2^7 - 1
    'min_child_samples': 20,               # Prevent leaf-level geohash memorisation
    'feature_fraction' : 0.75,
    'bagging_fraction' : 0.80,
    'lambda_l1'        : 0.05,
    'lambda_l2'        : 0.10,
}

model.fit(
    X_train, y_train,
    eval_set  = [(X_val, y_val)],
    callbacks = [lgb.early_stopping(300), lgb.log_evaluation(500)]
)
```

### CatBoost — Ensemble Partner

```python
params = {
    'loss_function'       : 'MAE',
    'depth'               : 8,
    'l2_leaf_reg'         : 3.0,
    'min_data_in_leaf'    : 20,
    'early_stopping_rounds': 300,
    'cat_features'        : [geohash_enc, RoadType_enc, ...],  # Native handling
}
```

### Why These Hyperparameters?

| Parameter | Choice | Rationale |
|---|---|---|
| `objective = regression_l1` | MAE loss | Right-skewed demand → MSE would over-focus on rare peaks during training |
| `max_depth = 8` | Deeper than default | Lag features carry the signal; model doesn't need to memorise geo×hour |
| `min_child_samples = 20` | Conservative | 1,249 geohashes with varying sample counts — prevent tiny-leaf overfitting |
| `early_stopping = 300` | Patient | Learning rate=0.03 needs more rounds to converge; 300 rounds gives sufficient patience |
| `feature_fraction = 0.75` | Subsampling | Reduces tree correlation in the ensemble, improves generalisation |

### Validation Strategy

```
❌ WRONG: Random K-Fold CV
    → Mixes future data into training folds
    → Local score inflated vs. submission score

✅ CORRECT: Purged Time Split
    Train  : Day 48 (all 24 hours)
    Val    : Day 49 train (hours 0-2)    ← mirrors test conditions
    Retrain: Day 48 + Day 49 train       ← no data wasted for final model
```

---

## 📊 Results

### Competition Score Progression

| Version | Key Change | Day49 MAE | Score |
|---|---|---|---|
| Ridge Baseline | Linear model, raw features | ~0.064 | 40.4 |
| LightGBM v1 | GBDT + cyclical encoding + geo features | ~0.049 | 75.88 → **78.99** (submitted) |
| LightGBM v2 | + Lag features + combined lookup | ~0.022 | **97.53** |

### Per-Bucket R² Improvement

```
Demand Bucket    │ v1 R²   │ v2 R²   │ Change
─────────────────┼─────────┼─────────┼──────────────
Very Low  (0-10%)│  −0.246 │  +0.997 │ ▲ +1.243
Low-Mid (10-30%) │  −2.167 │  +0.953 │ ▲ +3.120
High    (30-60%) │  −5.791 │  +0.342 │ ▲ +6.133
Very High(60-100)│  −8.346 │  +0.385 │ ▲ +8.731  ← Root cause fixed
```

### Feature Importance (LightGBM — Top 10)

```
lag_1            ████████████████████████████████  (dominant)
geo_hour_mean    ████████████████████████
lag_96           ████████████████████
lag_2            ███████████████
geo_hour_max     █████████████
lag_ewm          ████████████
lag_4            ██████████
geo_d48_mean     █████████
slot_cos         ████████
geo_frequency    ███████
```

---

## 📁 Repository Structure

```
traffic-demand-prediction/
│
├── 📂 data/
│   ├── train.csv                        # 77,299 rows — historical demand logs
│   └── test.csv                         # 41,778 rows — predictions required               # Score: 97.53 ← final
│
├── 📂 reports/
│   └── Traffic_Demand_DS_Report.docx    # Full data science report (8 sections)
│
├── traffic_demand_Prediction.py                 # Complete self-contained pipeline
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
https://github.com/chaitanya2424/Traffic_Demand-Prediction-.git
cd Traffic_Demand-Prediction
pip install -r requirements.txt
```

### 2. Requirements

```txt
# requirements.txt
lightgbm>=4.0.0
catboost>=1.2.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
```

### 3. Run the Full Pipeline

```bash
# Runs the complete v2 pipeline:
#   parse → feature engineering → validation → final model → submission.csv
python Traffic_Demand_Prediction.py
```

> Expected runtime: ~4–6 minutes on a standard laptop (early stopping engages around iteration 800–1200 depending on hardware).

### 4. Expected Output

```
============================================================
STEP 7 · LightGBM — Time-Split Validation
============================================================

  Day49 holdout → Competition score = 97.53
                  R² = 0.9753  |  MAE = 0.02214
  Best iteration  : 1,147

============================================================
STEP 9 · Final Training on Full Dataset (Day48 + Day49)
============================================================
  Saved → outputs/submission_v2.csv  (41,778 rows)
```

### 5. Key Function Reference

```python
from src.features import add_features, build_lookups
from src.models   import train_lgbm, train_catboost
from src.validation import time_split_score

# Build lookup tables (must run before add_features)
lookups = build_lookups(train_df)

# Engineer features — use_combined=True for val/test rows
train_fe = add_features(train_df, lookups, use_combined=False)  # Day 48
test_fe  = add_features(test_df,  lookups, use_combined=True)   # Test

# Train with time-based validation
model, score = train_lgbm(X_d48, y_d48, X_d49, y_d49)
print(f"Holdout score: {score:.2f}")
```

---

## 📖 Documentation

A full **8-section data science report** is included in `reports/Traffic_Demand_DS_Report.docx` covering:

| Section | Content |
|---|---|
| 1. Project Overview | Problem statement, metric explanation, why R² punishes extreme errors |
| 2. Dataset Description | Column reference, data split structure, demand distribution analysis |
| 3. EDA | Temporal patterns, geospatial analysis, correlation matrix, missing values |
| 4. Feature Engineering | Complete 27-feature catalogue with formulas, tiers, and rationale |
| 5. Modelling Strategy | Validation approach comparison, hyperparameter justification |
| 6. Results & Diagnosis | Bucket error analysis, root cause identification, breakthrough explanation |
| 7. Generalisation Guide | 7-step methodology template for similar problems, red flags table |
| 8. Lessons Learned | Five key principles distilled from the project |

---

## 🧠 Lessons Learned

These principles generalise to **any high-frequency time-series regression problem**:

**1. Lag features first, always.**
Before building any model, compute `lag_1` and check its correlation with the target. If `r > 0.90`, lag features are your anchor — everything else is incremental. Here `lag_1` achieved `r = 0.972`.

**2. Segment your error analysis by target range.**
Aggregate metrics lie. A model with overall MAE = 0.049 can have `R² = −8.35` on the most important rows. Always compute per-bucket R² before declaring a model "good".

**3. Map the exact time structure before building any feature.**
Draw the slot timeline. Know which train slots precede which test slots. Verify that your lag lookup for each test slot pulls from training data only. One overlooked slot boundary costs you the whole competition.

**4. The freshest data is the most valuable.**
Day 49 train rows are not just a validation set — they are the real-time state of the traffic system. Including them in the final model (and using them as lag sources for test) was the key architectural decision.

**5. Use MAE loss for right-skewed targets.**
MSE (L2) disproportionately penalises large residuals during *training* — causing the model to over-correct on rare high-demand events and under-fit the majority of low-demand rows. MAE (L1) gives equal gradient weight across the full demand range.

---

## 🗺 Roadmap

- [ ] Add CatBoost to weighted ensemble (+1–2 score points estimated)
- [ ] Geohash prefix features for spatial smoothing of rare locations
- [ ] Quantile regression heads for uncertainty estimation on high-demand predictions
- [ ] SHAP value analysis for per-prediction interpretability
- [ ] Streamlit dashboard for real-time demand visualisation

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with precision. Debugged with curiosity. Shared for learning.**

If this project helped you, consider giving it a ⭐

</div>
