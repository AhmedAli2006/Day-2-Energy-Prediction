#!/usr/bin/env python3
"""
predict_energy.py
-----------------
Predict appliance energy consumption from new sensor data using a trained CatBoost model.

Usage:
    python predict_energy.py --input ../data/new_data.csv --output ../data/predictions.csv
"""

import argparse
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor

# ============================================
# 🔧 Step 1. Argument parsing
# ============================================
parser = argparse.ArgumentParser(description="Predict appliance energy consumption")
parser.add_argument("--input", required=True, help="Path to input CSV with sensor data")
parser.add_argument("--output", default="../data/predictions.csv", help="Path to save output predictions")
parser.add_argument("--model", default="../models/energy_model_catboost.joblib", help="Path to trained model file")
args = parser.parse_args()

# ============================================
# 🧠 Step 2. Load data and model
# ============================================
print("📥 Loading model and input data...")
model = joblib.load(args.model)
df = pd.read_csv(args.input)

# ============================================
# 🕒 Step 3. Preprocess input (must match training)
# ============================================
print("⚙️ Preprocessing data...")

# Convert datetime
df['date'] = pd.to_datetime(df['date'])
df['hour'] = df['date'].dt.hour
df['minute'] = df['date'].dt.minute
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month

# Cyclic encoding
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)

# Drop unused columns
drop_cols = ['date', 'hour', 'minute']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# ============================================
# ⏳ Step 4. Add lag and rolling features (if possible)
# ============================================
# NOTE: Only works if dataset has continuous readings (like in training)
if 'Appliances' in df.columns:
    for lag in [1, 2, 3, 6]:
        df[f'Appliances_lag_{lag}'] = df['Appliances'].shift(lag)
    df['rolling_mean_6'] = df['Appliances'].rolling(window=6).mean()
    df['rolling_std_6'] = df['Appliances'].rolling(window=6).std()
    df = df.dropna().reset_index(drop=True)
else:
    print("⚠️ No 'Appliances' column found – skipping lag features (only predicting future readings).")

# ============================================
# 🤖 Step 5. Predict
# ============================================
print("🔮 Generating predictions...")
preds = model.predict(df)

# ============================================
# 💾 Step 6. Save predictions
# ============================================
df['Predicted_Appliances'] = preds
df.to_csv(args.output, index=False)
print(f"✅ Predictions saved to: {args.output}")
print(df[['Predicted_Appliances']].head())

