# 🏠 Energy Consumption Prediction (AI Project)

Predict household energy consumption using environmental and temporal data.

## 📘 Overview
This project builds machine learning models to predict appliance energy usage using temperature, humidity, and time-series data from the UCI "EnergyData Complete" dataset.

## 🚀 Features
- Full Exploratory Data Analysis (EDA)
- Feature Engineering (Cyclical Time Encoding + Lag Features)
- Model Comparison:
  - Linear Regression
  - Random Forest
  - XGBoost
  - LightGBM
  - CatBoost
- Tuned CatBoost model achieving **R² = 0.78**
- Inference script for predicting new sensor readings

## 📂 Folder Structure
Day2EnergyPrediction/
│
├── data/
│   ├── energydata_complete.csv
│   └── new_data.csv
│
├── models/
│   └── energy_model_catboost.joblib
│
├── notebooks/
│   └── Day2EnergyPrediction.ipynb
│
├── scripts/
│   └── predict_energy.py
│
├── requirements.txt
├── README.md
└── .gitignore
