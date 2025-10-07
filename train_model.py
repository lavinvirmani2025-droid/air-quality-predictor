# AI-Based Air Quality Predictor - Class XI Capstone (Compressed Model)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load cleaned dataset
data = pd.read_csv("clean_air_quality.csv")  # Ensure this file exists

# Drop missing values
data = data.dropna()

# Feature columns (independent)
X = data[['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']]
# Target column (dependent)
y = data['AQI']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest with fewer trees to reduce size
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae:.2f}, R2 Score: {r2:.2f}")

# Save model with maximum compression
joblib.dump(model, "air_quality_model.pkl", compress=9)
print("✅ Model saved as 'air_quality_model.pkl' with max compression")
