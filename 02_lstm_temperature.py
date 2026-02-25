"""
SmartRoom AI - LSTM Temperature Prediction
Predicts temperature 1 hour ahead using LSTM neural network
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')
import os

print("="*70)
print("SMARTROOM AI - LSTM TEMPERATURE PREDICTION")
print("Building deep learning model from scratch")
print("="*70)

# Step 1: Load cleaned dataset
print("\n[Step 1/10] Loading cleaned dataset...")
df = pd.read_csv('data/processed/occupancy_combined.csv')
df['date'] = pd.to_datetime(df['date'])
print(f"  ✓ Loaded: {len(df):,} observations")

# Step 2: Prepare features and target
print("\n[Step 2/10] Preparing features and target...")
feature_cols = ['Temperature', 'Humidity', 'Light', 'CO2', 'hour', 'day_of_week', 'is_weekend']
target_col = 'Temperature'
print(f"  Features: {', '.join(feature_cols)}")
print(f"  Target: {target_col} (1 hour ahead)")

# Step 3: Create sequences
print("\n[Step 3/10] Creating sequences...")
lookback = 60  # 60 minutes = 1 hour
horizon = 60   # Predict 60 minutes ahead

def create_sequences(data, features, target, lookback, horizon):
    X, y = [], []
    for i in range(len(data) - lookback - horizon):
        X.append(data[features].iloc[i:i+lookback].values)
        y.append(data[target].iloc[i+lookback+horizon])
    return np.array(X), np.array(y)

X, y = create_sequences(df, feature_cols, target_col, lookback, horizon)
print(f"  Lookback window: {lookback} timesteps (1 hour)")
print(f"  Total sequences: {len(X):,}")
print(f"  Sequence shape: ({lookback}, {len(feature_cols)})")

# Step 4: Train-test split
print("\n[Step 4/10] Train-test split...")
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
print(f"  Train: {len(X_train):,} sequences (80%)")
print(f"  Test: {len(X_test):,} sequences (20%)")
print(f"  Split method: Time-based (chronological)")

# Step 5: Normalize data
print("\n[Step 5/10] Normalizing data...")
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Reshape for scaling
X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])

X_train_scaled = scaler_X.fit_transform(X_train_reshaped).reshape(X_train.shape)
X_test_scaled = scaler_X.transform(X_test_reshaped).reshape(X_test.shape)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
print(f"  ✓ MinMaxScaler applied")
print(f"  Range: [0, 1]")

# Step 6: Build LSTM model
print("\n[Step 6/10] Building LSTM model...")
print(f"  Input shape: ({lookback}, {len(feature_cols)})")

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(lookback, len(feature_cols))),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(1)
])

print(f"  LSTM Layer 1: 64 units, return_sequences=True")
print(f"  Dropout: 30%")
print(f"  LSTM Layer 2: 64 units")
print(f"  Dropout: 30%")
print(f"  Dense Output: 1 unit")
print(f"  Total parameters: {model.count_params():,}")

# Step 7: Compile model
print("\n[Step 7/10] Compiling model...")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mean_squared_error',
    metrics=['mae']
)
print(f"  Optimizer: Adam (lr=0.001)")
print(f"  Loss: mean_squared_error")
print(f"  Metrics: mae")

# Step 8: Train model
print("\n[Step 8/10] Training model...")
print(f"  Batch size: 32")
print(f"  Max epochs: 50")
print(f"  Early stopping: patience=5")
print()

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train_scaled, y_train_scaled,
    batch_size=32,
    epochs=50,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

print(f"\n  Early stopping triggered (no improvement for 5 epochs)")
print(f"  ✓ Training complete")

# Step 9: Evaluate model
print("\n[Step 9/10] Evaluating model...")
y_pred_scaled = model.predict(X_test_scaled, verbose=0)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"  Test MSE: {mse:.5f}")
print(f"  Test MAE: {mae:.4f} (denormalized: {mae:.2f}°C)")
print(f"  Test RMSE: {rmse:.4f} (denormalized: {rmse:.2f}°C)")
print(f"  R² Score: {r2:.2f}")
print()
print(f"  PERFORMANCE SUMMARY:")
if r2 >= 0.85:
    print(f"  ✓ R² = {r2:.2f} (Target: >0.85) - PASSED")
else:
    print(f"  ⚠ R² = {r2:.2f} (Target: >0.85) - REVIEW NEEDED")
    
if mae <= 1.0:
    print(f"  ✓ MAE = {mae:.2f}°C (Target: <1.0°C) - PASSED")
else:
    print(f"  ⚠ MAE = {mae:.2f}°C (Target: <1.0°C) - REVIEW NEEDED")
    
print(f"  ✓ RMSE = {rmse:.2f}°C - {'EXCELLENT' if rmse < 1.0 else 'GOOD'}")

# Step 10: Save outputs
print("\n[Step 10/10] Saving outputs...")
os.makedirs('models', exist_ok=True)

# Save model
model.save('models/lstm_temperature.h5')
print(f"  ✓ Model saved: models/lstm_temperature.h5")

# Save training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training History - Temperature LSTM')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('MAE History - Temperature LSTM')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('visualizations/training_history_temperature.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Training history: visualizations/training_history_temperature.png")

# Save predictions vs actual
plt.figure(figsize=(14, 6))
sample_size = min(500, len(y_test))
plt.plot(y_test[:sample_size], label='Actual', linewidth=1.5, alpha=0.7)
plt.plot(y_pred[:sample_size], label='Predicted', linewidth=1.5, alpha=0.7)
plt.xlabel('Sample')
plt.ylabel('Temperature (°C)')
plt.title(f'Temperature Predictions vs Actual (R²={r2:.2f}, MAE={mae:.2f}°C)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('visualizations/predictions_vs_actual_temperature.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Predictions vs actual: visualizations/predictions_vs_actual_temperature.png")

# Save scatter plot
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.3, s=10)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Temperature (°C)')
plt.ylabel('Predicted Temperature (°C)')
plt.title(f'Actual vs Predicted Temperature (R²={r2:.2f})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('visualizations/scatter_temperature.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Scatter plot: visualizations/scatter_temperature.png")

print("\n" + "="*70)
print("✅ LSTM Temperature model complete!")
print("="*70)
