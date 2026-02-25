"""
SmartRoom AI - Occupancy Classification (Bonus Model)
Random Forest classifier for occupancy detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("SMARTROOM AI - OCCUPANCY CLASSIFICATION")
print("Random Forest Classifier (Bonus Model)")
print("="*70)

# Load data
print("\n[Step 1/8] Loading dataset...")
df = pd.read_csv('data/processed/occupancy_combined.csv')
print(f"  ✓ Loaded: {len(df):,} observations")

# Prepare features
print("\n[Step 2/8] Preparing features...")
feature_cols = ['Temperature', 'Humidity', 'Light', 'CO2', 'hour', 'day_of_week', 'is_weekend']
X = df[feature_cols]
y = df['Occupancy']
print(f"  Features: {', '.join(feature_cols)}")
print(f"  Target: Occupancy (binary: 0 or 1)")

# Train-test split
print("\n[Step 3/8] Train-test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train):,} samples (80%)")
print(f"  Test: {len(X_test):,} samples (20%)")

# Train Random Forest
print("\n[Step 4/8] Training Random Forest...")
print(f"  n_estimators: 100 trees")
print(f"  max_depth: 20")

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
print(f"  ✓ Training complete")

# Evaluate
print("\n[Step 5/8] Evaluating model...")
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n{classification_report(y_test, y_pred)}")
print(f"  Accuracy: {accuracy:.1%}")

if accuracy >= 0.90:
    print(f"  ✓ Accuracy = {accuracy:.1%} (Target: >90%) - PASSED")
else:
    print(f"  ⚠ Accuracy = {accuracy:.1%} (Target: >90%) - REVIEW NEEDED")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\n  Confusion Matrix:")
print(f"       Predicted")
print(f"         0     1")
print(f"  Actual")
print(f"    0   {cm[0,0]:4d}  {cm[0,1]:4d}")
print(f"    1   {cm[1,0]:4d}  {cm[1,1]:4d}")

# Feature importance
print("\n[Step 6/8] Feature importance...")
importances = rf.feature_importances_
for i, (feat, imp) in enumerate(zip(feature_cols, importances), 1):
    print(f"  {i}. {feat}: {imp:.3f}")

# Save model
print("\n[Step 7/8] Saving outputs...")
import os
os.makedirs('models', exist_ok=True)
joblib.dump(rf, 'models/occupancy_classifier.pkl')
print(f"  ✓ Model saved: models/occupancy_classifier.pkl")

# Save confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix (Accuracy: {accuracy:.1%})')
plt.tight_layout()
plt.savefig('visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Confusion matrix saved")

# Save feature importance
plt.figure(figsize=(10, 6))
sorted_idx = np.argsort(importances)[::-1]
plt.bar(range(len(importances)), importances[sorted_idx])
plt.xticks(range(len(importances)), [feature_cols[i] for i in sorted_idx], rotation=45)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance - Occupancy Prediction')
plt.tight_layout()
plt.savefig('visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Feature importance saved")

print("\n[Step 8/8] Summary...")
print(f"  Model: Random Forest (100 trees)")
print(f"  Accuracy: {accuracy:.1%}")
print(f"  Top 3 features: {', '.join([feature_cols[i] for i in sorted_idx[:3]])}")

print("\n" + "="*70)
print("✅ Random Forest classifier complete!")
print("="*70)
