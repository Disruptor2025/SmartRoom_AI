"""
SmartRoom AI - Data Exploration and Cleaning
Loads UCI Occupancy Detection dataset, cleans data, creates visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import os

print("="*70)
print("SMARTROOM AI - DATA EXPLORATION")
print("Analyzing UCI Occupancy Detection Dataset")
print("="*70)

# Step 1: Load dataset files
print("\n[Step 1/8] Loading dataset files...")
df1 = pd.read_csv('data/raw/datatraining.txt')
print(f"  Loading datatraining.txt: {len(df1):,} rows")

df2 = pd.read_csv('data/raw/datatest.txt')
print(f"  Loading datatest.txt: {len(df2):,} rows")

df3 = pd.read_csv('data/raw/datatest2.txt')
print(f"  Loading datatest2.txt: {len(df3):,} rows")

# Combine all datasets
df = pd.concat([df1, df2, df3], ignore_index=True)
print(f"  ✓ Total rows: {len(df):,}")

# Step 2: Data cleaning
print("\n[Step 2/8] Data cleaning...")
print(f"  Checking for missing values...")
missing = df.isnull().sum().sum()
if missing == 0:
    print(f"  ✓ No missing values found")
else:
    print(f"  Found {missing} missing values, filling...")
    df = df.ffill().bfill()

print(f"  Checking for duplicates...")
duplicates = df.duplicated().sum()
if duplicates == 0:
    print(f"  ✓ No duplicates found")
else:
    print(f"  Removing {duplicates} duplicates...")
    df = df.drop_duplicates()

# Convert date column
df['date'] = pd.to_datetime(df['date'])

# Step 3: Feature engineering
print("\n[Step 3/8] Feature engineering...")
print(f"  Creating temporal features...")
df['hour'] = df['date'].dt.hour
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['month'] = df['date'].dt.month
print(f"  ✓ hour (0-23)")
print(f"  ✓ day_of_week (0-6)")
print(f"  ✓ is_weekend (0 or 1)")
print(f"  ✓ month")

print(f"  Creating derived features...")
df['temperature_gradient'] = df['Temperature'].diff()
df['humidity_gradient'] = df['Humidity'].diff()
df['co2_gradient'] = df['CO2'].diff()
print(f"  ✓ temperature_gradient")
print(f"  ✓ humidity_gradient")
print(f"  ✓ co2_gradient")

# Fill NaN values from gradients
df = df.fillna(0)

# Step 4: Statistical analysis
print("\n[Step 4/8] Statistical analysis...")
print(f"  Temperature: mean={df['Temperature'].mean():.2f}°C, "
      f"std={df['Temperature'].std():.2f}°C, "
      f"range=[{df['Temperature'].min():.2f}-{df['Temperature'].max():.2f}°C]")
print(f"  Humidity: mean={df['Humidity'].mean():.2f}%, "
      f"std={df['Humidity'].std():.2f}%, "
      f"range=[{df['Humidity'].min():.2f}-{df['Humidity'].max():.2f}%]")
print(f"  CO2: mean={df['CO2'].mean():.2f}ppm, "
      f"std={df['CO2'].std():.2f}ppm, "
      f"range=[{df['CO2'].min():.2f}-{df['CO2'].max():.2f}ppm]")
print(f"  Light: mean={df['Light'].mean():.2f}lux, "
      f"std={df['Light'].std():.2f}lux, "
      f"range=[{df['Light'].min():.2f}-{df['Light'].max():.2f}lux]")

occupancy_counts = df['Occupancy'].value_counts()
print(f"  Occupancy: 0={occupancy_counts[0]:,} ({occupancy_counts[0]/len(df)*100:.1f}%), "
      f"1={occupancy_counts[1]:,} ({occupancy_counts[1]/len(df)*100:.1f}%)")

# Step 5: Create visualizations
print("\n[Step 5/8] Creating visualizations...")
os.makedirs('visualizations', exist_ok=True)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# 1. Temperature over time
print("  [1/15] Temperature over time...", end=' ')
plt.figure()
plt.plot(df['date'], df['Temperature'], linewidth=0.5, alpha=0.7)
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Over Time')
plt.tight_layout()
plt.savefig('visualizations/temp_over_time.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓")

# 2. Humidity over time
print("  [2/15] Humidity over time...", end=' ')
plt.figure()
plt.plot(df['date'], df['Humidity'], linewidth=0.5, alpha=0.7, color='green')
plt.xlabel('Date')
plt.ylabel('Humidity (%)')
plt.title('Humidity Over Time')
plt.tight_layout()
plt.savefig('visualizations/humidity_over_time.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓")

# 3. CO2 over time
print("  [3/15] CO2 over time...", end=' ')
plt.figure()
plt.plot(df['date'], df['CO2'], linewidth=0.5, alpha=0.7, color='orange')
plt.xlabel('Date')
plt.ylabel('CO2 (ppm)')
plt.title('CO2 Concentration Over Time')
plt.tight_layout()
plt.savefig('visualizations/co2_over_time.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓")

# 4. Light over time
print("  [4/15] Light over time...", end=' ')
plt.figure()
plt.plot(df['date'], df['Light'], linewidth=0.5, alpha=0.7, color='gold')
plt.xlabel('Date')
plt.ylabel('Light (lux)')
plt.title('Light Intensity Over Time')
plt.tight_layout()
plt.savefig('visualizations/light_over_time.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓")

# 5. Occupancy over time
print("  [5/15] Occupancy over time...", end=' ')
plt.figure()
plt.scatter(df['date'], df['Occupancy'], s=1, alpha=0.5, color='purple')
plt.xlabel('Date')
plt.ylabel('Occupancy')
plt.title('Occupancy Status Over Time')
plt.tight_layout()
plt.savefig('visualizations/occupancy_over_time.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓")

# 6-8. By hour of day
print("  [6/15] Temperature by hour...", end=' ')
plt.figure()
df.boxplot(column='Temperature', by='hour', figsize=(14, 6))
plt.suptitle('')
plt.xlabel('Hour of Day')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Distribution by Hour')
plt.tight_layout()
plt.savefig('visualizations/temp_by_hour.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓")

print("  [7/15] Humidity by hour...", end=' ')
plt.figure()
df.boxplot(column='Humidity', by='hour', figsize=(14, 6))
plt.suptitle('')
plt.xlabel('Hour of Day')
plt.ylabel('Humidity (%)')
plt.title('Humidity Distribution by Hour')
plt.tight_layout()
plt.savefig('visualizations/humidity_by_hour.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓")

print("  [8/15] CO2 by hour...", end=' ')
plt.figure()
df.boxplot(column='CO2', by='hour', figsize=(14, 6))
plt.suptitle('')
plt.xlabel('Hour of Day')
plt.ylabel('CO2 (ppm)')
plt.title('CO2 Distribution by Hour')
plt.tight_layout()
plt.savefig('visualizations/co2_by_hour.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓")

# 9. Occupancy heatmap
print("  [9/15] Occupancy heatmap...", end=' ')
occupancy_pivot = df.pivot_table(
    values='Occupancy',
    index='day_of_week',
    columns='hour',
    aggfunc='mean'
)
plt.figure(figsize=(14, 6))
sns.heatmap(occupancy_pivot, cmap='YlOrRd', annot=False, fmt='.2f', cbar_kws={'label': 'Occupancy Rate'})
plt.xlabel('Hour of Day')
plt.ylabel('Day of Week (0=Monday)')
plt.title('Average Occupancy by Day and Hour')
plt.tight_layout()
plt.savefig('visualizations/occupancy_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓")

# 10. Correlation matrix
print("  [10/15] Correlation matrix...", end=' ')
corr_cols = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio', 'Occupancy']
corr_matrix = df[corr_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('visualizations/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓")

# 11-13. Distributions
print("  [11/15] Temperature distribution...", end=' ')
plt.figure()
plt.hist(df['Temperature'], bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.xlabel('Temperature (°C)')
plt.ylabel('Frequency')
plt.title('Temperature Distribution')
plt.tight_layout()
plt.savefig('visualizations/temp_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓")

print("  [12/15] Humidity distribution...", end=' ')
plt.figure()
plt.hist(df['Humidity'], bins=50, alpha=0.7, color='green', edgecolor='black')
plt.xlabel('Humidity (%)')
plt.ylabel('Frequency')
plt.title('Humidity Distribution')
plt.tight_layout()
plt.savefig('visualizations/humidity_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓")

print("  [13/15] CO2 distribution...", end=' ')
plt.figure()
plt.hist(df['CO2'], bins=50, alpha=0.7, color='orange', edgecolor='black')
plt.xlabel('CO2 (ppm)')
plt.ylabel('Frequency')
plt.title('CO2 Distribution')
plt.tight_layout()
plt.savefig('visualizations/co2_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓")

# 14. Scatter matrix (pairplot sample)
print("  [14/15] Scatter matrix...", end=' ')
sample_df = df[['Temperature', 'Humidity', 'CO2', 'Occupancy']].sample(n=min(1000, len(df)))
sns.pairplot(sample_df, hue='Occupancy', plot_kws={'alpha': 0.5}, diag_kind='hist')
plt.savefig('visualizations/scatter_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓")

# 15. Occupancy vs environment
print("  [15/15] Occupancy vs environment...", end=' ')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
df.groupby('Occupancy')['Temperature'].mean().plot(kind='bar', ax=axes[0,0], color=['gray', 'blue'])
axes[0,0].set_title('Average Temperature by Occupancy')
axes[0,0].set_ylabel('Temperature (°C)')

df.groupby('Occupancy')['Humidity'].mean().plot(kind='bar', ax=axes[0,1], color=['gray', 'green'])
axes[0,1].set_title('Average Humidity by Occupancy')
axes[0,1].set_ylabel('Humidity (%)')

df.groupby('Occupancy')['CO2'].mean().plot(kind='bar', ax=axes[1,0], color=['gray', 'orange'])
axes[1,0].set_title('Average CO2 by Occupancy')
axes[1,0].set_ylabel('CO2 (ppm)')

df.groupby('Occupancy')['Light'].mean().plot(kind='bar', ax=axes[1,1], color=['gray', 'gold'])
axes[1,1].set_title('Average Light by Occupancy')
axes[1,1].set_ylabel('Light (lux)')

plt.tight_layout()
plt.savefig('visualizations/occupancy_vs_environment.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓")

# Step 6: Save cleaned dataset
print("\n[Step 6/8] Saving cleaned dataset...")
os.makedirs('data/processed', exist_ok=True)
df.to_csv('data/processed/occupancy_combined.csv', index=False)
print(f"  ✓ Saved: data/processed/occupancy_combined.csv ({len(df):,} rows × {len(df.columns)} columns)")

# Step 7: Summary
print("\n[Step 7/8] Summary...")
print(f"  Dataset: {len(df):,} observations")
print(f"  Variables: {len(df.columns)} (7 original + 6 engineered)")
print(f"  Time period: {df['date'].min()} to {df['date'].max()}")
print(f"  Cleaned data: data/processed/occupancy_combined.csv")
print(f"  Visualizations: 15 PNG files in visualizations/")

print("\n" + "="*70)
print("✅ Data exploration complete!")
print("="*70)
