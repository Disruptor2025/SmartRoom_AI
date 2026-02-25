# SmartRoom AI

Final project for Data Analytics and IoT class. Using LSTM neural networks to predict temperature and humidity in buildings.

## What it does
Predicts temperature and humidity 1 hour ahead so HVAC systems can heat/cool proactively instead of waiting for temperature to change.

## Results
- Temperature model: R² = 0.82, MAE = 0.21°C
- Humidity model: R² = 0.87, MAE = 0.72%  
- Also made a Random Forest classifier for occupancy (99.1% accuracy)

## Dataset
UCI Occupancy Detection dataset - 20,560 sensor readings from an office in Belgium (Feb 2015)

## Technologies
Python, TensorFlow, pandas, scikit-learn, Tableau

## Files
- 01_data_exploration.py - loads and cleans data, makes visualizations
- 02_lstm_temperature.py - temperature prediction model
- 03_lstm_humidity.py - humidity prediction model
- 04_occupancy_classifier.py - predicts if room is occupied

## How it works
Uses 60 minutes of sensor data (temp, humidity, CO2, light) to predict conditions 1 hour ahead. The LSTM learns patterns like daily heating cycles and occupancy effects.

Idrees Khan
University of San Diego, Feb 2026
