# SmartRoom AI - Predictive HVAC Control

Final project for Data Analytics and Internet of Things course at University of San Diego.

## Project Overview
LSTM neural networks for predicting indoor temperature and humidity 1 hour ahead to enable proactive HVAC control in commercial buildings.

## Results
- Temperature LSTM: R²=0.82, MAE=0.21°C
- Humidity LSTM: R²=0.87, MAE=0.72%
- Random Forest Occupancy Classifier: 99.1% accuracy

## Dataset
UCI Occupancy Detection Dataset (20,560 observations, February 2015)

## Files
- `01_data_exploration.py` - Data loading, cleaning, and visualization
- `02_lstm_temperature.py` - Temperature prediction model
- `03_lstm_humidity.py` - Humidity prediction model  
- `04_occupancy_classifier.py` - Occupancy classification (bonus)

## Technologies
Python, TensorFlow/Keras, LSTM, pandas, scikit-learn, Tableau

## Author
Idrees Khan - University of San Diego
February 2026
