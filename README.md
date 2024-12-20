# Metro Passenger Flow Prediction

This project generates synthetic metro passenger data and trains a machine learning model to predict passenger loads.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Generate synthetic data:
```bash
python data_generator.py
```
This will create a file named `metro_passenger_data.csv` with synthetic passenger flow data.

2. Train the model:
```bash
python train_model.py
```
This will:
- Train a Random Forest model on the generated data
- Save the trained model as `metro_passenger_model.joblib`
- Display model performance metrics and feature importance
- Make a sample prediction

## Data Description

The generated data includes:
- 4 metro lines (M1-M4)
- 57 stations (S1-S57)
- Train capacity of 650 passengers
- Realistic passenger flow patterns based on:
  - Time of day (peak/off-peak hours)
  - Station popularity
  - Day of week
  - Weekend/weekday patterns

## Model Features

The prediction model uses the following features:
- Hour of day
- Day of week
- Weekend indicator
- Station number
- Metro line

The model predicts the current passenger load at any given station based on these features.
