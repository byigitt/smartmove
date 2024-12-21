#!/usr/bin/env python3
"""
Train model and make predictions for Ankara Metro passenger occupancy
"""

import os
import sys
import argparse
import joblib
import pandas as pd

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from train.predictor import MetroPassengerPredictor
from train.utils import get_data_path, format_prediction_results
from train.evaluation import interpret_occupancy
from train.station_config import STATION_ORDER, WEATHER_FACTORS

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train and predict Ankara Metro passenger occupancy',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train a new model'
    )
    
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['rf', 'gb'],
        default='rf',
        help='Model type (rf: Random Forest, gb: Gradient Boosting)'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default='metro_predictor.joblib',
        help='Path to save/load the model'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        help='Path to training data (default: auto-detect)'
    )
    
    # Prediction parameters
    parser.add_argument(
        '--metro-line',
        type=str,
        choices=STATION_ORDER.keys(),
        help='Metro line for prediction'
    )
    
    parser.add_argument(
        '--station',
        type=str,
        help='Station name for prediction'
    )
    
    parser.add_argument(
        '--hour',
        type=int,
        choices=range(24),
        help='Hour of day (0-23) for prediction'
    )
    
    parser.add_argument(
        '--weather',
        type=str,
        choices=WEATHER_FACTORS.keys(),
        help='Weather condition for prediction'
    )
    
    parser.add_argument(
        '--weekend',
        action='store_true',
        help='Whether the prediction is for a weekend'
    )
    
    return parser.parse_args()

def validate_station(metro_line, station):
    """Validate that the station exists on the given metro line"""
    if station not in STATION_ORDER[metro_line]:
        print(f"Error: Station '{station}' not found on {metro_line} line")
        print("\nAvailable stations:")
        for s in STATION_ORDER[metro_line]:
            print(f"  - {s}")
        sys.exit(1)

def train_model(args):
    """Train a new model"""
    print(f"Training new {args.model_type.upper()} model...")
    
    # Load data
    data_path = args.data_path or get_data_path()
    if not os.path.exists(data_path):
        print(f"Error: Training data not found at {data_path}")
        print("Please run generate.py first to create the dataset.")
        sys.exit(1)
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, low_memory=False)
    
    # Train model
    predictor = MetroPassengerPredictor()
    predictor.train(df, model_type=args.model_type)
    
    # Save model
    print(f"\nSaving model to {args.model_path}...")
    joblib.dump(predictor, args.model_path)
    print("Done!")
    
    return predictor

def load_model(model_path):
    """Load a trained model"""
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model not found at {model_path}")
        print("Please train a model first using --train")
        sys.exit(1)

def make_prediction(predictor, args):
    """Make a prediction using the model"""
    # Validate station
    validate_station(args.metro_line, args.station)
    
    # Make prediction
    prediction = predictor.predict_specific_conditions(
        metro_line=args.metro_line,
        station=args.station,
        hour=args.hour,
        weather=args.weather,
        is_weekend=args.weekend
    )
    
    # Print results
    print(format_prediction_results(
        prediction=prediction,
        station=args.station,
        metro_line=args.metro_line,
        hour=args.hour,
        weather=args.weather,
        is_weekend=args.weekend
    ))
    print(f"Status: {interpret_occupancy(prediction)}")
    
    return prediction

def main():
    """Main function"""
    args = parse_args()
    
    if args.train:
        predictor = train_model(args)
    else:
        # Validate prediction parameters
        if not all([args.metro_line, args.station, args.hour is not None, args.weather]):
            print("Error: For prediction, please provide all of:")
            print("  --metro-line")
            print("  --station")
            print("  --hour")
            print("  --weather")
            print("\nOptional:")
            print("  --weekend")
            sys.exit(1)
        
        # Load model and make prediction
        predictor = load_model(args.model_path)
        make_prediction(predictor, args)

if __name__ == '__main__':
    main() 