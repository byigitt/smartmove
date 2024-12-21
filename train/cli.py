"""
Command-line interface for the Ankara Metro prediction system
"""

import argparse
import sys
import joblib
import pandas as pd
import os
from .predictor import MetroPassengerPredictor
from .utils import get_available_stations, format_prediction_results, get_data_path
from .evaluation import interpret_occupancy

def get_prediction_from_model(predictor_path='metro_predictor.joblib'):
    """
    Load the trained model and make a prediction based on user input
    
    Args:
        predictor_path (str): Path to the saved model file
    """
    try:
        predictor = joblib.load(predictor_path)
    except FileNotFoundError:
        print("Error: Model file not found. Please train the model first.")
        sys.exit(1)
    
    print("\nWelcome to Ankara Metro Occupancy Predictor!")
    
    # Get available metro lines
    metro_lines = predictor.categories['metro_lines'].tolist()
    print("\nAvailable metro lines:", ', '.join(metro_lines))
    
    while True:
        try:
            # Get metro line input
            print("\nAvailable metro lines:")
            for i, line in enumerate(metro_lines, 1):
                print(f"{i}. {line}")
            line_idx = int(input("\nSelect metro line (enter number): ")) - 1
            if not 0 <= line_idx < len(metro_lines):
                print("Invalid metro line selection")
                continue
            metro_line = metro_lines[line_idx]
            
            # Get available stations
            stations = get_available_stations(metro_line)
            
            print("\nAvailable stations:")
            for i, station in enumerate(stations, 1):
                print(f"{i}. {station}")
            station_idx = int(input("\nSelect station (enter number): ")) - 1
            if not 0 <= station_idx < len(stations):
                print("Invalid station selection")
                continue
            station = stations[station_idx]
            
            # Get time input
            hour = int(input("\nEnter hour (0-23): "))
            if not 0 <= hour <= 23:
                print("Hour must be between 0 and 23")
                continue
                
            # Get weather input
            weather_conditions = predictor.categories['weather'].tolist()
            print("\nAvailable weather conditions:")
            for i, weather in enumerate(weather_conditions, 1):
                print(f"{i}. {weather}")
            weather_idx = int(input("\nSelect weather condition (enter number): ")) - 1
            if not 0 <= weather_idx < len(weather_conditions):
                print("Invalid weather selection")
                continue
            weather = weather_conditions[weather_idx]
            
            # Get day type input
            is_weekend = input("\nIs it a weekend? (y/n): ").lower().startswith('y')
            
            break
        except ValueError:
            print("Invalid input. Please try again.")
    
    # Make prediction
    prediction = predictor.predict_specific_conditions(
        metro_line=metro_line,
        station=station,
        hour=hour,
        weather=weather,
        is_weekend=is_weekend
    )
    
    # Print results
    print(format_prediction_results(
        prediction, station, metro_line, hour, weather, is_weekend
    ))
    print(f"Status: {interpret_occupancy(prediction)}")
    
    return prediction

def main():
    """Train model or make predictions based on command-line arguments"""
    parser = argparse.ArgumentParser(description='Ankara Metro Occupancy Predictor')
    parser.add_argument('--predict', action='store_true', 
                      help='Enter prediction mode (requires trained model)')
    parser.add_argument('--train', action='store_true',
                      help='Train a new model')
    parser.add_argument('--generate-data', action='store_true',
                      help='Generate synthetic training data')
    
    args = parser.parse_args()
    
    if args.generate_data:
        # Import data generator only when needed
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "generate_data",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "generate_data.py")
        )
        generate_data_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generate_data_module)
        generate_data_module.main()
        return
    
    if args.predict:
        get_prediction_from_model()
    elif args.train:
        data_path = get_data_path()
        if not os.path.exists(data_path):
            print(f"Error: Training data not found at {data_path}")
            print("Please run with --generate-data first to create the dataset.")
            sys.exit(1)
            
        # Load the data
        print("Loading data...")
        df = pd.read_csv(data_path, low_memory=False)
        
        # Initialize and train the model
        predictor = MetroPassengerPredictor()
        predictor.train(df, model_type='rf')
        
        # Save the model
        joblib.dump(predictor, 'metro_predictor.joblib')
        print("\nModel saved as metro_predictor.joblib")
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 