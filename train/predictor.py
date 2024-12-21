"""
Main predictor class for the Ankara Metro prediction system
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
import pandas as pd
import os

from .data_preprocessing import prepare_features, create_feature_pipeline, get_time_factor, is_peak_hour
from .evaluation import evaluate_model
from .station_config import TERMINAL_STATIONS, STATION_ORDER, WEATHER_FACTORS
from .utils import (
    determine_time_period,
    get_service_frequency,
    calculate_trains_per_hour,
    get_station_position,
    get_data_path
)

class MetroPassengerPredictor:
    """Main class for predicting metro passenger occupancy rates"""
    
    def __init__(self):
        """Initialize the predictor"""
        self.model = None
        self.preprocessor = None
        self.categories = None
        
    def train(self, df, model_type='rf'):
        """
        Train the model with cross-validation
        
        Args:
            df (pd.DataFrame): Training data
            model_type (str): Model type ('rf' for Random Forest or 'gb' for Gradient Boosting)
        
        Returns:
            Pipeline: Trained model pipeline
        """
        print("Preparing features...")
        df, self.categories = prepare_features(df)
        
        # Define features and target
        feature_columns = [
            'Hour', 'Hour_Sin', 'Hour_Cos', 
            'DayOfWeek_Sin', 'DayOfWeek_Cos',
            'Metro_Line', 'Station_ID', 'Station_Type',
            'Weather_Condition', 'Time_Period',
            'Is_Weekend', 'Weather_Disruption',
            'Service_Frequency', 'Trains_Per_Hour',
            'Boarding_Passengers', 'Alighting_Passengers',
            'Transfer_Out', 'Capacity_Utilization'
        ]
        
        target_column = 'Occupancy_Rate'
        
        # Split features and target
        X = df[feature_columns]
        y = df[target_column]
        
        # Build and train model
        print(f"Training {model_type.upper()} model...")
        self.model = self._build_model(model_type)
        
        # Final train-test split for detailed evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False  # Keep temporal order
        )
        
        # Train final model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        evaluate_model(self.model, X_test, y_test)
        
        return self.model
    
    def _build_model(self, model_type='rf'):
        """
        Build the model pipeline
        
        Args:
            model_type (str): Model type ('rf' for Random Forest or 'gb' for Gradient Boosting)
            
        Returns:
            Pipeline: Model pipeline
        """
        preprocessor = create_feature_pipeline()
        
        if model_type == 'rf':
            regressor = RandomForestRegressor(
                n_estimators=300,
                max_depth=20,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features='sqrt',
                n_jobs=-1,
                random_state=42
            )
        else:  # gradient boosting
            regressor = GradientBoostingRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
        
        return Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', regressor)
        ])
    
    def predict_specific_conditions(self, metro_line, station, hour, weather, is_weekend=False):
        """
        Make prediction for specific conditions using real data patterns
        
        Args:
            metro_line (str): Metro line identifier
            station (str): Station name
            hour (int): Hour of the day (0-23)
            weather (str): Weather condition
            is_weekend (bool): Whether it's a weekend
            
        Returns:
            float: Predicted occupancy rate
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
            
        # Get station position information
        station_idx, total_stations, distance_from_kizilay = get_station_position(metro_line, station)
        is_terminal = station in TERMINAL_STATIONS[metro_line]
        
        # Calculate distance factor for M1-2-3 line
        if metro_line == 'M1-2-3' and distance_from_kizilay is not None:
            max_distance = max(
                STATION_ORDER[metro_line].index('15 Temmuz Kızılay Millî İrade'),
                total_stations - STATION_ORDER[metro_line].index('15 Temmuz Kızılay Millî İrade')
            )
            distance_factor = max(0.15, np.exp(-1.5 * (distance_from_kizilay / (max_distance/3))**2))
        else:
            distance_factor = 1.0
        
        # Get weather impact
        weather_info = WEATHER_FACTORS[weather]
        weather_disruption = np.random.random() < weather_info['disruption_prob']
        weather_factor = weather_info['factor']
        
        # Load similar conditions from training data
        df = pd.read_csv(get_data_path(), low_memory=False)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Hour'] = df['Timestamp'].dt.hour
        
        # Filter similar conditions
        similar_conditions = df[
            (df['Station_ID'] == station) &
            (df['Metro_Line'] == metro_line) &
            (df['Hour'] == hour) &
            (df['Weather_Condition'] == weather) &
            (df['Is_Weekend'] == is_weekend)
        ]
        
        if len(similar_conditions) == 0:
            print("Warning: No exact matches found, using closest available data")
            similar_conditions = df[
                (df['Station_ID'] == station) &
                (df['Metro_Line'] == metro_line) &
                (df['Hour'] == hour) &
                (df['Is_Weekend'] == is_weekend)
            ]
        
        if len(similar_conditions) == 0:
            raise ValueError("No similar conditions found in training data")
        
        # Determine time period and service frequency
        time_period = determine_time_period(hour)
        frequency = get_service_frequency(metro_line, time_period)
        trains_per_hour = calculate_trains_per_hour(frequency)
        
        # Calculate base values from similar conditions
        base_boarding = similar_conditions['Boarding_Passengers'].median() * weather_factor
        base_alighting = similar_conditions['Alighting_Passengers'].median() * weather_factor
        base_transfer = similar_conditions['Transfer_Out'].median() * weather_factor
        base_capacity = similar_conditions['Capacity_Utilization'].median() * weather_factor
        
        # Apply terminal station logic
        if is_terminal:
            if hour >= 16 and hour <= 19:  # Evening peak
                base_boarding *= 0.1  # Almost no one boards at terminal in evening
                base_alighting *= 1.2  # More people getting off
                base_capacity *= 0.15  # Much lower capacity utilization
            elif hour >= 7 and hour <= 9:  # Morning peak
                base_boarding *= 1.2  # More people boarding at terminal in morning
                base_alighting *= 0.1  # Very few people getting off
                base_capacity *= 0.4  # Lower capacity utilization
            else:
                base_boarding *= 0.2
                base_alighting *= 0.2
                base_capacity *= 0.15
        
        # Apply normal distribution adjustment for M1-2-3 line
        if metro_line == 'M1-2-3' and not is_terminal:
            base_capacity *= distance_factor
            base_boarding *= distance_factor
            base_alighting *= distance_factor
            base_transfer *= distance_factor

        # Get time-based factor
        time_factor = get_time_factor(hour, is_weekend)
        
        # Apply time factor to base values
        base_boarding *= time_factor
        base_alighting *= time_factor
        base_transfer *= time_factor
        base_capacity *= time_factor
        
        # Create sample data with adjusted values
        sample_data = pd.DataFrame({
            'Timestamp': [pd.Timestamp.now().replace(hour=hour)],
            'Metro_Line': [metro_line],
            'Station_ID': [station],
            'Station_Type': [similar_conditions['Station_Type'].mode().iloc[0]],
            'Weather_Condition': [weather],
            'Time_Period': [time_period],
            'Is_Weekend': [is_weekend],
            'Service_Frequency': [frequency],
            'Trains_Per_Hour': [trains_per_hour],
            'Weather_Disruption': [weather_disruption],
            'Boarding_Passengers': [base_boarding],
            'Alighting_Passengers': [base_alighting],
            'Transfer_Out': [base_transfer],
            'Capacity_Utilization': [base_capacity]
        })
        
        # Adjust values for special stations during peak hours
        if station == '15 Temmuz Kızılay Millî İrade':
            if is_peak_hour(hour, is_weekend):
                multiplier = 1.5 * weather_factor
                sample_data['Boarding_Passengers'] *= multiplier
                sample_data['Alighting_Passengers'] *= multiplier
                sample_data['Transfer_Out'] *= multiplier
                sample_data['Capacity_Utilization'] *= multiplier
        elif station in ['Bahçelievler', 'Millî Kütüphane']:
            if is_peak_hour(hour, is_weekend):
                multiplier = 1.2 * weather_factor
                sample_data['Boarding_Passengers'] *= multiplier
                sample_data['Alighting_Passengers'] *= multiplier
                sample_data['Transfer_Out'] *= multiplier
                sample_data['Capacity_Utilization'] *= multiplier
        
        # Enhanced debug information
        print("\nPrediction Context:")
        print(f"Station: {station} (Position: {station_idx + 1}/{total_stations})")
        if metro_line == 'M1-2-3':
            print(f"Distance from Kızılay: {distance_from_kizilay} stations")
            print(f"Distance Factor: {distance_factor:.2f}")
        print(f"Terminal Station: {'Yes' if is_terminal else 'No'}")
        print(f"Time: {hour:02d}:00")
        print(f"Time Factor: {time_factor:.2f}")
        print(f"Weather: {weather} (Impact factor: {weather_factor:.2f})")
        print(f"Weather Disruption: {'Yes' if weather_disruption else 'No'}")
        print(f"Weekend: {'Yes' if is_weekend else 'No'}")
        print(f"Time Period: {time_period}")
        
        # Prepare features
        sample_data, _ = prepare_features(sample_data, store_categories=False)
        
        # Make prediction
        prediction = self.model.predict(sample_data)[0]
        
        # Apply final adjustments for terminal stations
        if is_terminal:
            if hour >= 16 and hour <= 19:  # Evening peak
                prediction = max(5.0, min(prediction * 0.15, 15.0))
            else:
                prediction = max(3.0, min(prediction * 0.3, 30.0))
        elif metro_line == 'M1-2-3':
            if station == '15 Temmuz Kızılay Millî İrade':
                prediction = min(prediction * 1.2, 95.0)
            else:
                base_pred = prediction * (distance_factor ** 1.2)
                
                if distance_from_kizilay <= 3:
                    min_occupancy = max(15.0 * time_factor, 5.0)
                elif distance_from_kizilay <= 7:
                    min_occupancy = max(10.0 * time_factor, 3.0)
                else:
                    min_occupancy = max(5.0 * time_factor, 2.0)
                
                prediction = max(min_occupancy, base_pred)
        
        # Final sanity check for very distant stations
        if metro_line == 'M1-2-3' and distance_from_kizilay > 15:
            prediction = max(5.0 * time_factor, min(prediction, 20.0))
        
        # Ensure we never predict exactly 0%
        prediction = max(2.0, prediction)
            
        return prediction 