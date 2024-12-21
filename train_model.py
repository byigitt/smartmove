"""
Train and evaluate machine learning model for Ankara Metro passenger predictions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import joblib
from datetime import datetime, time
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys

class MetroPassengerPredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.weather_categories = None
        self.station_types = None
        self.metro_lines = None
        self.time_periods = None
        
        # Define terminal stations for each line
        self.terminal_stations = {
            'M1-2-3': ['Koru', 'OSB-Törekent'],  # Terminal stations for M1-2-3 line
            'M4': ['15 Temmuz Kızılay Millî İrade', 'Şehitler'],  # Terminal stations for M4
            'A1': ['AŞTİ', 'Dikimevi']  # Terminal stations for A1
        }
        
        # Define station order according to the metro map
        self.station_order = {
            'M1-2-3': [
                'Koru', 'Çayyolu', 'Ümitköy', 'Beytepe', 'Tarım Bakanlığı-Danıştay',
                'Bilkent', 'Orta Doğu Teknik Üniversitesi', 'Maden Tetkik ve Arama',
                'Söğütözü', 'Millî Kütüphane', 'Necatibey', '15 Temmuz Kızılay Millî İrade',
                'Sıhhiye', 'Ulus', 'Atatürk Kültür Merkezi', 'Akköprü', 'İvedik',
                'Yenimahalle', 'Demetevler', 'Hastane', 'Macunköy',
                'Orta Doğu Sanayi ve Ticaret Merkezi', 'Batıkent', 'Batı Merkez',
                'Mesa', 'Botanik', 'İstanbul Yolu', 'Eryaman 1-2', 'Eryaman 5',
                'Devlet Mahallesi/1910 Ankaragücü', 'Harikalar Diyarı', 'Fatih',
                'Gaziosmanpaşa', 'OSB-Törekent'
            ],
            'M4': [
                '15 Temmuz Kızılay Millî İrade', 'Adliye', 'Gar', 'Atatürk Kültür Merkezi',
                'Ankara Su ve Kanalizasyon İdaresi', 'Dışkapı', 'Meteoroloji', 'Belediye',
                'Mecidiye', 'Kuyubaşı', 'Dutluk', 'Şehitler'
            ],
            'A1': [
                'AŞTİ', 'Emek', 'Bahçelievler', 'Beşevler', 'Anadolu/Anıtkabir',
                'Maltepe', 'Demirtepe', '15 Temmuz Kızılay Millî İrade', 'Kolej',
                'Kurtuluş', 'Dikimevi'
            ]
        }
        
    def prepare_features(self, df):
        """Prepare features for the model with enhanced engineering"""
        # Convert timestamp to datetime if it's not already
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # Extract time-based features
        df['Hour'] = df['Timestamp'].dt.hour
        df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
        df['Month'] = df['Timestamp'].dt.month
        
        # Create cyclical time features
        df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour']/24)
        df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour']/24)
        df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek']/7)
        df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek']/7)
        
        # Store unique categories if not already stored
        if self.weather_categories is None:
            self.weather_categories = df['Weather_Condition'].unique()
            self.station_types = df['Station_Type'].unique()
            self.metro_lines = df['Metro_Line'].unique()
            self.time_periods = df['Time_Period'].unique()
        
        return df
    
    def create_feature_pipeline(self):
        """Create an enhanced feature preprocessing pipeline"""
        # Define feature groups
        numeric_features = [
            'Hour', 'Hour_Sin', 'Hour_Cos', 
            'DayOfWeek_Sin', 'DayOfWeek_Cos',
            'Service_Frequency', 'Trains_Per_Hour',
            'Boarding_Passengers', 'Alighting_Passengers',
            'Transfer_Out', 'Capacity_Utilization'
        ]
        
        categorical_features = [
            'Metro_Line', 'Station_ID', 'Station_Type', 
            'Weather_Condition', 'Time_Period'
        ]
        
        boolean_features = ['Is_Weekend', 'Weather_Disruption']
        
        # Create preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(
                    drop='first', 
                    sparse_output=False,
                    handle_unknown='ignore'
                ), categorical_features),
                ('bool', 'passthrough', boolean_features)
            ],
            remainder='drop'
        )
        
        return self.preprocessor
    
    def build_model(self, model_type='rf'):
        """Build the model pipeline with specified type"""
        preprocessor = self.create_feature_pipeline()
        
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
    
    def train(self, df, model_type='rf'):
        """Train the model with cross-validation"""
        print("Preparing features...")
        df = self.prepare_features(df)
        
        # Initialize station_order and terminal_stations if not already set
        if not hasattr(self, 'station_order'):
            self.station_order = {
                'M1-2-3': [
                    'Koru', 'Çayyolu', 'Ümitköy', 'Beytepe', 'Tarım Bakanlığı-Danıştay',
                    'Bilkent', 'Orta Doğu Teknik Üniversitesi', 'Maden Tetkik ve Arama',
                    'Söğütözü', 'Millî Kütüphane', 'Necatibey', '15 Temmuz Kızılay Millî İrade',
                    'Sıhhiye', 'Ulus', 'Atatürk Kültür Merkezi', 'Akköprü', 'İvedik',
                    'Yenimahalle', 'Demetevler', 'Hastane', 'Macunköy',
                    'Orta Doğu Sanayi ve Ticaret Merkezi', 'Batıkent', 'Batı Merkez',
                    'Mesa', 'Botanik', 'İstanbul Yolu', 'Eryaman 1-2', 'Eryaman 5',
                    'Devlet Mahallesi/1910 Ankaragücü', 'Harikalar Diyarı', 'Fatih',
                    'Gaziosmanpaşa', 'OSB-Törekent'
                ],
                'M4': [
                    '15 Temmuz Kızılay Millî İrade', 'Adliye', 'Gar', 'Atatürk Kültür Merkezi',
                    'Ankara Su ve Kanalizasyon İdaresi', 'Dışkapı', 'Meteoroloji', 'Belediye',
                    'Mecidiye', 'Kuyubaşı', 'Dutluk', 'Şehitler'
                ],
                'A1': [
                    'AŞTİ', 'Emek', 'Bahçelievler', 'Beşevler', 'Anadolu/Anıtkabir',
                    'Maltepe', 'Demirtepe', '15 Temmuz Kızılay Millî İrade', 'Kolej',
                    'Kurtuluş', 'Dikimevi'
                ]
            }
            
        if not hasattr(self, 'terminal_stations'):
            self.terminal_stations = {
                'M1-2-3': ['Koru', 'OSB-Törekent'],  # Terminal stations for M1-2-3 line
                'M4': ['15 Temmuz Kızılay Millî İrade', 'Şehitler'],  # Terminal stations for M4
                'A1': ['AŞTİ', 'Dikimevi']  # Terminal stations for A1
            }
        
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
        self.model = self.build_model(model_type)
        
        # Final train-test split for detailed evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False  # Keep temporal order
        )
        
        # Train final model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        self._evaluate_model(X_test, y_test)
        
        return self.model
    
    def predict_specific_conditions(self, metro_line, station, hour, weather, is_weekend=False):
        """Make prediction for specific conditions using real data patterns"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
            
        # Get station position information
        station_idx = self.station_order[metro_line].index(station)
        total_stations = len(self.station_order[metro_line])
        is_terminal = station in self.terminal_stations[metro_line]
        
        # Calculate distance from Kızılay for M1-2-3 line
        if metro_line == 'M1-2-3':
            kizilayIdx = self.station_order[metro_line].index('15 Temmuz Kızılay Millî İrade')
            distance_from_kizilay = abs(station_idx - kizilayIdx)
            # Create a steeper normal distribution factor (1.0 at Kızılay, rapidly decreasing with distance)
            max_distance = max(kizilayIdx, total_stations - kizilayIdx)
            # Using a much steeper decay but with a minimum value
            distance_factor = max(0.15, np.exp(-1.5 * (distance_from_kizilay / (max_distance/3))**2))
        else:
            distance_factor = 1.0
        
        # Weather impact factors - matched with data_generator
        weather_factors = {
            'Sunny': {'factor': 0.9, 'disruption_prob': 0.0},
            'Cloudy': {'factor': 1.0, 'disruption_prob': 0.0},
            'Rainy': {'factor': 1.1, 'disruption_prob': 0.02},
            'Snowy': {'factor': 1.2, 'disruption_prob': 0.10},
            'Stormy': {'factor': 1.25, 'disruption_prob': 0.15}
        }
        
        # Get weather impact
        weather_info = weather_factors[weather]
        weather_disruption = np.random.random() < weather_info['disruption_prob']
        weather_factor = weather_info['factor']
        
        # Load the training data to get realistic values
        df = pd.read_csv('ankara_metro_crowding_data_realistic.csv', low_memory=False)
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
        
        # Determine time period based on hour
        if (7 <= hour <= 9) or (16 <= hour <= 19):
            time_period = 'peak'
        elif hour >= 23 or hour <= 5:
            time_period = 'off_peak'
        else:
            time_period = 'regular'
        
        # Get service frequency based on time period and metro line
        frequencies = {
            'A1': {'peak': 3, 'regular': 5, 'off_peak': 10},
            'M1-2-3': {'peak': 3, 'regular': 5, 'off_peak': 10},
            'M4': {'peak': 4, 'regular': 7, 'off_peak': 15}
        }
        
        frequency = frequencies[metro_line][time_period]
        trains_per_hour = max(1, 60 // frequency)
        
        # Calculate base values from similar conditions
        base_boarding = similar_conditions['Boarding_Passengers'].median() * weather_factor
        base_alighting = similar_conditions['Alighting_Passengers'].median() * weather_factor
        base_transfer = similar_conditions['Transfer_Out'].median() * weather_factor
        base_capacity = similar_conditions['Capacity_Utilization'].median() * weather_factor
        
        # Apply terminal station logic
        if is_terminal:
            # For terminal stations, significantly reduce boarding and increase alighting
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
        time_factor = self._get_time_factor(hour, is_weekend)
        
        # Apply time factor to base values
        base_boarding *= time_factor
        base_alighting *= time_factor
        base_transfer *= time_factor
        base_capacity *= time_factor
        
        # Create sample data with adjusted values
        sample_data = pd.DataFrame({
            'Timestamp': [datetime.now().replace(hour=hour)],
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
        
        # Adjust values for special stations during peak hours (with weather consideration)
        if station == '15 Temmuz Kızılay Millî İrade':
            if self._is_peak_hour(hour, is_weekend):
                multiplier = 1.5 * weather_factor  # Increased multiplier for Kızılay
                sample_data['Boarding_Passengers'] *= multiplier
                sample_data['Alighting_Passengers'] *= multiplier
                sample_data['Transfer_Out'] *= multiplier
                sample_data['Capacity_Utilization'] *= multiplier
        elif station in ['Bahçelievler', 'Millî Kütüphane']:
            if self._is_peak_hour(hour, is_weekend):
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
        sample_data = self.prepare_features(sample_data)
        
        # Make prediction
        prediction = self.model.predict(sample_data)[0]
        
        # Apply final adjustments for terminal stations
        if is_terminal:
            if hour >= 16 and hour <= 19:  # Evening peak
                prediction = max(5.0, min(prediction * 0.15, 15.0))  # Minimum 5% for terminals in evening
            else:
                prediction = max(3.0, min(prediction * 0.3, 30.0))  # Minimum 3% for terminals at other times
        elif metro_line == 'M1-2-3':
            # Apply normal distribution to the final prediction as well
            if station == '15 Temmuz Kızılay Millî İrade':
                prediction = min(prediction * 1.2, 95.0)  # Cap Kızılay at 95%
            else:
                # Calculate base prediction with distance factor
                base_pred = prediction * (distance_factor ** 1.2)
                
                # Apply minimum based on distance from Kızılay and time
                if distance_from_kizilay <= 3:  # Close to Kızılay
                    min_occupancy = max(15.0 * time_factor, 5.0)
                elif distance_from_kizilay <= 7:  # Moderately distant
                    min_occupancy = max(10.0 * time_factor, 3.0)
                else:  # Far from Kızılay
                    min_occupancy = max(5.0 * time_factor, 2.0)
                
                prediction = max(min_occupancy, base_pred)
        
        # Final sanity check for very distant stations
        if metro_line == 'M1-2-3' and distance_from_kizilay > 15:
            prediction = max(5.0 * time_factor, min(prediction, 20.0))  # Cap distant stations but maintain time-based minimum
        
        # Ensure we never predict exactly 0%
        prediction = max(2.0, prediction)  # Global minimum of 2% occupancy
            
        return prediction
    
    def _is_peak_hour(self, hour, is_weekend=False):
        """Helper method to determine peak hours with gradual transition"""
        if is_weekend:
            return False  # No peak hours on weekends
        
        # Core peak hours
        morning_peak = 7 <= hour <= 9
        evening_peak = 16 <= hour <= 19
        
        # Transition hours (shoulder periods)
        morning_shoulder = 6 <= hour <= 10  # One hour before and after morning peak
        evening_shoulder = 15 <= hour <= 21  # One hour before and two hours after evening peak
        
        # Return peak status and transition factor
        if morning_peak or evening_peak:
            return True
        return morning_shoulder or evening_shoulder
    
    def _get_time_factor(self, hour, is_weekend=False):
        """Calculate time-based factor for occupancy"""
        if is_weekend:
            # Weekend factors
            if 10 <= hour <= 22:  # Active hours on weekend
                return 0.6
            elif 7 <= hour <= 9 or 22 <= hour <= 23:  # Early morning and late evening
                return 0.3
            else:
                return 0.2
        else:
            # Weekday factors
            if 7 <= hour <= 9 or 16 <= hour <= 19:  # Peak hours
                return 1.0
            elif hour == 6 or hour == 10:  # Morning shoulder
                return 0.7
            elif hour == 15 or hour == 20:  # Evening shoulder
                return 0.8
            elif hour == 21:  # Late evening transition
                return 0.5
            elif hour == 22:  # Late evening
                return 0.3
            elif hour == 23:  # Very late evening
                return 0.2
            else:
                return 0.15
    
    def _evaluate_model(self, X_test, y_test):
        """Perform detailed model evaluation"""
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Calculate MAPE only for non-zero actual values to avoid division by zero
        mask = y_test != 0
        mape = mean_absolute_percentage_error(y_test[mask], y_pred[mask])
        
        print("\nModel Performance Metrics:")
        print(f"RMSE: {rmse:.2f}")
        print(f"R² Score: {r2:.4f}")
        print(f"MAPE: {mape:.2%}")
        
        # Plot actual vs predicted with density
        plt.figure(figsize=(12, 8))
        plt.hexbin(y_test, y_pred, gridsize=30, cmap='YlOrRd')
        plt.colorbar(label='Count')
        plt.plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual Occupancy Rate')
        plt.ylabel('Predicted Occupancy Rate')
        plt.title('Actual vs Predicted Occupancy Rate\nwith Density Visualization')
        plt.legend()
        plt.tight_layout()
        plt.savefig('prediction_performance.png')
        plt.close()

def get_prediction_from_model(predictor_path='metro_predictor.joblib'):
    """Load the trained model and make a prediction based on user input"""
    try:
        predictor = joblib.load(predictor_path)
    except FileNotFoundError:
        print("Error: Model file not found. Please train the model first.")
        sys.exit(1)
    
    print("\nWelcome to Ankara Metro Occupancy Predictor!")
    
    # Get available metro lines
    metro_lines = predictor.metro_lines.tolist()
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
            
            # Load the data to get available stations for the selected line
            df = pd.read_csv('ankara_metro_crowding_data_realistic.csv', low_memory=False)
            stations = df[df['Metro_Line'] == metro_line]['Station_ID'].unique().tolist()
            
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
            weather_conditions = predictor.weather_categories.tolist()
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
    print("\nPrediction Results:")
    print("-" * 40)
    print(f"Location: {station} Station ({metro_line} Line)")
    print(f"Time: {hour:02d}:00")
    print(f"Weather: {weather}")
    print(f"Day type: {'Weekend' if is_weekend else 'Weekday'}")
    print(f"Predicted occupancy rate: {prediction:.1f}%")
    
    # Provide crowding interpretation
    if prediction < 30:
        status = "Low occupancy - Very comfortable"
    elif prediction < 50:
        status = "Moderate occupancy - Comfortable"
    elif prediction < 70:
        status = "High occupancy - Somewhat crowded"
    elif prediction < 85:
        status = "Very high occupancy - Crowded"
    else:
        status = "Extremely high occupancy - Very crowded"
    
    print(f"Status: {status}")
    
    return prediction

def main():
    """Train model or make predictions based on command-line arguments"""
    parser = argparse.ArgumentParser(description='Ankara Metro Occupancy Predictor')
    parser.add_argument('--predict', action='store_true', 
                      help='Enter prediction mode (requires trained model)')
    parser.add_argument('--train', action='store_true',
                      help='Train a new model')
    
    args = parser.parse_args()
    
    if args.predict:
        get_prediction_from_model()
    elif args.train:
        # Load the data
        print("Loading data...")
        df = pd.read_csv('ankara_metro_crowding_data_realistic.csv', low_memory=False)
        
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