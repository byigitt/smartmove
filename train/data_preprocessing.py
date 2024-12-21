"""
Data preprocessing and feature engineering functions for the Ankara Metro prediction system
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def prepare_features(df, store_categories=True):
    """
    Prepare features for the model with enhanced engineering
    
    Args:
        df (pd.DataFrame): Input dataframe
        store_categories (bool): Whether to store unique categories
        
    Returns:
        pd.DataFrame: Processed dataframe
        dict: Categories if store_categories is True, else None
    """
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
    
    categories = None
    if store_categories:
        categories = {
            'weather': df['Weather_Condition'].unique(),
            'station_types': df['Station_Type'].unique(),
            'metro_lines': df['Metro_Line'].unique(),
            'time_periods': df['Time_Period'].unique()
        }
    
    return df, categories

def create_feature_pipeline():
    """
    Create an enhanced feature preprocessing pipeline
    
    Returns:
        ColumnTransformer: Scikit-learn preprocessing pipeline
    """
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
    preprocessor = ColumnTransformer(
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
    
    return preprocessor

def get_time_factor(hour, is_weekend=False):
    """
    Calculate time-based factor for occupancy
    
    Args:
        hour (int): Hour of the day (0-23)
        is_weekend (bool): Whether it's a weekend
        
    Returns:
        float: Time-based occupancy factor
    """
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

def is_peak_hour(hour, is_weekend=False):
    """
    Determine if the given hour is a peak hour
    
    Args:
        hour (int): Hour of the day (0-23)
        is_weekend (bool): Whether it's a weekend
        
    Returns:
        bool: Whether it's a peak hour
    """
    if is_weekend:
        return False  # No peak hours on weekends
    
    # Core peak hours
    morning_peak = 7 <= hour <= 9
    evening_peak = 16 <= hour <= 19
    
    # Transition hours (shoulder periods)
    morning_shoulder = 6 <= hour <= 10  # One hour before and after morning peak
    evening_shoulder = 15 <= hour <= 21  # One hour before and two hours after evening peak
    
    # Return peak status
    return morning_peak or evening_peak or morning_shoulder or evening_shoulder 