"""
Utility functions for the Ankara Metro prediction system
"""

import pandas as pd
import os
from datetime import datetime
from .station_config import STATION_ORDER, SERVICE_FREQUENCIES

def get_data_path():
    """Get the path to the data file"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, 'data', 'ankara_metro_crowding_data_realistic.csv')

def get_available_stations(metro_line, data_path=None):
    """
    Get available stations for a given metro line
    
    Args:
        metro_line (str): Metro line identifier
        data_path (str): Path to the data file
        
    Returns:
        list: List of available stations
    """
    if data_path is None:
        data_path = get_data_path()
        
    df = pd.read_csv(data_path, low_memory=False)
    return df[df['Metro_Line'] == metro_line]['Station_ID'].unique().tolist()

def determine_time_period(hour):
    """
    Determine time period based on hour
    
    Args:
        hour (int): Hour of the day (0-23)
        
    Returns:
        str: Time period ('peak', 'regular', or 'off_peak')
    """
    if (7 <= hour <= 9) or (16 <= hour <= 19):
        return 'peak'
    elif hour >= 23 or hour <= 5:
        return 'off_peak'
    else:
        return 'regular'

def get_service_frequency(metro_line, time_period):
    """
    Get service frequency for a given metro line and time period
    
    Args:
        metro_line (str): Metro line identifier
        time_period (str): Time period ('peak', 'regular', or 'off_peak')
        
    Returns:
        int: Service frequency in minutes
    """
    return SERVICE_FREQUENCIES[metro_line][time_period]

def calculate_trains_per_hour(frequency):
    """
    Calculate trains per hour based on frequency
    
    Args:
        frequency (int): Service frequency in minutes
        
    Returns:
        int: Number of trains per hour
    """
    return max(1, 60 // frequency)

def get_station_position(metro_line, station):
    """
    Get station position information
    
    Args:
        metro_line (str): Metro line identifier
        station (str): Station name
        
    Returns:
        tuple: (station_idx, total_stations, distance_from_kizilay)
    """
    station_idx = STATION_ORDER[metro_line].index(station)
    total_stations = len(STATION_ORDER[metro_line])
    
    if metro_line == 'M1-2-3':
        kizilayIdx = STATION_ORDER[metro_line].index('15 Temmuz Kızılay Millî İrade')
        distance_from_kizilay = abs(station_idx - kizilayIdx)
    else:
        distance_from_kizilay = None
        
    return station_idx, total_stations, distance_from_kizilay

def format_prediction_results(prediction, station, metro_line, hour, weather, is_weekend):
    """
    Format prediction results for display
    
    Args:
        prediction (float): Predicted occupancy rate
        station (str): Station name
        metro_line (str): Metro line identifier
        hour (int): Hour of the day
        weather (str): Weather condition
        is_weekend (bool): Whether it's a weekend
        
    Returns:
        str: Formatted results string
    """
    results = [
        "\nPrediction Results:",
        "-" * 40,
        f"Location: {station} Station ({metro_line} Line)",
        f"Time: {hour:02d}:00",
        f"Weather: {weather}",
        f"Day type: {'Weekend' if is_weekend else 'Weekday'}",
        f"Predicted occupancy rate: {prediction:.1f}%"
    ]
    return "\n".join(results) 