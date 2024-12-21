"""
Ankara Metro Occupancy Prediction System

This package provides tools for predicting passenger occupancy rates
in the Ankara Metro system based on various factors including time,
weather, and station location.
"""

from .predictor import MetroPassengerPredictor
from .cli import get_prediction_from_model, main
from .evaluation import evaluate_model, interpret_occupancy
from .utils import (
    get_available_stations,
    determine_time_period,
    get_service_frequency,
    calculate_trains_per_hour,
    get_station_position,
    format_prediction_results
)

__version__ = '1.0.0'
__author__ = 'cyberia'

__all__ = [
    'MetroPassengerPredictor',
    'get_prediction_from_model',
    'main',
    'evaluate_model',
    'interpret_occupancy',
    'get_available_stations',
    'determine_time_period',
    'get_service_frequency',
    'calculate_trains_per_hour',
    'get_station_position',
    'format_prediction_results'
] 