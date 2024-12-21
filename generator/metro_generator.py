"""
Main generator class for Ankara Metro passenger data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from .station_config import StationConfig
from .time_patterns import TimePatterns
from .event_patterns import EventPatterns
from .utils import (
    calculate_transfer_passengers,
    calculate_boarding_ratio,
    calculate_station_popularity,
    calculate_occupancy_rate
)

class AnkaraMetroGenerator:
    def __init__(self):
        self.station_config = StationConfig()
        self.time_patterns = TimePatterns()
        self.event_patterns = EventPatterns()
    
    def generate_passenger_flow(self, num_days=7, start_date=None):
        """Generate synthetic passenger flow data for Ankara Metro"""
        if start_date is None:
            start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        station_characteristics = self.station_config.generate_station_characteristics()
        data = []

        for day in range(num_days):
            current_date = start_date + timedelta(days=day)
            is_weekend = current_date.weekday() >= 5
            month_factor = self.time_patterns.get_monthly_factor(current_date.month)

            for hour in range(24):
                time_factor = self.time_patterns.get_time_factor(hour, is_weekend)
                if time_factor == 0:
                    continue

                # Get random weather condition and its impact
                weather = np.random.choice(list(self.event_patterns.weather_factors.keys()))
                weather_info = self.event_patterns.get_weather_factor(weather)

                if np.random.random() < weather_info['disruption_prob']:
                    weather_factor = weather_info['factor'] * 0.8
                else:
                    weather_factor = weather_info['factor']

                for line, details in self.station_config.metro_lines.items():
                    # Line-specific weekend adjustments for base passenger numbers
                    if is_weekend:
                        if line == 'M1-2-3':
                            weekend_line_factor = 1.3  # Higher weekend activity
                        elif line == 'M4':
                            weekend_line_factor = 0.8  # Moderate weekend activity
                        else:  # A1 line
                            weekend_line_factor = 0.6  # Much lower weekend activity
                    else:
                        weekend_line_factor = 1.0

                    if self.time_patterns.is_peak_hour(hour, is_weekend):
                        frequency = details['frequency_minutes']['peak']
                    elif hour >= 23 or hour <= 5:
                        frequency = details['frequency_minutes']['off_peak']
                    else:
                        frequency = details['frequency_minutes']['regular']

                    trains_per_hour = max(1, 60 // frequency)

                    for station in details['stations']:
                        station_info = station_characteristics[station]
                        
                        # Calculate station popularity with all factors
                        station_popularity = calculate_station_popularity(
                            station, hour, is_weekend,
                            self.station_config.station_features,
                            details['stations']
                        )
                        
                        # Calculate base passenger flow
                        if station == '15 Temmuz Kızılay Millî İrade':
                            base_pass = 100 * weekend_line_factor
                        elif line == 'M4':
                            base_pass = 80 * weekend_line_factor
                        else:
                            base_pass = 70 * weekend_line_factor

                        # Terminal stations adjustments
                        if station in details['terminal_stations']:
                            if is_weekend:
                                if line == 'M1-2-3':
                                    base_pass = max(30, base_pass * 0.6)
                                else:
                                    base_pass = max(20, base_pass * 0.4)
                            else:
                                base_pass = max(25, base_pass * 0.5)

                        # Calculate event impact
                        event_factor = self.event_patterns.get_event_factor(station, current_date, hour)

                        base_flow = int(
                            station_popularity *
                            time_factor *
                            weather_factor *
                            month_factor *
                            event_factor *
                            trains_per_hour *
                            np.random.normal(1, 0.1) *
                            base_pass
                        )

                        # Calculate transfers and boarding/alighting split
                        transfers = calculate_transfer_passengers(
                            station, hour, is_weekend, base_flow,
                            sum([d['junction_stations'] for d in self.station_config.metro_lines.values()], [])
                        )
                        
                        boarding_ratio = calculate_boarding_ratio(
                            station, details['stations'], hour, is_weekend
                        )
                        
                        boarding = int(base_flow * boarding_ratio)
                        alighting = base_flow - boarding

                        # Calculate occupancy rate
                        occupancy_rate = calculate_occupancy_rate(
                            boarding, transfers, station_info['capacity'],
                            station, details['stations'], hour, is_weekend
                        )

                        data.append({
                            'Timestamp': current_date.replace(hour=hour),
                            'Metro_Line': line,
                            'Station_ID': station,
                            'Station_Type': station_info['type'],
                            'Weather_Condition': weather,
                            'Is_Weekend': is_weekend,
                            'Time_Period': self.time_patterns.get_time_period(hour),
                            'Service_Frequency': frequency,
                            'Trains_Per_Hour': trains_per_hour,
                            'Weather_Disruption': weather_factor < weather_info['factor'],
                            'Boarding_Passengers': boarding,
                            'Alighting_Passengers': alighting,
                            'Transfer_Out': transfers,
                            'Capacity_Utilization': (boarding + transfers) / station_info['capacity'],
                            'Occupancy_Rate': occupancy_rate
                        })

        df = pd.DataFrame(data)
        df = df.sort_values(['Timestamp', 'Metro_Line', 'Station_ID']).reset_index(drop=True)
        return df 