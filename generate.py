#!/usr/bin/env python3
"""
Generate synthetic Ankara Metro passenger data
"""

import os
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from train.station_config import STATION_ORDER, WEATHER_FACTORS
from data.generate_data import (
    generate_timestamps,
    get_station_type,
    generate_base_occupancy,
    generate_passenger_flows
)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Generate synthetic Ankara Metro passenger data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default='2023-01-01',
        help='Start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default='2023-12-31',
        help='End date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--freq',
        type=str,
        default='5min',
        help='Sampling frequency (e.g. 5min, 10min, 1H)'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default='ankara_metro_crowding_data_realistic.csv',
        help='Output CSV file name'
    )
    
    parser.add_argument(
        '--out-dir', 
        type=str, 
        default='data',
        help='Output directory for the CSV file'
    )
    
    return parser.parse_args()

def main():
    """Main function to generate data"""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Construct full output path
    output_path = os.path.join(args.out_dir, args.output)
    
    print(f"Generating Ankara Metro passenger data from {args.start_date} to {args.end_date}")
    print(f"Using {args.freq} sampling frequency")
    print(f"Output will be saved to: {output_path}")
    
    # Generate timestamps
    timestamps = generate_timestamps(args.start_date, args.end_date, args.freq)
    
    # Prepare data lists
    data = []
    
    # Generate data for each metro line and station
    total_stations = sum(len(stations) for stations in STATION_ORDER.values())
    station_count = 0
    
    for metro_line, stations in STATION_ORDER.items():
        print(f"\nGenerating data for {metro_line}...")
        
        for station in stations:
            station_count += 1
            print(f"Processing station {station_count}/{total_stations}: {station}")
            
            station_type = get_station_type(station)
            
            for ts in timestamps:
                hour = ts.hour
                is_weekend = ts.weekday() >= 5
                
                # Generate occupancy and flows
                occupancy = generate_base_occupancy(hour, station_type, is_weekend)
                boarding, alighting, transfers = generate_passenger_flows(
                    occupancy, station_type, hour, is_weekend
                )
                
                # Determine time period
                if (7 <= hour <= 9) or (16 <= hour <= 19):
                    time_period = 'peak'
                elif hour >= 23 or hour <= 5:
                    time_period = 'off_peak'
                else:
                    time_period = 'regular'
                
                # Get service frequency
                if metro_line == 'M4':
                    frequencies = {'peak': 4, 'regular': 7, 'off_peak': 15}
                else:
                    frequencies = {'peak': 3, 'regular': 5, 'off_peak': 10}
                
                frequency = frequencies[time_period]
                trains_per_hour = max(1, 60 // frequency)
                
                # Generate weather
                weather = np.random.choice(
                    list(WEATHER_FACTORS.keys()),
                    p=[0.4, 0.3, 0.2, 0.05, 0.05]
                )
                weather_info = WEATHER_FACTORS[weather]
                weather_disruption = np.random.random() < weather_info['disruption_prob']
                
                # Calculate capacity utilization
                capacity_utilization = (boarding + alighting) / (trains_per_hour * 1000)
                
                # Add row to data
                data.append({
                    'Timestamp': ts,
                    'Metro_Line': metro_line,
                    'Station_ID': station,
                    'Station_Type': station_type,
                    'Weather_Condition': weather,
                    'Time_Period': time_period,
                    'Is_Weekend': is_weekend,
                    'Weather_Disruption': weather_disruption,
                    'Service_Frequency': frequency,
                    'Trains_Per_Hour': trains_per_hour,
                    'Boarding_Passengers': int(boarding),
                    'Alighting_Passengers': int(alighting),
                    'Transfer_Out': int(transfers),
                    'Capacity_Utilization': min(1.0, capacity_utilization),
                    'Occupancy_Rate': occupancy
                })
    
    # Create DataFrame
    print("\nCreating DataFrame...")
    df = pd.DataFrame(data)
    
    # Save to CSV
    print(f"\nSaving dataset to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"Total records: {len(df):,}")

if __name__ == '__main__':
    main() 