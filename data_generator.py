"""
Metro Passenger Data Generator

This script generates synthetic metro passenger data with realistic patterns:
- 4 metro lines (M1-M4) with 57 stations total
- Bell-curve distribution of passengers (peaks at central stations)
- Time-based variations (peak hours, off-peak hours)
- Different station types (central, junction, terminal, regular)
- Transfer patterns at junction stations
- Weekend vs weekday patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class MetroDataGenerator:
    """
    Generates synthetic metro passenger flow data with realistic patterns.
    
    Key Parameters:
    - Train Capacity: 650 passengers per train
    - Number of Stations: 57 (S1-S57)
    - Central Station: S29 (highest traffic)
    - Junction Stations: S15, S30, S45 (transfer points between lines)
    """
    
    def __init__(self):
        # Basic system parameters
        self.TRAIN_CAPACITY = 650
        self.NUM_STATIONS = 57
        self.CENTRAL_STATION = 29
        
        # Define transfer points between lines
        self.junction_stations = {
            'S15': {'lines': ['M1', 'M2']},  # Transfer point between M1 and M2
            'S30': {'lines': ['M2', 'M3']},  # Transfer point between M2 and M3
            'S45': {'lines': ['M3', 'M4']}   # Transfer point between M3 and M4
        }
        
        # Define metro line configurations
        self.metro_lines = self._initialize_metro_lines()
        
        # Generate characteristics for each station
        self.station_characteristics = self._generate_station_characteristics()
    
    def _initialize_metro_lines(self):
        """
        Initialize metro line configurations with:
        - Station ranges
        - Terminal stations
        - Train frequencies for different time periods
        """
        return {
            'M1': {
                'stations': list(range(1, 16)),    # Stations S1-S15
                'terminals': [1, 15],              # First and last stations
                'frequency_minutes': {
                    'peak': 5,      # Train every 5 minutes during peak
                    'regular': 8,   # Train every 8 minutes during regular hours
                    'off_peak': 12  # Train every 12 minutes during off-peak
                }
            },
            'M2': {
                'stations': list(range(15, 31)),   # Stations S15-S30
                'terminals': [15, 30],
                'frequency_minutes': {'peak': 4, 'regular': 7, 'off_peak': 10}
            },
            'M3': {
                'stations': list(range(30, 46)),   # Stations S30-S45
                'terminals': [30, 45],
                'frequency_minutes': {'peak': 5, 'regular': 8, 'off_peak': 12}
            },
            'M4': {
                'stations': list(range(45, 58)),   # Stations S45-S57
                'terminals': [45, 57],
                'frequency_minutes': {'peak': 6, 'regular': 9, 'off_peak': 15}
            }
        }
    
    def _generate_station_characteristics(self):
        """
        Generate characteristics for each station based on:
        1. Distance from central station (bell curve distribution)
        2. Station type (central, junction, terminal, regular)
        3. Peak hour multipliers
        4. Weekend adjustment factors
        """
        characteristics = {}
        
        for i in range(1, self.NUM_STATIONS + 1):
            station_id = f'S{i}'
            distance = abs(i - self.CENTRAL_STATION)
            
            # Create bell curve effect (highest at center, decreasing towards ends)
            base_popularity = np.exp(-0.5 * (distance / 10)**2)
            
            # Determine station type and adjust popularity
            if station_id in self.junction_stations:
                station_type = 'junction'
                base_popularity *= 1.8  # Junction stations are 80% more popular
            elif i in [1, 15, 30, 45, 57]:
                station_type = 'terminal'
                base_popularity *= 0.4  # Terminal stations are 60% less popular
            elif abs(i - self.CENTRAL_STATION) <= 5:
                station_type = 'central'
                base_popularity *= 1.5  # Central area stations are 50% more popular
            else:
                station_type = 'regular'
            
            # Add small random variation (±5%)
            popularity = base_popularity * random.uniform(0.95, 1.05)
            
            # Store station characteristics
            characteristics[station_id] = {
                'popularity': popularity,
                'type': station_type,
                # Higher multiplier for central and junction stations during peak hours
                'peak_multiplier': random.uniform(2.0, 2.5) if station_type in ['central', 'junction'] 
                                 else random.uniform(1.2, 1.6),
                # Weekend factors (central stations remain busier on weekends)
                'weekend_factor': random.uniform(0.5, 0.7) if station_type == 'central' 
                                else random.uniform(0.3, 0.5)
            }
        
        return characteristics
    
    def _get_time_factors(self, hour, is_weekend):
        """
        Calculate passenger volume factors based on time of day.
        
        Time Periods:
        - Morning Peak (7-9 AM): Highest volume
        - Evening Peak (5-7 PM): Slightly higher than morning
        - Night/Early Morning (11 PM-5 AM): Lowest volume
        - Shoulder Peak (6-7 AM, 9-10 AM, 4-5 PM, 7-8 PM): Medium-high volume
        - Regular Hours: Medium volume
        """
        # Determine base time factor and period
        if 7 <= hour < 9:  # Morning peak
            base_factor = random.uniform(0.85, 1.0)
            time_period = 'peak'
        elif 17 <= hour < 19:  # Evening peak
            base_factor = random.uniform(0.90, 1.0)
            time_period = 'peak'
        elif 23 <= hour or hour < 5:  # Night/Early morning
            base_factor = random.uniform(0.05, 0.15)
            time_period = 'off_peak'
        elif hour in [6, 9, 16, 19]:  # Shoulder peak hours
            base_factor = random.uniform(0.65, 0.85)
            time_period = 'regular'
        else:  # Regular hours
            base_factor = random.uniform(0.35, 0.55)
            time_period = 'regular'
        
        # Adjust for weekends
        if is_weekend:
            if 10 <= hour < 20:  # Weekend shopping/leisure hours
                base_factor *= random.uniform(0.7, 0.9)
            else:
                base_factor *= random.uniform(0.3, 0.5)
        
        return base_factor, time_period
    
    def _calculate_transfer_passengers(self, station_id, current_line, current_load, time_factor):
        """
        Calculate number of passengers transferring between lines at junction stations.
        
        Transfer rates are higher during peak hours and at busier stations.
        """
        if station_id in self.junction_stations:
            # Get other lines at this junction
            transfer_lines = [line for line in self.junction_stations[station_id]['lines'] 
                            if line != current_line]
            
            # Higher transfer rates during peak hours
            if time_factor > 0.8:  # Peak hours
                transfer_rate = random.uniform(0.25, 0.35)  # 25-35% transfer during peak
            else:  # Off-peak hours
                transfer_rate = random.uniform(0.15, 0.25)  # 15-25% transfer during off-peak
            
            # Calculate total transfers
            base_transfer = int(current_load * transfer_rate * time_factor)
            
            # Distribute transfers among available lines
            transfers = {}
            for line in transfer_lines:
                transfers[line] = int(base_transfer / len(transfer_lines))
            
            return transfers
        return {}
    
    def generate_passenger_flow(self, start_date, num_days=1):
        """
        Generate passenger flow data for the specified number of days.
        
        For each day, generates:
        1. Hourly data for each metro line
        2. Multiple trains per hour based on frequency
        3. Passenger flows at each station (boarding, alighting, transfers)
        4. Capacity utilization metrics
        """
        data = []
        current_date = start_date
        
        for day in range(num_days):
            is_weekend = current_date.weekday() >= 5
            
            # Generate hourly data
            for hour in range(24):
                time_factor, time_period = self._get_time_factors(hour, is_weekend)
                
                # Generate data for each metro line
                for line, line_info in self.metro_lines.items():
                    # Calculate trains per hour based on frequency
                    frequency = line_info['frequency_minutes'][time_period]
                    trains_per_hour = 60 // frequency
                    
                    # Generate data for each train
                    for train in range(trains_per_hour):
                        self._generate_train_journey(
                            data, current_date, hour, train, frequency,
                            line, line_info, time_factor, time_period, is_weekend
                        )
            
            current_date += timedelta(days=1)
        
        return pd.DataFrame(data)
    
    def _generate_train_journey(self, data, current_date, hour, train, frequency,
                              line, line_info, time_factor, time_period, is_weekend):
        """
        Generate passenger flow data for a single train journey through all stations.
        """
        current_load = 0
        minute_offset = (train * frequency)
        timestamp = current_date + timedelta(hours=hour, minutes=minute_offset)
        
        # Process each station in the line
        for station_num in line_info['stations']:
            station_id = f'S{station_num}'
            station_chars = self.station_characteristics[station_id]
            
            # Calculate base boarding passengers
            base_boarding = int(
                station_chars['popularity'] * 
                time_factor * 
                station_chars['peak_multiplier'] * 
                (station_chars['weekend_factor'] if is_weekend else 1.0) * 
                300
            )
            
            # Add randomization to boarding
            base_boarding = int(base_boarding * random.uniform(0.8, 1.2))
            
            # Handle passenger flow at station
            boarding, alighting = self._calculate_station_flow(
                station_num, base_boarding, current_load, line_info
            )
            
            # Calculate transfers at junction stations
            transfers = self._calculate_transfer_passengers(
                station_id, line, current_load, time_factor
            )
            
            # Update current load
            current_load = min(
                current_load + boarding - alighting,
                self.TRAIN_CAPACITY
            )
            
            # Record station data
            record = {
                'Timestamp': timestamp,
                'Metro_Line': line,
                'Station_ID': station_id,
                'Station_Type': station_chars['type'],
                'Boarding_Passengers': boarding,
                'Alighting_Passengers': alighting,
                'Transfer_Out': sum(transfers.values()) if transfers else 0,
                'Current_Load': current_load,
                'Capacity_Utilization': (current_load / self.TRAIN_CAPACITY) * 100,
                'Is_Weekend': is_weekend,
                'Time_Period': time_period
            }
            
            # Add transfer details
            for transfer_line, transfer_count in transfers.items():
                record[f'Transfer_To_{transfer_line}'] = transfer_count
            
            data.append(record)
    
    def _calculate_station_flow(self, station_num, base_boarding, current_load, line_info):
        """
        Calculate boarding and alighting passengers at a station.
        Handles terminal stations differently from regular stations.
        """
        if station_num in line_info['terminals']:
            if station_num == line_info['terminals'][0]:  # First terminal
                return base_boarding, 0
            else:  # Last terminal
                return 0, current_load
        else:
            # Regular station flow
            boarding = min(base_boarding, self.TRAIN_CAPACITY - current_load)
            
            # Calculate alighting based on remaining stations
            remaining_stations = len(line_info['stations']) - line_info['stations'].index(station_num)
            alighting_rate = random.uniform(0.3, 0.5) if remaining_stations <= 3 else random.uniform(0.1, 0.3)
            alighting = int(current_load * alighting_rate)
            
            return boarding, alighting

def main():
    """
    Generate one week of metro passenger data and save to CSV.
    Also prints basic statistics about the generated data.
    """
    # Initialize generator
    generator = MetroDataGenerator()
    
    # Generate data for 7 days starting from today
    start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    df = generator.generate_passenger_flow(start_date, num_days=7)
    
    # Save to CSV
    output_file = 'metro_passenger_data.csv'
    df.to_csv(output_file, index=False)
    print(f"Generated {len(df)} records and saved to {output_file}")
    
    # Print statistics
    print("\nData Statistics:")
    print(f"Total number of records: {len(df)}")
    print("\nAverage passengers by time period:")
    print(df.groupby('Time_Period')['Current_Load'].mean())
    print("\nAverage capacity utilization by station type:")
    print(df.groupby('Station_Type')['Capacity_Utilization'].mean())
    print("\nBusiest stations (by total boarding passengers):")
    station_traffic = df.groupby('Station_ID')['Boarding_Passengers'].sum().sort_values(ascending=False)
    print(station_traffic.head())

if __name__ == "__main__":
    main() 