import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class MetroDataGenerator:
    def __init__(self):
        self.TRAIN_CAPACITY = 650
        self.NUM_STATIONS = 57
        self.CENTRAL_STATION = 29
        
        # Define junction stations (intersections between lines)
        self.junction_stations = {
            'S15': {'lines': ['M1', 'M2']},
            'S30': {'lines': ['M2', 'M3']},
            'S45': {'lines': ['M3', 'M4']}
        }
        
        # Metro lines configuration with terminals
        self.metro_lines = {
            'M1': {
                'stations': list(range(1, 16)),
                'terminals': [1, 15],
                'frequency_minutes': {'peak': 5, 'regular': 8, 'off_peak': 12}
            },
            'M2': {
                'stations': list(range(15, 31)),
                'terminals': [15, 30],
                'frequency_minutes': {'peak': 4, 'regular': 7, 'off_peak': 10}
            },
            'M3': {
                'stations': list(range(30, 46)),
                'terminals': [30, 45],
                'frequency_minutes': {'peak': 5, 'regular': 8, 'off_peak': 12}
            },
            'M4': {
                'stations': list(range(45, 58)),
                'terminals': [45, 57],
                'frequency_minutes': {'peak': 6, 'regular': 9, 'off_peak': 15}
            }
        }
        
        # Generate station characteristics
        self.station_characteristics = self._generate_station_characteristics()
        
    def _generate_station_characteristics(self):
        """Generate comprehensive station characteristics"""
        characteristics = {}
        for i in range(1, self.NUM_STATIONS + 1):
            station_id = f'S{i}'
            
            # Calculate distance from central station
            distance = abs(i - self.CENTRAL_STATION)
            
            # Base popularity based on distance from center
            base_popularity = np.exp(-0.1 * distance)
            
            # Add characteristics based on station type
            if station_id in self.junction_stations:
                # Junction stations have higher base popularity
                base_popularity *= 1.5
                station_type = 'junction'
            elif i in [1, 15, 30, 45, 57]:  # Terminal stations
                base_popularity *= 0.7
                station_type = 'terminal'
            elif abs(i - self.CENTRAL_STATION) <= 5:
                # Stations near center have higher popularity
                base_popularity *= 1.3
                station_type = 'central'
            else:
                station_type = 'regular'
            
            # Add some randomization for variety
            popularity = base_popularity * random.uniform(0.9, 1.1)
            
            characteristics[station_id] = {
                'popularity': popularity,
                'type': station_type,
                'peak_multiplier': random.uniform(1.8, 2.2) if station_type in ['central', 'junction'] else random.uniform(1.4, 1.8),
                'weekend_factor': random.uniform(0.4, 0.6) if station_type == 'central' else random.uniform(0.3, 0.5)
            }
        
        return characteristics
    
    def _get_time_factors(self, hour, is_weekend):
        """Calculate detailed time-based factors for passenger numbers"""
        # Base time factors
        if 7 <= hour < 9:  # Morning peak
            base_factor = random.uniform(0.8, 1.0)
            time_period = 'peak'
        elif 17 <= hour < 19:  # Evening peak
            base_factor = random.uniform(0.8, 1.0)
            time_period = 'peak'
        elif 23 <= hour or hour < 5:  # Night/Early morning
            base_factor = random.uniform(0.1, 0.2)
            time_period = 'off_peak'
        elif hour in [6, 9, 16, 19]:  # Shoulder peak hours
            base_factor = random.uniform(0.6, 0.8)
            time_period = 'regular'
        else:  # Regular hours
            base_factor = random.uniform(0.3, 0.5)
            time_period = 'regular'
        
        # Adjust for weekend
        if is_weekend:
            if 10 <= hour < 20:  # Weekend shopping/leisure hours
                base_factor *= random.uniform(0.6, 0.8)
            else:
                base_factor *= random.uniform(0.4, 0.6)
        
        return base_factor, time_period
    
    def _calculate_transfer_passengers(self, station_id, current_line, current_load, time_factor):
        """Calculate passengers transferring at junction stations"""
        if station_id in self.junction_stations:
            transfer_lines = [line for line in self.junction_stations[station_id]['lines'] 
                            if line != current_line]
            
            # Calculate transfer rate based on current load and time factor
            base_transfer = int(current_load * random.uniform(0.15, 0.25) * time_factor)
            
            # Distribute transfers among available lines
            transfers = {}
            for line in transfer_lines:
                transfers[line] = int(base_transfer / len(transfer_lines))
            
            return transfers
        return {}
    
    def generate_passenger_flow(self, start_date, num_days=1):
        """Generate passenger flow data for specified number of days"""
        data = []
        current_date = start_date
        
        for day in range(num_days):
            is_weekend = current_date.weekday() >= 5
            
            # Generate data for each hour
            for hour in range(24):
                time_factor, time_period = self._get_time_factors(hour, is_weekend)
                
                # Generate data for each metro line
                for line, line_info in self.metro_lines.items():
                    # Calculate number of trains per hour based on frequency
                    frequency = line_info['frequency_minutes'][time_period]
                    trains_per_hour = 60 // frequency
                    
                    # Generate data for each train
                    for train in range(trains_per_hour):
                        current_load = 0
                        minute_offset = (train * frequency)
                        timestamp = current_date + timedelta(hours=hour, minutes=minute_offset)
                        
                        for station_num in line_info['stations']:
                            station_id = f'S{station_num}'
                            station_chars = self.station_characteristics[station_id]
                            
                            # Calculate boarding passengers
                            base_boarding = int(
                                station_chars['popularity'] * 
                                time_factor * 
                                station_chars['peak_multiplier'] * 
                                (station_chars['weekend_factor'] if is_weekend else 1.0) * 
                                200
                            )
                            
                            # Adjust for terminal stations
                            if station_num in line_info['terminals']:
                                if station_num == line_info['terminals'][0]:  # First terminal
                                    boarding = base_boarding
                                    alighting = 0
                                else:  # Last terminal
                                    boarding = 0
                                    alighting = current_load
                            else:
                                # Regular station boarding/alighting
                                boarding = min(base_boarding, self.TRAIN_CAPACITY - current_load)
                                alighting = int(current_load * random.uniform(0.1, 0.3))
                            
                            # Handle transfers at junction stations
                            transfers = self._calculate_transfer_passengers(
                                station_id, line, current_load, time_factor
                            )
                            
                            # Update current load
                            current_load = current_load + boarding - alighting
                            
                            # Calculate capacity utilization
                            capacity_utilization = (current_load / self.TRAIN_CAPACITY) * 100
                            
                            record = {
                                'Timestamp': timestamp,
                                'Metro_Line': line,
                                'Station_ID': station_id,
                                'Station_Type': station_chars['type'],
                                'Boarding_Passengers': boarding,
                                'Alighting_Passengers': alighting,
                                'Transfer_Out': sum(transfers.values()) if transfers else 0,
                                'Current_Load': current_load,
                                'Capacity_Utilization': capacity_utilization,
                                'Is_Weekend': is_weekend,
                                'Time_Period': time_period
                            }
                            
                            # Add transfer details if applicable
                            for transfer_line, transfer_count in transfers.items():
                                record[f'Transfer_To_{transfer_line}'] = transfer_count
                            
                            data.append(record)
            
            current_date += timedelta(days=1)
        
        return pd.DataFrame(data)

def main():
    # Initialize generator
    generator = MetroDataGenerator()
    
    # Generate data for 7 days starting from today
    start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    df = generator.generate_passenger_flow(start_date, num_days=7)
    
    # Save to CSV
    output_file = 'metro_passenger_data.csv'
    df.to_csv(output_file, index=False)
    print(f"Generated {len(df)} records and saved to {output_file}")
    
    # Print some statistics
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