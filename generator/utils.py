"""
Utility functions for passenger flow calculations
"""

import numpy as np

def calculate_transfer_passengers(station, hour, is_weekend, base_flow, junction_stations):
    """Calculate number of transfer passengers at a station"""
    if station not in junction_stations:
        return 0
    if hour < 6 or hour >= 23:
        return 0

    if station == '15 Temmuz Kızılay Millî İrade':
        if is_weekend:
            transfer_rate = 0.5
        else:
            if 7 <= hour <= 9:
                transfer_rate = 0.8
            elif 16 <= hour <= 19:
                transfer_rate = 0.85
            else:
                transfer_rate = 0.6
    else:
        if is_weekend:
            transfer_rate = 0.3
        else:
            if 7 <= hour <= 9:
                transfer_rate = 0.5
            elif 16 <= hour <= 19:
                transfer_rate = 0.6
            else:
                transfer_rate = 0.4

    return int(base_flow * transfer_rate)

def calculate_boarding_ratio(station, line_stations, hour, is_weekend):
    """Calculate boarding vs alighting ratio based on station position and time"""
    station_idx = line_stations.index(station)
    total_stations = len(line_stations)
    
    # Find distance from Kızılay
    if '15 Temmuz Kızılay Millî İrade' in line_stations:
        kizilay_idx = line_stations.index('15 Temmuz Kızılay Millî İrade')
        # Calculate distance from Kızılay (0 to 1, where 1 is furthest)
        distance_from_kizilay = abs(station_idx - kizilay_idx) / (total_stations/2)
        
        if not is_weekend:
            if 7 <= hour <= 9:  # Morning peak
                if station_idx < kizilay_idx:  # Before Kızılay
                    # More boarding at residential areas in the morning
                    boarding_ratio = 0.8 * (1 - distance_from_kizilay * 0.3)
                else:  # After Kızılay
                    # More alighting as we get further from Kızılay
                    boarding_ratio = 0.3 * (1 - distance_from_kizilay * 0.5)
            elif 16 <= hour <= 19:  # Evening peak
                if station_idx < kizilay_idx:  # Before Kızılay
                    # More alighting in residential areas in the evening
                    boarding_ratio = 0.3 * (1 - distance_from_kizilay * 0.5)
                else:  # After Kızılay
                    # More boarding as people return home
                    boarding_ratio = 0.7 * (1 - distance_from_kizilay * 0.4)
            else:  # Off-peak
                # Base boarding ratio decreases with distance from Kızılay
                boarding_ratio = 0.5 * (1 - distance_from_kizilay * 0.6)
        else:  # Weekend
            # Less pronounced patterns on weekends
            boarding_ratio = 0.4 * (1 - distance_from_kizilay * 0.5)
    else:
        # For lines without Kızılay, use simpler distance-based logic
        mid_point = total_stations // 2
        relative_pos = (station_idx - mid_point) / total_stations
        boarding_ratio = 0.5 * (1 - abs(relative_pos))
    
    return max(0.1, min(0.8, boarding_ratio))

def calculate_station_popularity(station, hour, is_weekend, station_features, line_stations):
    """Calculate station popularity based on features and time"""
    base_popularity = 1.0
    features = station_features.get(station, [])
    
    # Special handling for Kızılay
    if station == '15 Temmuz Kızılay Millî İrade':
        base_popularity = 1.4  # Ensure Kızılay has highest base popularity
        if is_weekend:
            base_popularity *= 1.3  # Even higher on weekends due to shopping/entertainment
        return base_popularity
    
    # Find distance from Kızılay for popularity decay
    if '15 Temmuz Kızılay Millî İrade' in line_stations:
        kizilay_idx = line_stations.index('15 Temmuz Kızılay Millî İrade')
        station_idx = line_stations.index(station)
        distance_from_kizilay = abs(station_idx - kizilay_idx) / len(line_stations)
        
        # Steeper decay for stations immediately adjacent to Kızılay
        if abs(station_idx - kizilay_idx) <= 2:
            distance_factor = np.exp(-2.5 * distance_from_kizilay)
        else:
            distance_factor = np.exp(-2.0 * distance_from_kizilay)
        
        # Ensure adjacent stations never exceed 85% of Kızılay's popularity
        if abs(station_idx - kizilay_idx) <= 2:
            distance_factor = min(distance_factor, 0.85)
        
        base_popularity *= max(0.3, distance_factor)
    
    # Apply feature-based adjustments
    for feature in features:
        if feature == 'university_area':
            if is_weekend:
                base_popularity *= 0.3
            elif 8 <= hour <= 18:
                base_popularity *= 1.4
        elif feature == 'shopping_district':
            if is_weekend:
                if 12 <= hour <= 20:
                    base_popularity *= 1.3
            else:
                if 10 <= hour <= 19:
                    base_popularity *= 1.2
        elif feature == 'business_district':
            if not is_weekend:
                if 8 <= hour <= 10 or 16 <= hour <= 18:
                    base_popularity *= 1.3
                elif 10 <= hour <= 16:
                    base_popularity *= 1.2
            else:
                base_popularity *= 0.4
        elif feature == 'residential_high_density':
            if not is_weekend:
                if 7 <= hour <= 9:
                    base_popularity *= 1.4
                elif 17 <= hour <= 19:
                    base_popularity *= 1.3
        elif feature == 'hospital_zone':
            if 8 <= hour <= 17:
                base_popularity *= 1.2
            if is_weekend:
                base_popularity *= 0.7
    
    return base_popularity

def calculate_occupancy_rate(boarding, transfers, capacity, station, line_stations, hour, is_weekend):
    """Calculate station occupancy rate with realistic constraints"""
    base_occupancy = (boarding + transfers) * 100 / capacity
    
    if station == '15 Temmuz Kızılay Millî İrade':
        # Ensure Kızılay maintains highest occupancy
        if 7 <= hour <= 9 or 16 <= hour <= 19:
            occupancy_rate = min(95, base_occupancy * 1.8)
        else:
            occupancy_rate = min(85, base_occupancy * 1.5)
    else:
        # Calculate distance to Kızılay for occupancy adjustment
        if '15 Temmuz Kızılay Millî İrade' in line_stations:
            kizilay_idx = line_stations.index('15 Temmuz Kızılay Millî İrade')
            station_idx = line_stations.index(station)
            distance = abs(station_idx - kizilay_idx)
            
            # Stronger decay for stations near Kızılay
            if distance <= 2:
                max_occupancy = 75  # Cap nearby stations at 75%
            else:
                max_occupancy = 70  # Cap other stations at 70%
            
            if 7 <= hour <= 9 or 16 <= hour <= 19:
                occupancy_rate = min(max_occupancy, base_occupancy * (0.9 ** distance))
            else:
                occupancy_rate = min(max_occupancy - 10, base_occupancy * (0.85 ** distance))
        else:
            # For stations on lines without Kızılay
            if 7 <= hour <= 9 or 16 <= hour <= 19:
                occupancy_rate = min(70, base_occupancy)
            else:
                occupancy_rate = min(60, base_occupancy)
    
    return max(5, occupancy_rate)  # Ensure minimum 5% occupancy during operational hours 