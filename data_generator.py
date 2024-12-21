"""
Generate synthetic metro passenger data for Ankara Metro system with realistic local patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm

class AnkaraMetroGenerator:
    def __init__(self):
        self._initialize_metro_lines()
        self.train_capacity = 1200  # Updated capacity for Ankara Metro trains
        
    def _initialize_metro_lines(self):
        """Initialize metro lines with their stations and characteristics"""
        # Define special locations near stations that affect passenger flow
        self.station_features = {
            # M1-2-3 Line Stations (Koru - OSB-Törekent)
            'Koru': ['residential_high_income', 'terminal', 'park_and_ride'],
            'Çayyolu': ['residential_high_income', 'shopping_district', 'weekend_active'],
            'Ümitköy': ['residential_high_income', 'shopping_district'],
            'Beytepe': ['university_hacettepe', 'education_zone', 'student_residential'],
            'Tarım Bakanlığı-Danıştay': ['government_offices', 'office_district'],
            'Bilkent': ['university_bilkent', 'education_zone', 'student_residential', 'shopping_mall'],
            'Orta Doğu Teknik Üniversitesi': ['university_metu', 'education_zone', 'student_residential', 'research_center'],
            'Maden Tetkik ve Arama': ['government_offices', 'research_center'],
            'Söğütözü': ['business_district', 'shopping_mall', 'office_district'],
            'Millî Kütüphane': ['education_zone', 'cultural_center', 'government_offices'],
            'Necatibey': ['business_district', 'office_district'],
            '15 Temmuz Kızılay Millî İrade': ['central_hub', 'shopping_district', 'business_district', 'transfer_hub', 
                                             'entertainment_district', 'restaurant_district', 'youth_center'],
            'Sıhhiye': ['hospital_zone', 'education_zone', 'government_offices', 'shopping_district'],
            'Ulus': ['historic_center', 'shopping_district', 'traditional_market', 'tourist_attraction'],
            'Atatürk Kültür Merkezi': ['cultural_center', 'transfer_hub', 'government_offices', 'museum_district'],
            'Akköprü': ['shopping_mall', 'residential'],
            'İvedik': ['industrial_zone', 'business_park', 'manufacturing'],
            'Yenimahalle': ['residential_mixed', 'local_shopping'],
            'Demetevler': ['residential_high_density', 'local_market'],
            'Hastane': ['hospital_zone', 'medical_center'],
            'Macunköy': ['industrial_zone', 'manufacturing'],
            'Orta Doğu Sanayi ve Ticaret Merkezi': ['industrial_zone', 'business_park', 'wholesale_market'],
            'Batıkent': ['residential_mixed', 'shopping_district', 'transfer_hub'],
            'Batı Merkez': ['residential', 'local_shopping'],
            'Mesa': ['residential_planned', 'park'],
            'Botanik': ['park', 'recreation_area', 'residential'],
            'İstanbul Yolu': ['residential', 'industrial_mixed'],
            'Eryaman 1-2': ['residential_planned', 'local_shopping'],
            'Eryaman 5': ['residential_planned', 'education_zone'],
            'Devlet Mahallesi/1910 Ankaragücü': ['sports_complex', 'stadium', 'residential'],
            'Harikalar Diyarı': ['park', 'recreation_area', 'family_entertainment'],
            'Fatih': ['residential_mixed', 'local_market'],
            'Gaziosmanpaşa': ['residential_high_income', 'embassy_district'],
            'OSB-Törekent': ['industrial_zone', 'terminal', 'manufacturing_hub'],

            # M4 Line Stations
            'Adliye': ['government_offices', 'courthouse', 'office_district'],
            'Gar': ['transfer_hub', 'historic_station', 'transport_hub'],
            'Ankara Su ve Kanalizasyon İdaresi': ['government_offices', 'utility_services'],
            'Dışkapı': ['hospital_zone', 'medical_center', 'education_zone'],
            'Meteoroloji': ['government_offices', 'research_center'],
            'Belediye': ['government_offices', 'civic_center'],
            'Mecidiye': ['residential_mixed', 'local_shopping'],
            'Kuyubaşı': ['residential', 'local_market'],
            'Dutluk': ['residential', 'park'],
            'Şehitler': ['residential', 'terminal'],

            # A1 Line Stations
            'AŞTİ': ['transport_hub', 'terminal', 'shopping_center'],
            'Emek': ['residential_mixed', 'office_district'],
            'Bahçelievler': ['residential_high_income', 'shopping_district', 'education_zone', 'restaurant_district'],
            'Beşevler': ['university_area', 'education_zone', 'student_residential'],
            'Anadolu/Anıtkabir': ['historic_monument', 'tourist_attraction', 'cultural_center'],
            'Maltepe': ['residential_mixed', 'business_district', 'shopping_district'],
            'Demirtepe': ['business_district', 'office_district'],
            'Kolej': ['education_zone', 'student_area'],
            'Kurtuluş': ['residential_mixed', 'local_shopping'],
            'Dikimevi': ['hospital_zone', 'education_zone', 'terminal']
        }

        # Define special time periods for different location types
        self.location_time_patterns = {
            'university_metu': {
                'active_months': [2,3,4,5,10,11,12],  # Academic months
                'exam_months': [1,6],  # Exam periods
                'peak_hours': [8,9,12,13,16,17],  # Class change hours
                'weekend_factor': 0.2
            },
            'university_bilkent': {
                'active_months': [2,3,4,5,10,11,12],
                'exam_months': [1,6],
                'peak_hours': [8,9,12,13,16,17],
                'weekend_factor': 0.2
            },
            'university_hacettepe': {
                'active_months': [2,3,4,5,10,11,12],
                'exam_months': [1,6],
                'peak_hours': [8,9,12,13,16,17],
                'weekend_factor': 0.2
            },
            'government_offices': {
                'peak_hours': [8,9,12,13,17,18],
                'active_days': [0,1,2,3,4],  # Monday to Friday
                'weekend_factor': 0.1
            },
            'shopping_district': {
                'peak_hours': [12,13,14,17,18,19,20],
                'weekend_peak_hours': [13,14,15,16,17,18,19],
                'weekend_factor': 1.2
            },
            'hospital_zone': {
                'peak_hours': [8,9,10,14,15,16],
                'weekend_factor': 0.7,
                'visiting_hours': [14,15,16,17,18,19]
            },
            'industrial_zone': {
                'shift_changes': [6,14,22],  # Shift change hours
                'peak_factor': 1.5,
                'weekend_factor': 0.4
            },
            'business_district': {
                'peak_hours': [8,9,12,13,17,18,19],
                'weekend_factor': 0.3
            },
            'residential_high_income': {
                'morning_peak': [7,8,9],
                'evening_peak': [17,18,19],
                'weekend_factor': 0.8
            },
            'residential_high_density': {
                'morning_peak': [6,7,8],
                'evening_peak': [16,17,18],
                'weekend_factor': 0.9
            },
            'sports_complex': {
                'match_day_factor': 2.0,
                'weekend_factor': 1.5,
                'event_hours': [18,19,20,21]
            },
            'tourist_attraction': {
                'peak_hours': [10,11,12,13,14,15,16],
                'weekend_factor': 1.4,
                'seasonal_peaks': [4,5,6,7,8,9]  # Spring and Summer months
            }
        }

        # Define special events and their impact
        self.special_events = {
            'national_holidays': {
                'dates': ['2024-04-23', '2024-05-19', '2024-10-29'],  # National holidays
                'affected_stations': ['Anıtkabir', 'Kızılay', 'Sıhhiye', 'Ulus'],
                'crowd_factor': 1.8
            },
            'ramadan': {
                'iftar_impact': 0.7,  # Reduced traffic during iftar
                'post_iftar_impact': 1.3,  # Increased traffic after iftar
                'affected_stations': ['Kızılay', 'Ulus', 'Bahçelievler']
            },
            'football_matches': {
                'affected_stations': ['Devlet Mahallesi/1910 Ankaragücü'],
                'pre_match_hours': 3,
                'post_match_hours': 2,
                'crowd_factor': 2.0
            }
        }
        
        self.metro_lines = {
            'A1': {'stations': ['AŞTİ', 'Emek', 'Bahçelievler', 'Beşevler', 'Anadolu/Anıtkabir', 'Maltepe', 'Demirtepe', '15 Temmuz Kızılay Millî İrade', 'Kolej', 'Kurtuluş', 'Dikimevi'],
                   'terminal_stations': ['AŞTİ', 'Dikimevi'],
                   'junction_stations': ['15 Temmuz Kızılay Millî İrade'],
                   'frequency_minutes': {
                       'peak': 3,
                       'regular': 5,
                       'off_peak': 10
                   }},
            'M1-2-3': {'stations': ['Koru', 'Çayyolu', 'Ümitköy', 'Beytepe', 'Tarım Bakanlığı-Danıştay', 'Bilkent', 'Orta Doğu Teknik Üniversitesi', 'Maden Tetkik ve Arama', 'Söğütözü', 'Millî Kütüphane', 'Necatibey', '15 Temmuz Kızılay Millî İrade', 'Sıhhiye', 'Ulus', 'Atatürk Kültür Merkezi', 'Akköprü', 'İvedik', 'Yenimahalle', 'Demetevler', 'Hastane', 'Macunköy', 'Orta Doğu Sanayi ve Ticaret Merkezi', 'Batıkent', 'Batı Merkez', 'Mesa', 'Botanik', 'İstanbul Yolu', 'Eryaman 1-2', 'Eryaman 5', 'Devlet Mahallesi/1910 Ankaragücü', 'Harikalar Diyarı', 'Fatih', 'Gaziosmanpaşa', 'OSB-Törekent'],
                      'terminal_stations': ['Koru', 'OSB-Törekent'],
                      'junction_stations': ['15 Temmuz Kızılay Millî İrade', 'Batıkent'],
                      'frequency_minutes': {
                          'peak': 3,
                          'regular': 5,
                          'off_peak': 10
                      }},
            'M4': {'stations': ['15 Temmuz Kızılay Millî İrade', 'Adliye', 'Gar', 'Atatürk Kültür Merkezi', 'Ankara Su ve Kanalizasyon İdaresi', 'Dışkapı', 'Meteoroloji', 'Belediye', 'Mecidiye', 'Kuyubaşı', 'Dutluk', 'Şehitler'],
                   'terminal_stations': ['15 Temmuz Kızılay Millî İrade', 'Şehitler'],
                   'junction_stations': ['15 Temmuz Kızılay Millî İrade', 'Atatürk Kültür Merkezi'],
                   'frequency_minutes': {
                       'peak': 4,
                       'regular': 7,
                       'off_peak': 15
                   }}
        }
        
        # Define peak hours with more granular time periods
        self.time_periods = {
            'early_morning': (6, 7),
            'morning_peak': (7, 9),
            'late_morning': (9, 11),
            'midday': (11, 16),
            'evening_peak': (16, 19),
            'late_evening': (19, 23),
            'night': (23, 6)
        }

        # Monthly seasonality factors (1.0 is baseline)
        self.monthly_factors = {
            1: 0.90,   # January (slightly lower due to winter holidays)
            2: 0.95,
            3: 1.00,
            4: 1.05,
            5: 1.10,
            6: 1.00,
            7: 0.90,
            8: 0.90,
            9: 1.10,
            10: 1.05,
            11: 1.00,
            12: 0.95
        }
        
        # Weather impact factors with service disruption probabilities
        self.weather_factors = {
            'Sunny': {
                'factor': 0.9,
                'disruption_prob': 0.0
            },
            'Cloudy': {
                'factor': 1.0,
                'disruption_prob': 0.0
            },
            'Rainy': {
                'factor': 1.1,
                'disruption_prob': 0.02
            },
            'Snowy': {
                'factor': 1.2,
                'disruption_prob': 0.10
            },
            'Stormy': {
                'factor': 1.25,
                'disruption_prob': 0.15
            }
        }
        
        # Create time factors using normal distributions
        self._create_time_distributions()
    
    def _create_time_distributions(self):
        """Create more realistic time distributions with better weekend patterns"""
        hours = np.arange(24)

        # Morning peak (centered at 8)
        morning_peak = norm.pdf(hours, loc=8, scale=1.0)
        # Evening peak (centered at 17)
        evening_peak = norm.pdf(hours, loc=17, scale=1.0)
        # Midday moderate activity
        midday = norm.pdf(hours, loc=13, scale=2.0)
        # Late evening activity (centered at 21, wider spread)
        late_evening = norm.pdf(hours, loc=21, scale=2.5) * 0.6

        # Combine distributions for weekdays
        weekday_dist = (morning_peak * 1.2 + evening_peak * 1.2 + midday * 0.8 + late_evening)
        weekday_dist = weekday_dist / weekday_dist.max() * 5.0

        # Weekend distribution: later start, higher midday activity
        weekend_midday = norm.pdf(hours, loc=14, scale=4.0) * 1.2  # Wider spread, higher amplitude
        weekend_evening = norm.pdf(hours, loc=20, scale=3.0) * 0.9  # Later evening peak
        weekend_dist = (weekend_midday + weekend_evening)
        weekend_dist = weekend_dist / weekend_dist.max() * 4.0  # Higher maximum for weekends

        # Ensure zero occupancy during non-operational hours
        weekday_dist = np.where((hours >= 6) & (hours <= 23), weekday_dist, 0.0)
        weekend_dist = np.where((hours >= 6) & (hours <= 23), weekend_dist, 0.0)

        # Add gradual ramp-up in the early morning (6:00-8:00)
        early_morning_ramp = np.zeros(24)
        early_morning_ramp[6:8] = np.linspace(0.1, 0.5, 2)
        weekday_dist = np.maximum(weekday_dist, early_morning_ramp)
        weekend_dist = np.maximum(weekend_dist, early_morning_ramp * 0.8)  # Slightly lower early morning on weekends

        # Ensure minimum activity levels during operational hours (after ramp-up)
        weekday_dist = np.where((hours >= 8) & (hours <= 23), 
                               np.maximum(weekday_dist, 0.3),
                               weekday_dist)
        weekend_dist = np.where((hours >= 8) & (hours <= 23),
                               np.maximum(weekend_dist, 0.4),  # Higher minimum for weekends
                               weekend_dist)

        # Add random variations to make it more realistic
        np.random.seed(42)  # For reproducibility
        variation = np.random.normal(0, 0.05, 24)
        weekday_dist += variation
        weekend_dist += variation

        # Create time factors dictionary with proper early morning handling
        self.time_factors = {
            'weekday': {hour: (0.0 if hour < 6 or hour > 23 else
                             max(0.1, factor) if hour < 8 else  # Early morning minimum
                             max(0.2, factor))  # Regular minimum
                       for hour, factor in enumerate(weekday_dist)},
            'weekend': {hour: (0.0 if hour < 6 or hour > 23 else
                             max(0.1, factor) if hour < 8 else  # Early morning minimum
                             max(0.3, factor))  # Higher weekend minimum
                       for hour, factor in enumerate(weekend_dist)}
        }

        # Ensure proper early morning ramp-up for both weekday and weekend
        for day_type in ['weekday', 'weekend']:
            self.time_factors[day_type][6] = 0.1  # Start at 10% at 6:00
            self.time_factors[day_type][7] = 0.3  # Ramp up to 30% by 7:00
    
    def _get_station_popularity(self, station, hour, is_weekend):
        """Calculate station popularity with enhanced distance-based patterns"""
        base_popularity = 1.0
        features = self.station_features.get(station, [])
        
        # Special handling for Kızılay
        if station == '15 Temmuz Kızılay Millî İrade':
            base_popularity = 1.4  # Ensure Kızılay has highest base popularity
            if is_weekend:
                base_popularity *= 1.3  # Even higher on weekends due to shopping/entertainment
            return base_popularity
        
        # Find the line and distance from Kızılay
        for line, details in self.metro_lines.items():
            if station in details['stations']:
                stations = details['stations']
                if '15 Temmuz Kızılay Millî İrade' in stations:
                    kizilay_idx = stations.index('15 Temmuz Kızılay Millî İrade')
                    station_idx = stations.index(station)
                    distance_from_kizilay = abs(station_idx - kizilay_idx) / len(stations)
                    
                    # Steeper decay for stations immediately adjacent to Kızılay
                    if abs(station_idx - kizilay_idx) <= 2:
                        distance_factor = np.exp(-2.5 * distance_from_kizilay)  # Steeper decay for nearby stations
                    else:
                        # Line-specific weekend adjustments
                        if is_weekend:
                            if line == 'M1-2-3':
                                distance_factor = np.exp(-1.2 * distance_from_kizilay) * 1.4
                            elif line == 'M4':
                                distance_factor = np.exp(-1.5 * distance_from_kizilay) * 0.8
                            else:  # A1 line
                                distance_factor = np.exp(-1.8 * distance_from_kizilay) * 0.6
                        else:
                            if line == 'M4':
                                distance_factor = np.exp(-1.5 * distance_from_kizilay) * 1.2
                            else:
                                distance_factor = np.exp(-2.0 * distance_from_kizilay)
                    
                    # Ensure adjacent stations never exceed 85% of Kızılay's popularity
                    if abs(station_idx - kizilay_idx) <= 2:
                        distance_factor = min(distance_factor, 0.85)
                    
                    base_popularity *= max(0.3, distance_factor)
                    
                    # Terminal station adjustments
                    if station in details['terminal_stations']:
                        if is_weekend:
                            if line == 'M1-2-3':
                                base_popularity *= 0.7
                            else:
                                base_popularity *= 0.5
                        else:
                            base_popularity *= 0.6
                break
        
        # Get the current date information
        current_date = datetime.now()
        current_month = current_date.month
        
        # Apply location-specific time patterns
        for feature in features:
            if feature in self.location_time_patterns:
                pattern = self.location_time_patterns[feature]
                
                # Apply weekend factors
                if is_weekend:
                    base_popularity *= pattern.get('weekend_factor', 1.0)
                
                # Apply peak hour factors
                if hour in pattern.get('peak_hours', []):
                    base_popularity *= 1.4
                
                # For universities, check academic calendar
                if feature.startswith('university_'):
                    if current_month in pattern['active_months']:
                        base_popularity *= 1.4
                    elif current_month in pattern['exam_months']:
                        base_popularity *= 1.6
                    else:  # Holiday months
                        base_popularity *= 0.3
                
                # For industrial zones, check shift changes
                if feature == 'industrial_zone' and hour in pattern['shift_changes']:
                    base_popularity *= pattern['peak_factor']
                
                # For shopping districts, apply different weekend patterns
                if feature == 'shopping_district' and is_weekend:
                    if hour in pattern['weekend_peak_hours']:
                        base_popularity *= 1.3
                
                # For hospital zones, consider visiting hours
                if feature == 'hospital_zone' and hour in pattern.get('visiting_hours', []):
                    base_popularity *= 1.2
                
                # For residential areas, apply appropriate peak hours
                if feature.startswith('residential_'):
                    if not is_weekend:
                        if hour in pattern.get('morning_peak', []):
                            base_popularity *= 1.3
                        elif hour in pattern.get('evening_peak', []):
                            base_popularity *= 1.4
                
                # For tourist attractions, consider seasonal peaks
                if feature == 'tourist_attraction' and current_month in pattern.get('seasonal_peaks', []):
                    base_popularity *= 1.3
        
        # Apply special event factors
        for event_type, event_info in self.special_events.items():
            if station in event_info.get('affected_stations', []):
                if event_type == 'national_holidays':
                    # Check if current date is a national holiday
                    current_date_str = current_date.strftime('%Y-%m-%d')
                    if current_date_str in event_info['dates']:
                        base_popularity *= event_info['crowd_factor']
                
                elif event_type == 'ramadan':
                    # Example: Apply Ramadan patterns (you would need to add actual Ramadan date checking)
                    if self._is_ramadan_time(current_date):
                        if self._is_iftar_time(hour):
                            base_popularity *= event_info['iftar_impact']
                        elif self._is_post_iftar_time(hour):
                            base_popularity *= event_info['post_iftar_impact']
                
                elif event_type == 'football_matches':
                    # Example: Apply match day patterns (you would need to add actual match schedule checking)
                    if self._is_match_day(current_date):
                        if self._is_pre_match_time(hour):
                            base_popularity *= event_info['crowd_factor']
                        elif self._is_post_match_time(hour):
                            base_popularity *= event_info['crowd_factor'] * 0.8
        
        return base_popularity
    
    def _is_ramadan_time(self, date):
        """Check if the given date falls within Ramadan"""
        # Add actual Ramadan date checking logic
        return False
    
    def _is_iftar_time(self, hour):
        """Check if it's iftar time"""
        # Simplified example - would need actual sunset times
        return hour in [19, 20]
    
    def _is_post_iftar_time(self, hour):
        """Check if it's post-iftar time"""
        return hour in [20, 21, 22]
    
    def _is_match_day(self, date):
        """Check if there's a football match on the given date"""
        # Add actual match schedule checking logic
        return False
    
    def _is_pre_match_time(self, hour):
        """Check if it's pre-match time"""
        return hour in [16, 17, 18]
    
    def _is_post_match_time(self, hour):
        """Check if it's post-match time"""
        return hour in [20, 21, 22]
    
    def _get_station_capacity(self, station):
        """Update station capacities"""
        if station == 'Kızılay':
            return 2000
        elif station in ['Bahçelievler', 'Milli Kütüphane']:
            return 1200  # Higher capacity for these busy stations
        elif station in sum([details['junction_stations'] for details in self.metro_lines.values()], []):
            return 1500
        else:
            return 800
    
    def _generate_station_characteristics(self):
        station_data = {}
        for line, details in self.metro_lines.items():
            stations = details['stations']
            terminal_stations = details['terminal_stations']
            junction_stations = details['junction_stations']

            for station in stations:
                if station not in station_data:
                    if station == 'Kızılay':
                        base_popularity = 1.0
                        capacity = 2000
                    elif station in ['Bahçelievler', 'Milli Kütüphane']:
                        base_popularity = 0.95  # Higher base popularity
                        capacity = 1200
                    elif station in junction_stations:
                        base_popularity = 0.9
                        capacity = 1500
                    else:
                        # Just a slight decay based on position
                        station_idx = stations.index(station)
                        center_idx = 0 if 'Kızılay' in stations else len(stations)//2
                        distance = abs(station_idx - center_idx)
                        base_popularity = 0.85 * np.exp(-0.03 * distance)
                        capacity = 800

                    station_data[station] = {
                        'base_popularity': base_popularity,
                        'type': 'central' if station == 'Kızılay' else
                               'high_traffic' if station in ['Bahçelievler', 'Milli Kütüphane'] else
                               'junction' if station in junction_stations else
                               'terminal' if station in terminal_stations else 'regular',
                        'capacity': capacity
                    }

        return station_data
    
    def _get_time_factors(self, hour, is_weekend):
        day_type = 'weekend' if is_weekend else 'weekday'
        return self.time_factors[day_type].get(hour, 0.0)

    def _calculate_transfer_passengers(self, station, hour, is_weekend, base_flow):
        # Less extreme transfer rates
        if station not in sum([details['junction_stations'] for details in self.metro_lines.values()], []):
            return 0
        if hour < 6 or hour >= 23:
            return 0

        if station == 'Kızılay':
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
    
    def _is_peak_hour(self, hour, is_weekend):
        if is_weekend:
            # Generally less pronounced peaks on weekends
            return False
        return (7 <= hour <= 9) or (16 <= hour <= 19)

    def _calculate_boarding_ratio(self, station, line, hour, is_weekend):
        """Calculate boarding vs alighting ratio based on station position and distance from Kızılay"""
        stations = self.metro_lines[line]['stations']
        station_idx = stations.index(station)
        total_stations = len(stations)
        
        # Find distance from Kızılay
        if '15 Temmuz Kızılay Millî İrade' in stations:
            kizilay_idx = stations.index('15 Temmuz Kızılay Millî İrade')
            # Calculate distance from Kızılay (0 to 1, where 1 is furthest)
            distance_from_kizilay = abs(station_idx - kizilay_idx) / (total_stations/2)
            
            # Terminal station check
            is_terminal = station in self.metro_lines[line]['terminal_stations']
            
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
            
            # Adjust for terminal stations
            if is_terminal:
                if station_idx == 0:  # First terminal
                    boarding_ratio = max(0.7, boarding_ratio) if 7 <= hour <= 9 else min(0.3, boarding_ratio)
                else:  # Last terminal
                    boarding_ratio = min(0.2, boarding_ratio) if 7 <= hour <= 9 else max(0.6, boarding_ratio)
        else:
            # For lines without Kızılay, use simpler distance-based logic
            mid_point = total_stations // 2
            relative_pos = (station_idx - mid_point) / total_stations
            boarding_ratio = 0.5 * (1 - abs(relative_pos))
        
        return max(0.1, min(0.8, boarding_ratio))

    def generate_passenger_flow(self, num_days=7, start_date=None):
        if start_date is None:
            start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        station_characteristics = self._generate_station_characteristics()
        data = []

        for day in range(num_days):
            current_date = start_date + timedelta(days=day)
            is_weekend = current_date.weekday() >= 5
            month_factor = self.monthly_factors[current_date.month]

            for hour in range(24):
                time_factor = self._get_time_factors(hour, is_weekend)
                if time_factor == 0:
                    continue

                weather = np.random.choice(list(self.weather_factors.keys()))
                weather_info = self.weather_factors[weather]

                if np.random.random() < weather_info['disruption_prob']:
                    weather_factor = weather_info['factor'] * 0.8
                else:
                    weather_factor = weather_info['factor']

                for line, details in self.metro_lines.items():
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

                    if self._is_peak_hour(hour, is_weekend):
                        frequency = details['frequency_minutes']['peak']
                    elif hour >= 23 or hour <= 5:
                        frequency = details['frequency_minutes']['off_peak']
                    else:
                        frequency = details['frequency_minutes']['regular']

                    trains_per_hour = max(1, 60 // frequency)

                    for station in details['stations']:
                        station_info = station_characteristics[station]
                        
                        # Calculate distance factor from Kızılay with line-specific adjustments
                        if '15 Temmuz Kızılay Millî İrade' in details['stations']:
                            kizilay_idx = details['stations'].index('15 Temmuz Kızılay Millî İrade')
                            station_idx = details['stations'].index(station)
                            distance_from_kizilay = abs(station_idx - kizilay_idx) / len(details['stations'])
                            
                            # Line-specific distance factors with weekend adjustments
                            if is_weekend:
                                if line == 'M1-2-3':
                                    distance_factor = max(0.4, np.exp(-1.5 * distance_from_kizilay))
                                elif line == 'M4':
                                    distance_factor = max(0.3, np.exp(-1.8 * distance_from_kizilay))
                                else:  # A1 line
                                    distance_factor = max(0.2, np.exp(-2.0 * distance_from_kizilay))
                            else:
                                if line == 'M4':
                                    distance_factor = max(0.4, np.exp(-1.8 * distance_from_kizilay))
                                else:
                                    distance_factor = max(0.3, np.exp(-2.2 * distance_from_kizilay))
                        else:
                            distance_factor = 0.8

                        station_popularity = self._get_station_popularity(station, hour, is_weekend)
                        
                        # Adjust base passenger generation with weekend line factors
                        if station == '15 Temmuz Kızılay Millî İrade':
                            base_pass = 100 * weekend_line_factor
                        elif line == 'M4':
                            base_pass = 80 * distance_factor * weekend_line_factor
                        else:
                            base_pass = 70 * distance_factor * weekend_line_factor

                        # Terminal stations adjustments
                        if station in details['terminal_stations']:
                            if is_weekend:
                                if line == 'M1-2-3':
                                    base_pass = max(30, base_pass * 0.6)  # Higher minimum for M1-2-3 terminals
                                else:
                                    base_pass = max(20, base_pass * 0.4)  # Lower for other terminals
                            else:
                                base_pass = max(25, base_pass * 0.5)

                        base_flow = int(
                            station_popularity *
                            time_factor *
                            weather_factor *
                            month_factor *
                            trains_per_hour *
                            np.random.normal(1, 0.1) *
                            base_pass
                        )

                        transfers = self._calculate_transfer_passengers(station, hour, is_weekend, base_flow)
                        boarding_ratio = self._calculate_boarding_ratio(station, line, hour, is_weekend)
                        
                        boarding = int(base_flow * boarding_ratio)
                        alighting = base_flow - boarding

                        # Calculate occupancy with enhanced Kızılay dominance
                        base_occupancy = (boarding + transfers) * 100 / station_info['capacity']
                        
                        if station == '15 Temmuz Kızılay Millî İrade':
                            # Ensure Kızılay maintains highest occupancy
                            if self._is_peak_hour(hour, is_weekend):
                                occupancy_rate = min(95, base_occupancy * 1.8 * weekend_line_factor)
                            else:
                                occupancy_rate = min(85, base_occupancy * 1.5 * weekend_line_factor)
                        else:
                            # Calculate distance to Kızılay for occupancy adjustment
                            if '15 Temmuz Kızılay Millî İrade' in details['stations']:
                                kizilay_idx = details['stations'].index('15 Temmuz Kızılay Millî İrade')
                                station_idx = details['stations'].index(station)
                                distance = abs(station_idx - kizilay_idx)
                                
                                # Stronger decay for stations near Kızılay
                                if distance <= 2:
                                    max_occupancy = 75  # Cap nearby stations at 75%
                                else:
                                    max_occupancy = 70  # Cap other stations at 70%
                                
                                if station in details['terminal_stations']:
                                    if is_weekend:
                                        if line == 'M1-2-3':
                                            occupancy_rate = min(50, max(20, base_occupancy * 0.7))
                                        else:
                                            occupancy_rate = min(30, max(10, base_occupancy * 0.5))
                                    else:
                                        if self._is_peak_hour(hour, is_weekend):
                                            occupancy_rate = min(45, max(15, base_occupancy * 0.6))
                                        else:
                                            occupancy_rate = min(35, max(10, base_occupancy * 0.5))
                                else:
                                    if line == 'M4':
                                        if self._is_peak_hour(hour, is_weekend):
                                            occupancy_rate = min(max_occupancy, max(20, base_occupancy * distance_factor * 1.3))
                                        else:
                                            occupancy_rate = min(max_occupancy - 10, max(15, base_occupancy * distance_factor * 1.1))
                                    else:
                                        if self._is_peak_hour(hour, is_weekend):
                                            occupancy_rate = min(max_occupancy, max(15, base_occupancy * distance_factor * 1.2))
                                        else:
                                            occupancy_rate = min(max_occupancy - 10, max(10, base_occupancy * distance_factor))
                            else:
                                # For stations on lines without Kızılay
                                if self._is_peak_hour(hour, is_weekend):
                                    occupancy_rate = min(70, base_occupancy * distance_factor * 1.2)
                                else:
                                    occupancy_rate = min(60, base_occupancy * distance_factor)

                        # Ensure Kızılay dominance by applying a final adjustment
                        if station != '15 Temmuz Kızılay Millî İrade' and '15 Temmuz Kızılay Millî İrade' in details['stations']:
                            kizilay_idx = details['stations'].index('15 Temmuz Kızılay Millî İrade')
                            station_idx = details['stations'].index(station)
                            if abs(station_idx - kizilay_idx) <= 2:
                                occupancy_rate = min(occupancy_rate, 75)  # Hard cap for adjacent stations

                        # Determine time period
                        if self._is_peak_hour(hour, is_weekend):
                            time_period = 'peak'
                        elif hour >= 23 or hour <= 5:
                            time_period = 'off_peak'
                        else:
                            time_period = 'regular'

                        data.append({
                            'Timestamp': current_date.replace(hour=hour),
                            'Metro_Line': line,
                            'Station_ID': station,
                            'Station_Type': station_info['type'],
                            'Weather_Condition': weather,
                            'Is_Weekend': is_weekend,
                            'Time_Period': time_period,
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

def main():
    """Generate synthetic data and save to CSV"""
    print("Generating synthetic metro passenger data...")
    generator = AnkaraMetroGenerator()
    df = generator.generate_passenger_flow(num_days=365)  # One month for example
    output_file = 'ankara_metro_crowding_data_realistic.csv'
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")
    print(f"Generated {len(df)} records for {len(df['Station_ID'].unique())} stations")

if __name__ == "__main__":
    main()