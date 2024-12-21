"""
Special events and weather patterns affecting passenger flow
"""

from datetime import datetime

class EventPatterns:
    def __init__(self):
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
    
    def is_ramadan_time(self, date):
        """Check if the given date falls within Ramadan"""
        # Add actual Ramadan date checking logic
        return False
    
    def is_iftar_time(self, hour):
        """Check if it's iftar time"""
        # Simplified example - would need actual sunset times
        return hour in [19, 20]
    
    def is_post_iftar_time(self, hour):
        """Check if it's post-iftar time"""
        return hour in [20, 21, 22]
    
    def is_match_day(self, date):
        """Check if there's a football match on the given date"""
        # Add actual match schedule checking logic
        return False
    
    def is_pre_match_time(self, hour):
        """Check if it's pre-match time"""
        return hour in [16, 17, 18]
    
    def is_post_match_time(self, hour):
        """Check if it's post-match time"""
        return hour in [20, 21, 22]
    
    def get_event_factor(self, station, date, hour):
        """Calculate combined event factor for a station at a specific time"""
        event_factor = 1.0
        
        # Check national holidays
        date_str = date.strftime('%Y-%m-%d')
        if (date_str in self.special_events['national_holidays']['dates'] and 
            station in self.special_events['national_holidays']['affected_stations']):
            event_factor *= self.special_events['national_holidays']['crowd_factor']
        
        # Check Ramadan effects
        if (self.is_ramadan_time(date) and 
            station in self.special_events['ramadan']['affected_stations']):
            if self.is_iftar_time(hour):
                event_factor *= self.special_events['ramadan']['iftar_impact']
            elif self.is_post_iftar_time(hour):
                event_factor *= self.special_events['ramadan']['post_iftar_impact']
        
        # Check football matches
        if (self.is_match_day(date) and 
            station in self.special_events['football_matches']['affected_stations']):
            if self.is_pre_match_time(hour):
                event_factor *= self.special_events['football_matches']['crowd_factor']
            elif self.is_post_match_time(hour):
                event_factor *= self.special_events['football_matches']['crowd_factor'] * 0.8
        
        return event_factor
    
    def get_weather_factor(self, weather_condition):
        """Get the weather impact factor and disruption probability"""
        return self.weather_factors.get(weather_condition, {
            'factor': 1.0,
            'disruption_prob': 0.0
        }) 