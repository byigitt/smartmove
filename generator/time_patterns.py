"""
Time-related patterns and distributions for passenger flow
"""

import numpy as np
from scipy.stats import norm

class TimePatterns:
    def __init__(self):
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
    
    def get_time_factor(self, hour, is_weekend):
        """Get the time factor for a specific hour"""
        day_type = 'weekend' if is_weekend else 'weekday'
        return self.time_factors[day_type].get(hour, 0.0)
    
    def get_monthly_factor(self, month):
        """Get the seasonality factor for a specific month"""
        return self.monthly_factors.get(month, 1.0)
    
    def is_peak_hour(self, hour, is_weekend):
        """Determine if the given hour is a peak hour"""
        if is_weekend:
            # Generally less pronounced peaks on weekends
            return False
        return (7 <= hour <= 9) or (16 <= hour <= 19)
    
    def get_time_period(self, hour):
        """Get the time period for a specific hour"""
        if hour >= 23 or hour <= 5:
            return 'off_peak'
        elif self.is_peak_hour(hour, False):  # Using weekday definition
            return 'peak'
        else:
            return 'regular' 