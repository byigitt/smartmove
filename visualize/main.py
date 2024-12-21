"""
Main visualization module for Ankara Metro data
"""

import os
import pandas as pd
from data_generator import AnkaraMetroGenerator
from .styles import set_matplotlib_style
from .daily_patterns import plot_daily_patterns, plot_weekday_weekend_comparison
from .network_graph import create_station_network_graph
from .line_analysis import (
    plot_line_comparison, 
    plot_line_peak_analysis, 
    plot_hourly_line_analysis
)
from .heatmaps import plot_station_heatmap, plot_station_correlations
from .impact_analysis import (
    plot_weather_impact,
    plot_special_events_impact,
    plot_transfer_analysis,
    plot_capacity_utilization
)
from .rankings import plot_station_rankings, plot_line_rankings

# Set default matplotlib style
set_matplotlib_style()

class MetroDataVisualizer:
    def __init__(self, data_file='ankara_metro_crowding_data_realistic.csv', out_dir='out'):
        """Initialize visualizer with data file and output directory"""
        self.out_dir = os.path.abspath(out_dir)
        os.makedirs(self.out_dir, exist_ok=True)
        
        # Load or generate data
        if os.path.exists(data_file):
            self.df = pd.read_csv(data_file)
            self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])
            self.df['Hour'] = self.df['Timestamp'].dt.hour
        else:
            print("Generating new data...")
            self.generator = AnkaraMetroGenerator()
            self.df = self.generator.generate_passenger_flow(num_days=7)
            self.df.to_csv(data_file, index=False)
            print("Data generated and saved.")
        
        self.generator = AnkaraMetroGenerator()

    def generate_all_visualizations(self):
        """Generate all visualizations"""
        print("Generating visualizations...")
        
        print("1. Creating daily patterns plot...")
        plot_daily_patterns(self.df, self.generator, self.out_dir)
        plot_weekday_weekend_comparison(self.df, self.out_dir)
        
        print("2. Creating station heatmap...")
        plot_station_heatmap(self.df, self.out_dir)
        
        print("3. Creating line comparison plot...")
        plot_line_comparison(self.df, self.out_dir)
        plot_line_peak_analysis(self.df, self.out_dir)
        
        print("4. Creating weather impact plot...")
        plot_weather_impact(self.df, self.out_dir)
        
        print("5. Creating network graph...")
        create_station_network_graph(self.df, self.generator, self.out_dir)
        
        print("6. Creating special events impact plot...")
        plot_special_events_impact(self.df, self.out_dir)
        
        print("7. Creating station rankings...")
        plot_station_rankings(self.df, self.out_dir)
        plot_line_rankings(self.df, self.out_dir)
        
        print("8. Creating transfer analysis...")
        plot_transfer_analysis(self.df, self.out_dir)
        
        print("9. Creating capacity utilization analysis...")
        plot_capacity_utilization(self.df, self.out_dir)
        
        print("10. Creating hourly line analysis...")
        plot_hourly_line_analysis(self.df, self.out_dir)
        
        print("11. Creating station correlations...")
        plot_station_correlations(self.df, self.out_dir)
        
        print("Visualizations completed! Check the output files in the 'out' folder.") 