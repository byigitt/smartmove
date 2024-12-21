"""
Impact analysis visualizations for metro data
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from .styles import MODERN_COLORS

def plot_weather_impact(df, out_dir='out'):
    """Visualize weather impact on occupancy rates"""
    # Overall weather impact
    weather_impact = df.groupby(['Weather_Condition', 'Time_Period'])['Occupancy_Rate'].mean().unstack()
    plt.figure(figsize=(12, 6))
    weather_impact.plot(kind='bar', width=0.8)
    plt.title('Weather Impact on Occupancy Rates')
    plt.xlabel('Weather Condition')
    plt.ylabel('Average Occupancy Rate (%)')
    plt.legend(title='Time Period')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'weather_impact.png'))
    plt.close()

    # Weather impact by line
    plt.figure(figsize=(15, 8))
    line_weather = df.groupby(['Metro_Line', 'Weather_Condition'])['Occupancy_Rate'].mean().unstack()
    line_weather.plot(kind='bar', width=0.8)
    plt.title('Weather Impact by Metro Line')
    plt.xlabel('Metro Line')
    plt.ylabel('Average Occupancy Rate (%)')
    plt.legend(title='Weather Condition')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'weather_impact_by_line.png'))
    plt.close()

def plot_special_events_impact(df, out_dir='out'):
    """Analyze and visualize impact of special events"""
    special_stations = ['Anıtkabir', 'Kızılay', 'Sıhhiye', 'Ulus']
    special_data = df[df['Station_ID'].isin(special_stations)]
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=special_data, x='Station_ID', y='Occupancy_Rate', 
               hue='Time_Period')
    plt.title('Occupancy Distribution at Key Stations During Different Periods')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'special_events_impact.png'))
    plt.close()

def plot_transfer_analysis(df, out_dir='out'):
    """Analyze transfer patterns at major stations"""
    transfer_stations = ['15 Temmuz Kızılay Millî İrade', 'Atatürk Kültür Merkezi', 'Batıkent']
    transfer_data = df[df['Station_ID'].isin(transfer_stations)]

    plt.figure(figsize=(15, 8))
    for station in transfer_stations:
        station_data = transfer_data[transfer_data['Station_ID'] == station]
        weekday_data = station_data[~station_data['Is_Weekend']].groupby('Hour')['Transfer_Out'].mean()
        plt.plot(weekday_data.index, weekday_data, label=station, marker='o')
    
    plt.title('Transfer Patterns at Major Transfer Stations (Weekday)')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Transfer Passengers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'transfer_patterns.png'))
    plt.close()

def plot_capacity_utilization(df, out_dir='out'):
    """Analyze capacity utilization patterns"""
    plt.figure(figsize=(15, 8))
    line_capacity = df.groupby(['Hour', 'Metro_Line'])['Capacity_Utilization'].mean().unstack()
    line_capacity.plot(marker='o')
    plt.title('Daily Capacity Utilization by Metro Line')
    plt.xlabel('Hour of Day')
    plt.ylabel('Capacity Utilization Rate')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Metro Line')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'capacity_utilization.png'))
    plt.close() 