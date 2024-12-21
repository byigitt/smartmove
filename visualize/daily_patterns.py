"""
Daily pattern visualizations for metro data
"""

import os
import matplotlib.pyplot as plt
from .styles import MODERN_COLORS

def plot_daily_patterns(df, generator, out_dir='out'):
    """Plot average daily passenger patterns with modern design"""
    plt.figure(figsize=(15, 8))
    
    if 'Station_Type' not in df.columns:
        df['Station_Type'] = df['Station_ID'].map(
            {station: 'terminal' if station in sum(generator.terminal_stations.values(), []) else
             'transfer_hub' if any('transfer_hub' in generator.station_features.get(station, []) for station in df['Station_ID'].unique()) else
             'regular'
             for station in df['Station_ID'].unique()}
        )
    
    hourly_avg = df.groupby(['Hour', 'Station_Type'])['Occupancy_Rate'].mean().unstack()
    
    # Use color dictionary
    type_colors = {
        'regular': MODERN_COLORS['primary'],
        'terminal': MODERN_COLORS['secondary'],
        'transfer_hub': MODERN_COLORS['accent1']
    }
    
    for station_type in hourly_avg.columns:
        plt.plot(hourly_avg.index, hourly_avg[station_type], 
                label=station_type.title(), 
                color=type_colors.get(station_type, MODERN_COLORS['primary']),
                marker='o',
                markersize=6,
                linewidth=2.5)
    
    plt.title('Daily Occupancy Patterns by Station Type', 
             pad=20, 
             fontsize=16, 
             fontweight='bold')
    plt.xlabel('Hour of Day', labelpad=10)
    plt.ylabel('Average Occupancy Rate (%)', labelpad=10)
    plt.legend(bbox_to_anchor=(1.05, 1), 
              loc='upper left',
              frameon=True,
              facecolor=MODERN_COLORS['background'],
              edgecolor=MODERN_COLORS['grid'])
    plt.grid(True, alpha=0.2)
    
    # Add subtle background shading for time periods
    plt.axvspan(6, 10, alpha=0.1, color=MODERN_COLORS['accent1'], label='Morning Peak')
    plt.axvspan(16, 20, alpha=0.1, color=MODERN_COLORS['accent1'], label='Evening Peak')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'daily_patterns.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_weekday_weekend_comparison(df, out_dir='out'):
    """Plot weekday vs weekend comparison"""
    plt.figure(figsize=(15, 8))
    weekday_avg = df[~df['Is_Weekend']].groupby('Hour')['Occupancy_Rate'].mean()
    weekend_avg = df[df['Is_Weekend']].groupby('Hour')['Occupancy_Rate'].mean()
    
    plt.plot(weekday_avg.index, weekday_avg, 
            label='Weekday', 
            color=MODERN_COLORS['primary'],
            marker='o',
            linewidth=2.5,
            markersize=6)
    plt.plot(weekend_avg.index, weekend_avg, 
            label='Weekend', 
            color=MODERN_COLORS['secondary'],
            marker='o',
            linewidth=2.5,
            markersize=6)
    
    plt.title('Weekday vs Weekend Occupancy Patterns', 
             pad=20, 
             fontsize=16, 
             fontweight='bold')
    plt.xlabel('Hour of Day', labelpad=10)
    plt.ylabel('Average Occupancy Rate (%)', labelpad=10)
    plt.legend(frameon=True,
              facecolor=MODERN_COLORS['background'],
              edgecolor=MODERN_COLORS['grid'])
    plt.grid(True, alpha=0.2)
    
    # Add subtle background shading
    plt.axvspan(6, 10, alpha=0.1, color=MODERN_COLORS['accent1'])
    plt.axvspan(16, 20, alpha=0.1, color=MODERN_COLORS['accent1'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'weekday_weekend_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close() 