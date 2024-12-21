"""
Rankings visualizations for metro data
"""

import os
import matplotlib.pyplot as plt
from .styles import MODERN_COLORS

def plot_station_rankings(df, out_dir='out'):
    """Create station ranking visualizations"""
    # Average occupancy rankings
    plt.figure(figsize=(15, 10))
    station_avg = df.groupby('Station_ID')['Occupancy_Rate'].mean().sort_values(ascending=True)
    station_avg.plot(kind='barh', color=MODERN_COLORS['primary'])
    plt.title('Stations Ranked by Average Occupancy', pad=20, fontsize=16, fontweight='bold')
    plt.xlabel('Average Occupancy Rate (%)', labelpad=10)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'station_rankings.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Peak hour rankings
    plt.figure(figsize=(15, 10))
    peak_avg = df[df['Time_Period'] == 'peak'].groupby('Station_ID')['Occupancy_Rate'].mean().sort_values(ascending=True)
    peak_avg.plot(kind='barh', color=MODERN_COLORS['secondary'])
    plt.title('Stations Ranked by Peak Hour Occupancy', pad=20, fontsize=16, fontweight='bold')
    plt.xlabel('Average Peak Hour Occupancy Rate (%)', labelpad=10)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'peak_hour_rankings.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_line_rankings(df, out_dir='out'):
    """Create line ranking visualizations"""
    # Average line occupancy
    plt.figure(figsize=(12, 6))
    line_avg = df.groupby('Metro_Line')['Occupancy_Rate'].mean().sort_values(ascending=True)
    line_avg.plot(kind='barh', color=MODERN_COLORS['accent1'])
    plt.title('Metro Lines Ranked by Average Occupancy', pad=20, fontsize=16, fontweight='bold')
    plt.xlabel('Average Occupancy Rate (%)', labelpad=10)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'line_rankings.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Peak vs Off-peak comparison
    plt.figure(figsize=(12, 6))
    peak_comparison = df.pivot_table(
        values='Occupancy_Rate',
        index='Metro_Line',
        columns='Time_Period',
        aggfunc='mean'
    ).sort_values('peak', ascending=True)
    
    peak_comparison.plot(kind='barh')
    plt.title('Metro Lines: Peak vs Off-Peak Occupancy', pad=20, fontsize=16, fontweight='bold')
    plt.xlabel('Average Occupancy Rate (%)', labelpad=10)
    plt.grid(True, alpha=0.2)
    plt.legend(title='Time Period')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'line_peak_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close() 