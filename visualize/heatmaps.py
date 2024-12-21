"""
Heatmap visualizations for metro data
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from .styles import MODERN_COLORS

def plot_station_heatmap(df, out_dir='out'):
    """Create heatmap of station occupancy throughout the day"""
    # Weekday heatmap
    plt.figure(figsize=(20, 12))
    weekday_data = df[~df['Is_Weekend']].pivot_table(
        values='Occupancy_Rate',
        index='Station_ID',
        columns='Hour',
        aggfunc='mean'
    )
    sns.heatmap(weekday_data, cmap='YlOrRd', 
               center=50, vmin=0, vmax=100,
               cbar_kws={'label': 'Average Occupancy Rate (%)'})
    plt.title('Weekday Station Occupancy Patterns')
    plt.xlabel('Hour of Day')
    plt.ylabel('Station')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'weekday_heatmap.png'))
    plt.close()

    # Weekend heatmap
    plt.figure(figsize=(20, 12))
    weekend_data = df[df['Is_Weekend']].pivot_table(
        values='Occupancy_Rate',
        index='Station_ID',
        columns='Hour',
        aggfunc='mean'
    )
    sns.heatmap(weekend_data, cmap='YlOrRd', 
               center=50, vmin=0, vmax=100,
               cbar_kws={'label': 'Average Occupancy Rate (%)'})
    plt.title('Weekend Station Occupancy Patterns')
    plt.xlabel('Hour of Day')
    plt.ylabel('Station')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'weekend_heatmap.png'))
    plt.close()

def plot_station_correlations(df, out_dir='out'):
    """Analyze correlations between stations"""
    # Create pivot table for station correlations
    station_pivot = df.pivot_table(
        values='Occupancy_Rate',
        index='Timestamp',
        columns='Station_ID',
        aggfunc='mean'
    )
    
    # Calculate correlation matrix
    corr_matrix = station_pivot.corr()
    
    # Create static heatmap
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr_matrix, cmap='RdYlBu_r', center=0,
               annot=False, fmt='.2f', square=True)
    plt.title('Station Occupancy Correlations')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'station_correlations.png'))
    plt.close()
    
    # Create interactive version
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0
    ))
    
    fig.update_layout(
        title='Station Occupancy Correlations (Interactive)',
        width=1500,
        height=1500
    )
    
    fig.write_html(os.path.join(out_dir, 'station_correlations.html')) 