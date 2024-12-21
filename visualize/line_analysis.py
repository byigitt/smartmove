"""
Line analysis visualizations for metro data
"""

import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .styles import MODERN_COLORS, modern_template

def plot_line_comparison(df, out_dir='out'):
    """Compare occupancy patterns between different metro lines"""
    fig = make_subplots(rows=2, cols=1,
                      subplot_titles=('Weekday Patterns', 'Weekend Patterns'))
    
    for is_weekend in [False, True]:
        weekend_data = df[df['Is_Weekend'] == is_weekend]
        line_patterns = weekend_data.groupby(['Hour', 'Metro_Line'])['Occupancy_Rate'].mean().unstack()
        
        for line in line_patterns.columns:
            fig.add_trace(
                go.Scatter(x=line_patterns.index, y=line_patterns[line],
                         name=f"{line} ({'Weekend' if is_weekend else 'Weekday'})",
                         mode='lines+markers'),
                row=(2 if is_weekend else 1), col=1
            )
    
    fig.update_layout(height=800, title_text="Metro Line Comparison: Weekday vs Weekend",
                     showlegend=True)
    fig.write_html(os.path.join(out_dir, 'line_comparison.html'))

def plot_line_peak_analysis(df, out_dir='out'):
    """Create peak hour analysis for each line"""
    for line in df['Metro_Line'].unique():
        line_data = df[df['Metro_Line'] == line]
        
        plt.figure(figsize=(15, 8))
        peak_data = line_data[line_data['Time_Period'] == 'peak'].groupby('Station_ID')['Occupancy_Rate'].mean()
        peak_data.plot(kind='bar')
        plt.title(f'{line} Peak Hour Station Occupancy')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Average Occupancy Rate (%)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{line}_peak_analysis.png'))
        plt.close()

def plot_hourly_line_analysis(df, out_dir='out'):
    """Create detailed hourly analysis with modern design"""
    for line in df['Metro_Line'].unique():
        line_data = df[df['Metro_Line'] == line]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Hourly Occupancy Pattern',
                'Station Comparison',
                'Peak vs Off-Peak Distribution',
                'Weather Impact'
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Hourly pattern
        weekday_data = line_data[~line_data['Is_Weekend']].groupby('Hour')['Occupancy_Rate'].mean()
        weekend_data = line_data[line_data['Is_Weekend']].groupby('Hour')['Occupancy_Rate'].mean()
        
        fig.add_trace(
            go.Scatter(
                x=weekday_data.index, 
                y=weekday_data, 
                name='Weekday',
                mode='lines+markers',
                line=dict(color=MODERN_COLORS['primary'], width=3),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=weekend_data.index, 
                y=weekend_data, 
                name='Weekend',
                mode='lines+markers',
                line=dict(color=MODERN_COLORS['secondary'], width=3),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        # Station comparison
        station_avg = line_data.groupby('Station_ID')['Occupancy_Rate'].mean().sort_values(ascending=True)
        fig.add_trace(
            go.Bar(
                x=station_avg.index, 
                y=station_avg, 
                name='Average Occupancy',
                marker_color=MODERN_COLORS['accent1']
            ),
            row=1, col=2
        )
        
        # Peak vs off-peak distribution
        fig.add_trace(
            go.Box(
                y=line_data[line_data['Time_Period'] == 'peak']['Occupancy_Rate'],
                name='Peak Hours',
                boxpoints='outliers',
                marker_color=MODERN_COLORS['primary'],
                line_color=MODERN_COLORS['primary']
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Box(
                y=line_data[line_data['Time_Period'] != 'peak']['Occupancy_Rate'],
                name='Off-Peak Hours',
                boxpoints='outliers',
                marker_color=MODERN_COLORS['secondary'],
                line_color=MODERN_COLORS['secondary']
            ),
            row=2, col=1
        )
        
        # Weather impact
        weather_impact = line_data.groupby('Weather_Condition')['Occupancy_Rate'].mean()
        fig.add_trace(
            go.Bar(
                x=weather_impact.index, 
                y=weather_impact, 
                name='Weather Impact',
                marker_color=MODERN_COLORS['accent2']
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            template=modern_template,
            height=1000,
            width=1500,
            title=dict(
                text=f"Detailed Analysis: {line} Line",
                font=dict(size=24),
                y=0.95
            ),
            showlegend=True,
            legend=dict(
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor=MODERN_COLORS['grid'],
                borderwidth=1
            )
        )
        
        # Update axes
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=MODERN_COLORS['grid'])
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=MODERN_COLORS['grid'])
        
        fig.write_html(os.path.join(out_dir, f'{line}_detailed_analysis.html')) 