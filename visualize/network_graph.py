"""
Network graph visualization for metro system
"""

import os
import pandas as pd
import plotly.graph_objects as go
from .styles import MODERN_COLORS, LINE_COLORS, modern_template

def create_station_network_graph(df, generator, out_dir='out'):
    """Create an interactive network graph with modern design"""
    # Define layout coordinates for each line
    layout_coordinates = {
        'M1-2-3': {
            # Koru to Kızılay (going diagonal down-right)
            'Koru': (0, 100),
            'Çayyolu': (5, 95),
            'Ümitköy': (10, 90),
            'Beytepe': (15, 85),
            'Tarım Bakanlığı-Danıştay': (20, 80),
            'Bilkent': (25, 75),
            'Orta Doğu Teknik Üniversitesi': (30, 70),
            'Maden Tetkik ve Arama': (35, 65),
            'Söğütözü': (40, 60),
            'Millî Kütüphane': (45, 55),
            'Necatibey': (50, 52),
            '15 Temmuz Kızılay Millî İrade': (55, 50),  # Central point
            # Kızılay to OSB (going diagonal up-right)
            'Sıhhiye': (60, 51),
            'Ulus': (65, 52),
            'Atatürk Kültür Merkezi': (70, 50),  # Adjusted to be at same y-level as Kızılay
            'Akköprü': (75, 55),
            'İvedik': (80, 60),
            'Yenimahalle': (85, 65),
            'Demetevler': (90, 70),
            'Hastane': (95, 75),
            'Macunköy': (100, 80),
            'Orta Doğu Sanayi ve Ticaret Merkezi': (105, 85),
            'Batıkent': (110, 90),
            'Batı Merkez': (115, 95),
            'Mesa': (120, 100),
            'Botanik': (125, 105),
            'İstanbul Yolu': (130, 110),
            'Eryaman 1-2': (135, 115),
            'Eryaman 5': (140, 120),
            'Devlet Mahallesi/1910 Ankaragücü': (145, 125),
            'Harikalar Diyarı': (150, 130),
            'Fatih': (155, 135),
            'Gaziosmanpaşa': (160, 140),
            'OSB-Törekent': (165, 145)
        },
        'M4': {
            # Kızılay to Şehitler (going horizontal right then up)
            '15 Temmuz Kızılay Millî İrade': (55, 50),
            'Adliye': (60, 50),
            'Gar': (65, 50),
            'Atatürk Kültür Merkezi': (70, 50),  # Same coordinates as M1-2-3 for proper transfer
            'Ankara Su ve Kanalizasyon İdaresi': (70, 40),  # Changed to go up after AKM
            'Dışkapı': (70, 35),
            'Meteoroloji': (70, 30),
            'Belediye': (70, 25),
            'Mecidiye': (70, 20),
            'Kuyubaşı': (70, 15),
            'Dutluk': (70, 10),
            'Şehitler': (70, 5)
        },
        'A1': {
            # AŞTİ to Dikimevi (horizontal with slight curve)
            'AŞTİ': (20, 40),
            'Emek': (25, 42),
            'Bahçelievler': (30, 44),
            'Beşevler': (35, 46),
            'Anadolu/Anıtkabir': (40, 48),
            'Maltepe': (45, 49),
            'Demirtepe': (50, 50),
            '15 Temmuz Kızılay Millî İrade': (55, 50),
            'Kolej': (60, 48),
            'Kurtuluş': (65, 46),
            'Dikimevi': (70, 44)
        }
    }

    # Create nodes and edges
    nodes = []
    edges = []
    added_stations = set()
    
    # Calculate metrics
    peak_occupancy = df[df['Time_Period'] == 'peak'].groupby('Station_ID')['Occupancy_Rate'].mean()
    offpeak_occupancy = df[df['Time_Period'] != 'peak'].groupby('Station_ID')['Occupancy_Rate'].mean()
    weekday_occupancy = df[~df['Is_Weekend']].groupby('Station_ID')['Occupancy_Rate'].mean()
    weekend_occupancy = df[df['Is_Weekend']].groupby('Station_ID')['Occupancy_Rate'].mean()
    
    # Create nodes and edges
    for line, stations_coords in layout_coordinates.items():
        stations = list(stations_coords.keys())
        for i, station in enumerate(stations):
            if station not in added_stations:
                # Get station metrics
                station_data = df[df['Station_ID'] == station]
                avg_occupancy = station_data['Occupancy_Rate'].mean()
                max_occupancy = station_data['Occupancy_Rate'].max()
                peak_avg = peak_occupancy.get(station, 0)
                offpeak_avg = offpeak_occupancy.get(station, 0)
                weekday_avg = weekday_occupancy.get(station, 0)
                weekend_avg = weekend_occupancy.get(station, 0)
                
                nodes.append({
                    'id': station,
                    'label': station,
                    'line': line,
                    'x': stations_coords[station][0],
                    'y': stations_coords[station][1],
                    'avg_occupancy': avg_occupancy,
                    'max_occupancy': max_occupancy,
                    'peak_occupancy': peak_avg,
                    'offpeak_occupancy': offpeak_avg,
                    'weekday_occupancy': weekday_avg,
                    'weekend_occupancy': weekend_avg,
                    'is_transfer': station in ['15 Temmuz Kızılay Millî İrade', 'Atatürk Kültür Merkezi'],
                    'is_terminal': station in sum([details['terminal_stations'] for details in generator.metro_lines.values()], [])
                })
                added_stations.add(station)
            
            if i < len(stations) - 1:
                edges.append({
                    'source': station,
                    'target': stations[i + 1],
                    'line': line
                })
    
    # Create plotly figure
    node_df = pd.DataFrame(nodes)
    edge_df = pd.DataFrame(edges)
    
    fig = go.Figure()
    
    # Add edges (metro lines)
    for line in edge_df['line'].unique():
        line_edges = edge_df[edge_df['line'] == line]
        for _, edge in line_edges.iterrows():
            source_node = node_df[node_df['id'] == edge['source']].iloc[0]
            target_node = node_df[node_df['id'] == edge['target']].iloc[0]
            
            fig.add_trace(go.Scatter(
                x=[source_node['x'], target_node['x']],
                y=[source_node['y'], target_node['y']],
                mode='lines',
                name=f'{line} Line',
                line=dict(width=4, color=LINE_COLORS[line]),
                showlegend=True if edge.name == 0 else False
            ))
    
    # Add regular stations
    regular_nodes = node_df[~(node_df['is_transfer'] | node_df['is_terminal'])]
    fig.add_trace(go.Scatter(
        x=regular_nodes['x'],
        y=regular_nodes['y'],
        mode='markers+text',
        text=regular_nodes['label'],
        textposition='top center',
        name='Regular Stations',
        marker=dict(
            size=regular_nodes['avg_occupancy']/2 + 10,
            color=regular_nodes['avg_occupancy'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title=dict(
                    text='Average Occupancy Rate (%)',
                    font=dict(size=14)
                ),
                thickness=15,
                len=0.7
            )
        ),
        hovertemplate=(
            '<b>%{text}</b><br>' +
            'Average Occupancy: %{marker.color:.1f}%<br>' +
            'Peak Hours: %{customdata[0]:.1f}%<br>' +
            'Off-Peak: %{customdata[1]:.1f}%<br>' +
            'Weekday: %{customdata[2]:.1f}%<br>' +
            'Weekend: %{customdata[3]:.1f}%<br>' +
            'Maximum: %{customdata[4]:.1f}%<extra></extra>'
        ),
        customdata=regular_nodes[['peak_occupancy', 'offpeak_occupancy', 
                                'weekday_occupancy', 'weekend_occupancy', 'max_occupancy']]
    ))
    
    # Add terminal stations
    terminal_nodes = node_df[node_df['is_terminal']]
    fig.add_trace(go.Scatter(
        x=terminal_nodes['x'],
        y=terminal_nodes['y'],
        mode='markers+text',
        text=terminal_nodes['label'],
        textposition='top center',
        name='Terminal Stations',
        marker=dict(
            size=terminal_nodes['avg_occupancy']/2 + 12,
            color=terminal_nodes['avg_occupancy'],
            colorscale='Viridis',
            line=dict(color=MODERN_COLORS['accent2'], width=2),
            symbol='diamond',
            showscale=False
        ),
        hovertemplate=(
            '<b>%{text}</b> (Terminal)<br>' +
            'Average Occupancy: %{marker.color:.1f}%<br>' +
            'Peak Hours: %{customdata[0]:.1f}%<br>' +
            'Off-Peak: %{customdata[1]:.1f}%<br>' +
            'Weekday: %{customdata[2]:.1f}%<br>' +
            'Weekend: %{customdata[3]:.1f}%<br>' +
            'Maximum: %{customdata[4]:.1f}%<extra></extra>'
        ),
        customdata=terminal_nodes[['peak_occupancy', 'offpeak_occupancy', 
                                 'weekday_occupancy', 'weekend_occupancy', 'max_occupancy']]
    ))
    
    # Add transfer stations
    transfer_nodes = node_df[node_df['is_transfer']]
    fig.add_trace(go.Scatter(
        x=transfer_nodes['x'],
        y=transfer_nodes['y'],
        mode='markers+text',
        text=transfer_nodes['label'],
        textposition='top center',
        name='Transfer Stations',
        marker=dict(
            size=transfer_nodes['avg_occupancy']/2 + 15,
            color=transfer_nodes['avg_occupancy'],
            colorscale='Viridis',
            line=dict(color=MODERN_COLORS['primary'], width=2),
            symbol='square',
            showscale=False
        ),
        hovertemplate=(
            '<b>%{text}</b> (Transfer)<br>' +
            'Average Occupancy: %{marker.color:.1f}%<br>' +
            'Peak Hours: %{customdata[0]:.1f}%<br>' +
            'Off-Peak: %{customdata[1]:.1f}%<br>' +
            'Weekday: %{customdata[2]:.1f}%<br>' +
            'Weekend: %{customdata[3]:.1f}%<br>' +
            'Maximum: %{customdata[4]:.1f}%<extra></extra>'
        ),
        customdata=transfer_nodes[['peak_occupancy', 'offpeak_occupancy', 
                                 'weekday_occupancy', 'weekend_occupancy', 'max_occupancy']]
    ))
    
    # Update layout
    fig.update_layout(
        template=modern_template,
        title={
            'text': 'Ankara Metro Network Analysis',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor=MODERN_COLORS['grid'],
            borderwidth=1
        ),
        hovermode='closest',
        height=1000,
        width=1500,
        margin=dict(t=100, l=50, r=50, b=50),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        annotations=[
            dict(
                text="Hover over stations for detailed metrics",
                xref="paper", yref="paper",
                x=0, y=-0.05,
                showarrow=False
            )
        ]
    )
    
    fig.write_html(os.path.join(out_dir, 'metro_network.html')) 