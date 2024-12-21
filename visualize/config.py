"""
Configuration for visualization styles and colors
"""

import plotly.graph_objects as go

# Modern color schemes
MODERN_COLORS = {
    'primary': '#1f77b4',     # Modern blue
    'secondary': '#2ca02c',   # Modern green
    'accent1': '#ff7f0e',     # Modern orange
    'accent2': '#9467bd',     # Modern purple
    'background': '#ffffff',  # White
    'text': '#2d3047',       # Dark blue-gray
    'grid': '#e5e5e5'        # Light gray
}

LINE_COLORS = {
    'M1-2-3': '#ff4b4b',  # Vibrant red
    'M4': '#4b7bff',      # Vibrant blue
    'A1': '#47d16c'       # Vibrant green
}

# Modern plotly template
modern_template = go.layout.Template()
modern_template.layout = go.Layout(
    paper_bgcolor=MODERN_COLORS['background'],
    plot_bgcolor=MODERN_COLORS['background'],
    title=dict(x=0.5, xanchor='center'),
    legend=dict(
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor=MODERN_COLORS['grid'],
        borderwidth=1
    ),
    xaxis=dict(
        gridcolor=MODERN_COLORS['grid'],
        gridwidth=1,
        zerolinecolor=MODERN_COLORS['grid']
    ),
    yaxis=dict(
        gridcolor=MODERN_COLORS['grid'],
        gridwidth=1,
        zerolinecolor=MODERN_COLORS['grid']
    )
) 