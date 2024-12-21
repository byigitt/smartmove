"""
Main script to generate visualizations for Ankara Metro data
"""

from visualize.main import MetroDataVisualizer

def main():
    """Generate all visualizations for the metro data"""
    visualizer = MetroDataVisualizer()
    visualizer.generate_all_visualizations()

if __name__ == "__main__":
    main()