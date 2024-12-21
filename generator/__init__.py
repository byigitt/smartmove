"""
Ankara Metro Data Generator Package
"""

from .metro_generator import AnkaraMetroGenerator
__version__ = '1.0.0'
__author__ = 'cyberia'

def generate_data(num_days=365, output_file='ankara_metro_crowding_data_realistic.csv'):
    """Generate synthetic metro passenger data and save to CSV"""
    print("Generating synthetic metro passenger data...")
    generator = AnkaraMetroGenerator()
    df = generator.generate_passenger_flow(num_days=num_days)
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")
    print(f"Generated {len(df)} records for {len(df['Station_ID'].unique())} stations") 