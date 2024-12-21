#!/usr/bin/env python3
"""
Generate synthetic Ankara Metro passenger data
"""

import os
import argparse
from datetime import datetime
from generator import generate_data

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Generate synthetic Ankara Metro passenger data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--days', 
        type=int, 
        default=365,
        help='Number of days to generate data for'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default='ankara_metro_crowding_data.csv',
        help='Output CSV file path'
    )
    
    parser.add_argument(
        '--out-dir', 
        type=str, 
        default='data',
        help='Output directory for the CSV file'
    )
    
    return parser.parse_args()

def main():
    """Main function to generate data"""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Construct full output path
    output_path = os.path.join(args.out_dir, args.output)
    
    print(f"Generating {args.days} days of Ankara Metro passenger data...")
    print(f"Output will be saved to: {output_path}")
    
    # Generate the data
    generate_data(
        num_days=args.days,
        output_file=output_path
    )

if __name__ == '__main__':
    main() 