# Instructions for Generating Metro Passenger Data CSV

## Data Requirements
1. System Parameters:
   - 4 metro lines (M1, M2, M3, M4)
   - 57 stations (S1 to S57)
   - Train capacity: 650 passengers per train
   - S29 is the central/busiest station
   - S1 and S57 are the least busy stations
   - Passenger count must be 0 at terminal stations

## Steps to Generate Data

1. Setup Phase:
   - Create lists of metro lines and station names
   - Define time periods (peak hours, off-peak hours)
   - Set up passenger flow patterns based on station popularity

2. Station Popularity Calculation:
   - Create a popularity curve with S29 as the peak
   - Decrease popularity as stations move away from S29
   - Assign base passenger flow rates to each station

3. Passenger Flow Generation:
   - Calculate boarding passengers based on:
     * Station popularity
     * Time of day
     * Available capacity
   - Calculate alighting passengers based on:
     * Distance from destination
     * Current passenger load
     * Terminal station requirements

4. Data Validation Rules:
   - Ensure total passengers never exceed train capacity (650)
   - Maintain realistic ratios between boarding and alighting passengers
   - Verify passenger count reaches 0 at terminal stations
   - Check that passenger flows align with station popularity

5. CSV Structure:
   - Columns to include:
     * Timestamp
     * Metro Line
     * Station ID
     * Station Name
     * Boarding Passengers
     * Alighting Passengers
     * Current Load
     * Capacity Utilization (%)

6. Time Series Considerations:
   - Include multiple trips throughout the day
   - Account for peak hours (morning/evening rush)
   - Consider reduced service during off-peak hours
   - Include weekday/weekend variations

7. Quality Checks:
   - Verify no negative passenger counts
   - Ensure logical passenger flow between stations
   - Validate capacity constraints
   - Check for realistic distribution patterns

## Expected Output
The final CSV file should provide a realistic simulation of passenger flows across the metro system, reflecting real-world patterns such as:
- Higher passenger volumes during peak hours
- Concentrated activity at central stations
- Gradual passenger accumulation and dispersal
- Realistic transfer patterns at major junction stations
