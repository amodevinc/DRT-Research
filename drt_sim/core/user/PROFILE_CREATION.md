# User Profiles for DRT Simulation

This document describes how user profiles are created for the Demand-Responsive Transport (DRT) simulation system, including the methodology, assumptions, and data structures.

## Overview

The system creates user profiles based on acceptance weights data. Each profile contains:
- User identification (with format "U1", "U2", etc.)
- Time and cost preference parameters
- Simplified service preferences
- Feature weights for acceptance decisions
- Historical usage data

## Data Source

User profiles are generated from `acceptance_weights.csv`, which contains the following columns:
- `id`: Numeric identifier (converted to format "U1", "U2", etc.)
- `access`: Weight for access/walking time to origin
- `wait`: Weight for waiting time
- `ivt`: Weight for in-vehicle travel time
- `egress`: Weight for walking time from destination

## Profile Creation Process

1. **Weight Normalization**: Weights are converted to absolute values and normalized to sum to 1.0
2. **Service Preference Assignment**: Simplified preferences are assigned based on comparing in-vehicle time and wait time weights
3. **Time and Cost Parameter Generation**: Parameters are generated based on weight values
4. **Historical Data Simulation**: Trip counts and ratings are randomly generated

## Key Assumptions

### Weight Interpretation
- **access**: Higher values indicate users care more about walking time to pickup points
- **wait**: Higher values indicate users care more about waiting time for pickups
- **ivt**: Higher values indicate users care more about in-vehicle time
- **egress**: Higher values indicate users care more about walking time from drop-off stop to final destination point

### Simplified Service Preference Assignment
Service preferences have been simplified to two categories that can be reasonably inferred from the data:
- **SPEED**: Assigned when in-vehicle time weight > wait time weight
- **RELIABILITY**: Assigned when wait time weight ≥ in-vehicle time weight

This simplification was made because the original four preferences (SPEED, COMFORT, ECONOMY, RELIABILITY) would require additional data not available in the weight profiles.

### Time Parameter Assumptions
- Users with higher weight for a time component have lower tolerance (max value) for that component
- Walking time values range between 1-6 minutes
- Waiting time values range between 5-15 minutes
- In-vehicle time values range between 15-40 minutes
- Maximum cost ranges between 20-40 currency units
- Maximum acceptable delay ranges between 5-10 minutes

### Historical Data Generation
- Trip counts range between 0-50 past trips
- Acceptance rates range between 50-95%
- Ratings range between 3.0-5.0 (up to 5 ratings per user)

## Profile Structure

Each user profile is stored as a JSON file with the following structure:
```json
{
  "id": "U1",
  "max_walking_time_to_origin": 3.0,
  "max_walking_time_from_destination": 3.0,
  "max_waiting_time": 10.0,
  "max_in_vehicle_time": 25.0,
  "max_cost": 30.0,
  "max_acceptable_delay": 7.0,
  "service_preference": "speed",
  "weights": {
    "walking_time_to_origin": 0.4,
    "wait_time": 0.3,
    "in_vehicle_time": 0.2,
    "walking_time_from_destination": 0.1,
    "time_of_day": 0.0,
    "day_of_week": 0.0,
    "distance_to_pickup": 0.0
  },
  "historical_trips": 25,
  "historical_acceptance_rate": 0.75,
  "historical_ratings": [4.5, 3.8, 4.2, 4.0]
}
```

## Statistics

During profile generation, the system collects and saves the following statistics:
- Total number of profiles created
- Distribution of service preferences
- Average historical trips
- Average maximum walking times
- Average maximum waiting time
- Average maximum in-vehicle time

## Usage

To create user profiles, run:
```bash
python create_user_profiles.py --weights acceptance_weights.csv --output data/users
```

Optional arguments:
- `--analyze`: Generate visualizations of weight distributions (requires matplotlib and seaborn)

## Integration with User Profile Manager

These profiles are designed to be loaded by the `UserProfileManager` class, which provides methods for:
- Loading profiles from disk
- Accessing user-specific preferences and weights
- Updating user data
- Saving modified profiles

## Testing and Verification

You can test the generated profiles with the `load_and_test_profiles.py` script:

```bash
python load_and_test_profiles.py --profiles data/users/user_profiles --output data/test_results --visualize
```

This script will:
- Load all profiles and verify their structure
- Test loading them into the UserProfile class
- Generate statistics about the profile distributions
- Create visualizations (with the `--visualize` flag)

## Limitations and Future Improvements

1. **Limited service preference modeling**: Currently simplified to just SPEED and RELIABILITY preferences
2. **No temporal variations**: Current implementation does not account for temporal variations in user preferences
3. **No group behavior**: No modeling of group travel behavior
4. **Limited historical data**: Historical data is randomly generated rather than based on actual usage patterns
5. **Potential improvements**:
   - Incorporate more data sources to better determine preferences
   - Use machine learning approaches to identify preference patterns
   - Add temporal variation in preferences (time of day, day of week)
   - Incorporate demographic factors if available