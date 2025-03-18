# User Weight Evolution Tracking System

This document explains the enhanced user profile system for the DRT simulation platform, focusing on the ability to track weight changes over time across different simulation studies.

## Overview

The updated system allows researchers to:

1. Create user profiles with base weights from acceptance weights data
2. Track how weights evolve over time as users interact with the system
3. Organize weight changes by study and simulation run
4. Reset weights to base values or specific simulation points
5. Analyze and visualize weight evolution patterns

## Key Components

### Modified Files

- `create_user_profiles.py`: Updated to remove service preference and add weight history structure
- `user.py`: Modified `UserProfile` class to track weight history and provide weight evolution analysis
- `user_profile_manager.py`: Enhanced to support study/simulation contexts and manage weight changes

### New Files

- `analyze_weight_changes.py`: Script to analyze and visualize weight evolution patterns
- `example_weight_update.py`: Example script demonstrating how to update weights in a simulation

## Directory Structure

```
data/
  users/
    user_profiles/       # User profile JSON files
    change_logs/         # Weight change logs
      study_1/           # First study
        simulation_1/    # First simulation in study_1
          U1_changes.json
          U2_changes.json
          ...
        simulation_2/
          ...
      study_2/
        ...
    stats/               # Summary statistics
    plots/               # Visualizations (if analyze option used)
```

## User Profile Structure

User profiles now include:

```json
{
  "id": "U1",
  "max_walking_time_to_origin": 3.0,
  "max_walking_time_from_destination": 3.0,
  "max_waiting_time": 10.0,
  "max_in_vehicle_time": 25.0,
  "max_price": 30.0,
  "max_acceptable_delay": 7.0,
  "base_weights": {
    "walking_time_to_origin": 0.4,
    "wait_time": 0.3,
    "in_vehicle_time": 0.2,
    "walking_time_from_destination": 0.1,
    "time_of_day": 0.0,
    "day_of_week": 0.0,
    "distance_to_pickup": 0.0
  },
  "weights": { /* Current weights - same structure as base_weights */ },
  "weight_history": [
    {
      "timestamp": "2023-05-10T14:30:45",
      "weights": { /* Weights at this point */ },
      "study_id": "study_1",
      "simulation_id": "simulation_1",
      "reason": "Profile created"
    },
    {
      "timestamp": "2023-05-10T14:35:12",
      "weights": { /* Weights at this point */ },
      "study_id": "study_1",
      "simulation_id": "simulation_1",
      "reason": "Accepted ride with long wait time",
      "prev_weights": { /* Previous weights before this update */ }
    },
    // More weight change records...
  ],
  "historical_trips": 25,
  "historical_acceptance_rate": 0.75,
  "historical_ratings": [4.5, 3.8, 4.2, 4.0],
  "created_at": "2023-05-10T14:30:45",
  "last_updated": "2023-05-10T15:30:45"
}
```

## Change Log Structure

Each change log file (`user_id_changes.json`) contains:

```json
[
  {
    "timestamp": "2023-05-10T14:30:45",
    "type": "creation",
    "study_id": "study_1",
    "simulation_id": "simulation_1",
    "description": "Profile created",
    "weights": { /* Weights at creation */ }
  },
  {
    "timestamp": "2023-05-10T14:35:12",
    "type": "weight_update",
    "study_id": "study_1",
    "simulation_id": "simulation_1",
    "description": "Accepted ride with long wait time",
    "weights": { /* Updated weights */ }
  },
  // More change records...
]
```

## Usage Examples

### Creating User Profiles

```bash
python create_user_profiles.py --weights data/user/acceptance_weights.csv --output data/users
```

### Running a Simulation with Weight Updates

```bash
python example_weight_update.py --profiles data/users/user_profiles --study study_1 --simulation sim_1
```

### Analyzing Weight Changes

```bash
python analyze_weight_changes.py --logs data/users/change_logs --output data/analysis --weights wait_time in_vehicle_time
```

### Analyzing a Specific User

```bash
python analyze_weight_changes.py --logs data/users/change_logs --output data/analysis --user U1
```

## Key User Profile Methods

- `update_weights(new_weights, reason)`: Update weights with a reason
- `reset_to_base_weights(reason)`: Reset weights to original base values
- `reset_to_simulation_point(study_id, simulation_id, reason)`: Reset to a specific point
- `get_weight_evolution(weight_name, study_id)`: Get history of weight changes

## Key UserProfileManager Methods

- `set_study_context(study_id, simulation_id)`: Set current study context
- `update_user_weights(user_id, new_weights, reason)`: Update weights with tracking
- `reset_user_weights_to_base(user_id, reason)`: Reset to base weights
- `reset_user_weights_to_simulation(user_id, study_id, simulation_id, reason)`: Reset to simulation point
- `get_weight_change_history(user_id, weight_name, study_id)`: Get weight change history
- `compare_weights(user_id, study1, sim1, study2, sim2)`: Compare weights between points

## Analysis Features

The `analyze_weight_changes.py` script provides:

1. **Weight evolution visualization**: Plots showing how weights change over time
2. **Weight correlation analysis**: Heatmaps showing correlations between different weights
3. **Cross-user comparisons**: Comparisons of how different users adapt their preferences
4. **Statistical summaries**: Key statistics about weight changes

## Future Enhancements

Potential future enhancements to consider:

1. Machine learning models to predict weight evolution patterns
2. Clustering users based on weight adaptation patterns
3. More sophisticated visualization tools
4. Integration with external analytics platforms 