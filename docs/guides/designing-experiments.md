# DRT Research Platform: Experiment Design Patterns Guide

## Core Experiment Types

### 1. Parameter Sweeps
Used when you want to understand how varying specific parameters affects the system while keeping everything else constant.

```yaml
type: "parameter_sweep"
parameter_sweep:
  enabled: true
  method: "grid"
  parameters:
    vehicle.fleet_size: [5, 10, 15, 20]
    algorithm.batch_interval: [30, 60, 90]
```

**Use When:**
- Exploring sensitivity to specific parameters
- Finding optimal parameter values
- Understanding parameter interactions
- Conducting systematic parameter space exploration

**Key Characteristics:**
- All other configuration remains constant
- Automatically generates scenarios for each parameter combination
- Focused on understanding specific parameter effects
- More suitable for optimization studies

### 2. Scenario Comparisons
Used when comparing fundamentally different configurations or strategies.

```yaml
type: "scenario_comparison"
experiments:
  peak_vs_offpeak:
    scenarios:
      morning_peak:
        demand:
          generator_type: "csv"
          csv_config:
            files:
              - file_path: "data/demands/morning_peak.csv"
        vehicle:
          fleet_size: 20
      
      off_peak:
        demand:
          generator_type: "csv"
          csv_config:
            files:
              - file_path: "data/demands/off_peak.csv"
        vehicle:
          fleet_size: 15
```

**Use When:**
- Comparing different strategies
- Testing system behavior under different conditions
- Evaluating multiple aspects changing simultaneously
- Conducting case studies

**Key Characteristics:**
- Explicitly defined scenarios
- Can vary multiple parameters between scenarios
- Focused on comparing distinct configurations
- More suitable for strategy evaluation

## Decision Guide

### Choose Parameter Sweep When:
1. Question Format: "How does X affect the system?"
2. You're varying 1-3 specific parameters systematically
3. You want to find optimal parameter values
4. All other conditions should remain constant

Example Study:
```yaml
type: "parameter_sweep"
parameter_sweep:
  enabled: true
  method: "grid"
  parameters:
    vehicle.fleet_size: [5, 10, 15, 20]  # How does fleet size affect performance?
```

### Choose Scenario Comparison When:
1. Question Format: "Which approach works better?"
2. You're comparing different strategies or conditions
3. Multiple parameters vary between scenarios
4. You're testing distinct use cases

Example Study:
```yaml
type: "scenario_comparison"
experiments:
  dispatch_comparison:
    scenarios:
      fcfs_baseline:
        algorithm:
          dispatch_strategy: "fcfs"
          batch_interval: 30
        vehicle:
          fleet_size: 15
      
      ga_dispatch:
        algorithm:
          dispatch_strategy: "ga_dispatch"
          batch_interval: 60
        vehicle:
          fleet_size: 15
```

## Common Patterns

### 1. Algorithm Comparison Study
```yaml
type: "scenario_comparison"
experiments:
  algorithm_comparison:
    scenarios:
      fcfs:
        algorithm:
          dispatch_strategy: "fcfs"
      genetic:
        algorithm:
          dispatch_strategy: "ga_dispatch"
      reinforcement:
        algorithm:
          dispatch_strategy: "rl_dispatch"
```

### 2. Demand Pattern Study
```yaml
type: "scenario_comparison"
experiments:
  demand_patterns:
    scenarios:
      base_demand:
        demand:
          csv_config:
            files:
              - file_path: "data/demands/base.csv"
      peak_demand:
        demand:
          csv_config:
            files:
              - file_path: "data/demands/peak.csv"
```

### 3. Parameter Sensitivity Study
```yaml
type: "parameter_sweep"
parameter_sweep:
  enabled: true
  method: "grid"
  parameters:
    vehicle.fleet_size: [5, 10, 15, 20]
    vehicle.capacity: [4, 6, 8]
```

## Best Practices

1. **Clear Naming:**
   - Use descriptive names for experiments and scenarios
   - Include the purpose in the description field
   - Add relevant tags for easy filtering

2. **Documentation:**
   - Always include a clear description
   - Document what you're trying to learn
   - Note any assumptions or constraints

3. **Metrics:**
   - Define metrics specific to your study goals
   - Include both general and scenario-specific metrics
   - Consider adding custom metrics for specific analyses

4. **Configuration Management:**
   - Keep configurations in version control
   - Document any external dependencies
   - Include data source versions

5. **Validation:**
   - Start with small-scale test runs
   - Verify metrics collection
   - Check for configuration conflicts

## Example Study Templates

### Performance Optimization Study
```yaml
type: "parameter_sweep"
metadata:
  name: "fleet_optimization"
  description: "Optimize fleet size and batch interval for peak efficiency"
parameter_sweep:
  enabled: true
  method: "grid"
  parameters:
    vehicle.fleet_size: [5, 10, 15, 20, 25]
    algorithm.batch_interval: [30, 60, 90]
metrics:
  additional_metrics:
    - fleet_efficiency
    - peak_performance
    - cost_per_trip
```

### Strategy Comparison Study
```yaml
type: "scenario_comparison"
metadata:
  name: "dispatch_strategies"
  description: "Compare different dispatch strategies under various demand patterns"
experiments:
  strategy_comparison:
    scenarios:
      baseline_fcfs:
        algorithm:
          dispatch_strategy: "fcfs"
      advanced_ga:
        algorithm:
          dispatch_strategy: "ga_dispatch"
      hybrid_approach:
        algorithm:
          dispatch_strategy: "hybrid"
metrics:
  additional_metrics:
    - computational_time
    - solution_quality
    - adaptation_speed
```

## Common Mistakes to Avoid

1. **Mixing Types:**
   - Don't enable parameter sweep while trying to run scenario comparisons
   - Choose one approach based on your study goals

2. **Over-complication:**
   - Don't vary too many parameters at once
   - Keep scenarios focused and well-defined

3. **Insufficient Metrics:**
   - Don't forget to add study-specific metrics
   - Ensure metrics align with study goals

4. **Poor Organization:**
   - Don't use generic names like "test1", "test2"
   - Keep related scenarios grouped in meaningful experiments

## When to Combine Approaches

While you typically choose one approach or the other, there are cases where you might want to run both types of studies sequentially:

1. First, use parameter sweeps to find optimal parameters for different strategies
2. Then, use scenario comparison to compare the optimized strategies

This two-phase approach helps ensure fair comparison between strategies while understanding parameter sensitivity.