
# DRT Research Simulation Platform - Reference Blueprint

**Purpose:**  
This document serves as a comprehensive reference for building a modular, extensible, and research-oriented Demand Responsive Transportation (DRT) simulation platform. It outlines the system’s goals, architectural structure, key components, and considerations for implementation. Presenting this context to a language model in future interactions will assist in coding specific components, integrating new algorithms, or extending functionalities.

---

## High-Level Goals & Philosophy

1. **Research-Focused:**  
   The system is designed for experimentation with various DRT strategies, algorithms, and scenarios. It supports comparing methods (e.g., genetic algorithms, heuristic-based dispatching, reinforcement learning) and analyzing performance under diverse conditions.

2. **Modular & Extensible:**  
   A plug-in architecture and well-defined interfaces allow easy integration of new algorithms, cost functions, demand models, and user acceptance models without restructuring core logic.

3. **User-Centric Modeling:**  
   Incorporate user preferences, acceptance models (logit, ML-based), pre-booked vs. real-time requests, and multi-user segment scenarios. The system considers how changes in service attributes affect user behavior.

4. **Realistic & Multi-Faceted Scenarios:**  
   Handle dynamic networks (OSM), vehicle routing with disruptions, weather influences, seasonal demand, multi-operator environments, EV fleets, and integration with multi-modal transit options.

5. **Comprehensive Analysis Tools:**  
   Provide extensive metrics, statistical analysis, parameter sweeps, scenario comparisons, and visualization (maps, time-series charts, demand heatmaps). These tools enable insightful interpretation of results and support reproducible research.

---

## Target Features

- **Discrete Event Simulation (DES):**  
  A core simulation engine to manage events (vehicle movements, passenger requests, route assignments) in simulated time.

- **Parameterized Scenarios:**  
  Load scenario configurations from structured files (YAML/JSON) defining network data, demand patterns, user acceptance parameters, algorithms used, and experimental conditions.

- **Demand Modeling & User Behavior:**  
  Generate time-varying, event-driven demand with user segments defined by distinct acceptance behaviors. Incorporate pre-bookings, user tolerance for wait times, willingness-to-pay, and mode choice modeling.

- **Algorithmic Flexibility:**
  - Stop Selection: Cost-based selection, grid-based etc, h3 spatial querying etc.
  - Routing: Dijkstra, time-dependent shortest paths, GA-based routing, heuristic or ML-enhanced routing strategies.
  - Dispatch: Naive FCFS, batch matching, GA-based optimization, RL-based dispatch policies.
  - Matching & Cost Functions: Flexible policies, multi-objective considerations (service quality, cost, emissions).
  - User Acceptance Models: Logit models, ML-predicted acceptance, adaptive models responding to historical performance.

- **Stop Selection & Network Adaptation:**
  - Virtual stop generation via clustering or optimization.
  - Scoring stops by safety, accessibility, and coverage.
  - Iterative improvements in stop placement informed by previous runs.

- **Analysis & Visualization:**
  - Metrics: passenger related times (wait time, walking time, in vehicle time etc.), fleet utilization, user satisfaction etc.
  - Statistical Tools: Compare algorithms statistically, test scenario variations.
  - Visualizations: Interactive maps (e.g., Folium), route animations, temporal charts, demand heatmaps, and Pareto analysis for multi-objective optimization.

- **Reproducibility & Experiment Management:**
  - Parameter sweeps, batch experiments, versioned scenarios, and logging of random seeds.
  - Automated results extraction, scenario comparison, and standardized reporting.

---

## Proposed Directory Structure

A logical file structure that supports modularity, clarity, and future growth:

```
drt_research_platform/
├─ docs/
│  ├─ guides/
│  │  ├─ getting_started.md
│  │  ├─ scenario_setup.md
│  │  ├─ parameter_sweeps.md
│  │  ├─ algorithm_integration.md
│  │  ├─ user_behavior_modeling.md
│  │  ├─ visualization_and_analysis.md
│  │  ├─ best_practices_research.md
│  │  └─ ...
│  ├─ design_overviews/
│  │  ├─ architecture_diagram.md
│  │  ├─ data_flow.md
│  │  ├─ scenario_design_principles.md
│  │  └─ ...
│  ├─ research_case_studies/
│  │  ├─ prebooking_experiment.md
│  │  ├─ genetic_algorithm_example.md
│  │  ├─ user_acceptance_models.md
│  │  └─ ...
│  ├─ api_reference/    # Auto-generated API docs
│  └─ CHANGELOG.md

├─ drt_sim/
│  ├─ core/
│  │  ├─ simulation_engine.py        # DES loop
│  │  ├─ event_manager.py            # Event scheduling
│  │  ├─ state_management.py         # Track vehicles, requests
│  │  ├─ logging_config.py
│  │  └─ hooks.py                    # Hooks for re-optimization, RL triggers
│  │
│  ├─ config/
│  │  ├─ parameters.py               # Param classes & validation
│  │  ├─ config_loader.py
│  │  ├─ scenario_definitions.py
│  │  └─ defaults/ (base scenarios)
│
│  ├─ network/
│  │  ├─ network_loader.py           # OSM/GTFS loading
│  │  ├─ graph_operations.py         # Shortest paths, caching
│  │  ├─ multi_modal_integration.py
│  │  ├─ disruptions.py
│  │  └─ transfer_points.py
│
│  ├─ demand/
│  │  ├─ demand_generator.py         # Time-varying requests
│  │  ├─ user_profiles.py            # Segments, preferences
│  │  ├─ user_acceptance.py          # Logit/ML models
│  │  ├─ prebooking_manager.py
│  │  └─ pattern_models.py
│
│  ├─ stops/
│  │  ├─ stop_selector.py
│  │  ├─ stop_scoring.py
│  │  ├─ adaptive_stop_placement.py
│  │  └─ ...
│
│  ├─ algorithms/
│  │  ├─ base_interfaces/
│  │  │  ├─ routing_base.py
│  │  │  ├─ dispatch_base.py
│  │  │  ├─ matching_base.py
│  │  │  ├─ cost_function_base.py
│  │  │  ├─ user_acceptance_base.py
│  │  │  └─ ...
│  │
│  │  ├─ routing/
│  │  │  ├─ dijkstra_routing.py
│  │  │  ├─ time_dependent_routing.py
│  │  │  ├─ genetic_routing.py
│  │  │  └─ ...
│  │
│  │  ├─ dispatch/
│  │  │  ├─ naive_dispatch.py
│  │  │  ├─ rl_dispatch.py
│  │  │  ├─ ga_dispatch.py
│  │  │  └─ ...
│  │
│  │  ├─ matching/
│  │  │  ├─ batch_matching.py
│  │  │  ├─ heuristic_matching.py
│  │  │  └─ ...
│  │
│  │  ├─ cost_functions/
│  │  │  ├─ simple_cost.py
│  │  │  ├─ multi_objective_cost.py
│  │  │  └─ ...
│  │
│  │  ├─ user_acceptance_models/
│  │  │  ├─ logit_acceptance.py
│  │  │  ├─ ml_acceptance.py
│  │  │  └─ ...
│  │
│  │  └─ plugin_loader.py
│
│  ├─ analysis/
│  │  ├─ metrics_collection.py
│  │  ├─ statistics.py
│  │  ├─ visualization.py
│  │  ├─ scenario_comparison.py
│  │  ├─ pareto_analysis.py
│  │  └─ data_extraction.py
│
│  ├─ experiments/
│  │  ├─ parameter_sweeps.py
│  │  ├─ batch_runner.py
│  │  ├─ scenario_repository.py
│  │  └─ reproducibility_tools.py
│
│  ├─ integration/
│  │  ├─ external_solvers.py
│  │  ├─ ml_integration.py
│  │  └─ traffic_sim_integration.py
│
│  ├─ utils/
│  │  ├─ caching.py
│  │  ├─ geometry.py
│  │  ├─ random_seed_manager.py
│  │  └─ ...
│
│  └─ __init__.py
│
├─ tests/
│  ├─ unit/
│  ├─ integration/
│  ├─ regression/
│  ├─ performance/
│  └─ ...
│
├─ scripts/
│  ├─ run_experiment.py      # CLI to run scenarios
│  ├─ analyze_results.py     # Generate reports, plots
│  ├─ tune_ga_parameters.py
│  └─ ...
│
├─ examples/
│  ├─ simple_scenario/
│  ├─ prebooking_example/
│  ├─ ga_dispatch_example/
│  ├─ user_logit_acceptance_example/
│  └─ ...
│
├─ requirements.txt
├─ README.md
└─ setup.py or pyproject.toml
```

This structure is designed so each subsystem (network, demand, algorithms, analysis) is isolated and can be extended independently.

---

## Implementation Guidelines

- **Abstract Interfaces & Base Classes:**  
  Use Python’s `abc` module to define abstract bases for routing, dispatch, matching, cost functions, and user acceptance. This ensures that adding a new algorithm requires only implementing the defined interface methods.

- **Configuration-Driven:**  
  Almost every aspect (which algorithm to use, parameter values, scenario selection) should be driven by external configuration files. This empowers non-developers and facilitates rapid experimentation.

- **Logging & Debugging:**  
  Provide multiple logging levels for debugging complex scenarios. Store full event logs to enable retrospective analyses.

- **Testing & Validation:**
  - **Unit Tests:** For each component (event scheduling, routing correctness).
  - **Integration Tests:** Run small scenarios with known outcomes.
  - **Regression Tests:** Ensure new code changes do not break previously validated behavior.
  - **Performance Profiling:** Identify bottlenecks (e.g., routing queries) and optimize.

---

## Example Use Cases

1. **Run a Simple Scenario:**  
   Use `run_experiment.py` with a basic scenario file to simulate a small fleet serving a small city network. Validate correctness and understand output formats.

2. **Parameter Sweeps & Sensitivity Analysis:**  
   Use `parameter_sweeps.py` to vary fleet sizes or GA parameters systematically, comparing results using `scenario_comparison.py`.

3. **Incorporate a Genetic Algorithm for Dispatch:**
   - Implement `GA_DispatchStrategy` in `algorithms/dispatch/ga_dispatch.py`.
   - Update the scenario config’s `dispatch` field.
   - Run experiments and analyze differences in wait times and user satisfaction relative to a baseline approach.

4. **Add a User Acceptance Model:**
   - Implement a new ML-based acceptance model in `algorithms/user_acceptance_models/ml_acceptance.py`.
   - Modify the scenario config to use it for certain user segments.
   - Observe how changed acceptance probabilities affect overall demand and service KPIs.

---

## Best Practices for Research

- **Reproducibility:**  
  Always record random seeds, commit scenario configs to version control, and document key parameters in logs.

- **Multi-Run Averages & Statistics:**  
  Run multiple simulations with different seeds and apply statistical tests (in `analysis/statistics.py`) to confirm differences are significant.

- **Visualization & Reporting:**  
  Use `analysis/visualization.py` and related scripts to create maps, charts, and dashboards. Combine them with scenario notes for comprehensive reporting in research papers or presentations.

---

## Future Extensions

- **Machine Learning Integration:**  
  Predict demand with ML models (`integration/ml_integration.py`), integrate deep RL for adaptive dispatch, or apply advanced generative models for scenario creation.

- **External Tools:**  
  Connect to traffic simulators (SUMO) or external solvers (OR-Tools, Gurobi) for more complex optimization tasks.

- **Policy & Equity Studies:**  
  Model the impact of fare subsidies, priority service areas, or EV fleet transition strategies on diverse communities.