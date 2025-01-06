
---

# DRT Research Simulation Platform - Reference Blueprint (Version 1.4)

---

Here's the properly aligned Change History Log with your new entry:

**## Change History Log**
| Version | Date | Changes | Contributors |
|---------|------|----------|--------------|
| **1.4** | Dec 17, 2024 | - Enhanced parameter models with research-focused capabilities and better organization<br>- Added ExperimentType enum and BaseParameters with experiment modification support<br>- Introduced comprehensive research API with Study, ExperimentConfiguration, and analysis tools<br>- Added support for parameter sweeps, variants, and scenarios<br>- Enhanced metadata handling and experimental tracking<br>- Added research-specific features like replications, random seeds, and statistical testing<br>- Restructured directory to better support research workflows | User, Claude |
| **1.3** | Dec 17, 2024 | - Implemented new modular architecture with clear separation of concerns<br>- Added comprehensive event system with validation, processing, and dispatching<br>- Created new monitoring system with MetricsCollector and ResourceMonitor<br>- Implemented robust state management with type-safe containers and history tracking<br>- Introduced SimulationOrchestrator for high-level control<br>- Enhanced event handling with specialized handlers and proper error management<br>- Added comprehensive metrics collection for service, vehicle, and system analysis | User, Claude |
| **1.2** | Dec 16, 2024 | - Enhanced core infrastructure with robust configuration management<br>- Implemented comprehensive base interfaces for routing, dispatch, matching, and user acceptance<br>- Added detailed parameter structures and validation using Pydantic<br>- Created foundation for extensible logging system<br>- Introduced core data types (Location, Request, Vehicle) with status tracking | User, Claude |
| **1.1** | Dec 16, 2024 | - Introduced versioning and Change History Log section<br>- Added a placeholder for RL-based dispatch in `algorithms/dispatch/rl_dispatch.py`<br>- Updated instructions for presenting this blueprint to AI assistants to ensure contextual alignment with system evolution | User, ChatGPT |
| **1.0** | Dec 12, 2024 | - Initial version of the blueprint based on high-level design goals and proposed architecture | User |

---

## Introduction

This document serves as the living blueprint for the Demand Responsive Transportation (DRT) Research Simulation Platform. It provides a structured view of the architecture, components, and design principles. It is intended as the central reference for developers, researchers, and AI-assisted tools like ChatGPT and Claude. By maintaining and presenting this document at the start of new AI interaction sessions, you will ensure the model has the context needed to remain aligned with the current system state.

---

## High-Level Goals & Philosophy

1. **Research-Focused:**  
   The platform supports experimentation with DRT strategies, algorithms, and scenarios for research purposes. The goal is to facilitate exploration of varied methods (e.g., genetic algorithms, heuristic-based dispatching, reinforcement learning) and to allow performance analysis under diverse conditions.

2. **Modular & Extensible:**  
   A plug-in architecture and well-defined interfaces enable easy integration of new algorithms, cost functions, demand models, and user acceptance models without requiring major refactoring.

3. **User-Centric Modeling:**  
   The system models user preferences, acceptance behaviors (logit, ML-based), pre-booked vs. real-time requests, and multi-user segments. Changes in service attributes affect user decisions and overall demand.

4. **Realistic & Multi-Faceted Scenarios:**  
   Handle dynamic networks (e.g., OSM-based), disruptions, seasonal variations, EV fleets, multi-operator settings, and integration with multi-modal transit options.

5. **Comprehensive Analysis Tools:**  
   Offer rich metrics, statistical analysis, parameter sweeps, scenario comparisons, and visualizations. Ensure reproducibility and provide tools for interpreting complex results.

---

## Target Features

- **Discrete Event Simulation (DES):**  
  Central to the platform is a simulation engine that manages events (vehicle movements, passenger requests) over simulated time.

- **Parameterized Scenarios:**  
  Scenarios are defined by configuration files (YAML/JSON), specifying network data, demand patterns, user acceptance models, and algorithm choices.

- **Demand & User Behavior Modeling:**  
  Time-varying demand, user profiles with distinct acceptance thresholds, willingness-to-pay models, and the ability to handle both pre-booked and on-demand requests.

- **Algorithmic Flexibility:**
  - **Routing:** Dijkstra, time-dependent shortest paths, genetic algorithms, and potentially machine learning-assisted routing.
  - **Dispatch:** Naive FCFS, batch matching, genetic algorithm-based dispatch, RL-based dispatch (**new in Version 1.1 as an example**).
  - **Matching:** Batch matching, heuristic approaches, and more advanced frameworks as needed.
  - **Cost Functions:** Simple travel-time-based to complex multi-objective models.
  - **User Acceptance Models:** Logit, ML-based predictions, adaptive models that evolve with historical performance data.

- **Stop Selection & Network Adaptation:**
  - Virtual stop generation through clustering or optimization.
  - Scoring stops on accessibility and service coverage.
  - Handling network disruptions and dynamically adapting routes.

- **Analysis & Visualization:**
  - Metrics: wait times, in-vehicle times, walk times, reliability, and satisfaction scores.
  - Statistical tools: scenario comparisons, parameter sweeps, significance testing.
  - Visualizations: interactive maps, temporal charts, animations, and Pareto front analyses.

- **Reproducibility & Experiment Management:**
  - Parameter sweeps, batch experiments, versioned scenarios, logging of random seeds.
  - Automated results extraction and standardized reporting procedures.

---

# Directory Structure

Root: `/Users/alainmorrisdev/development/KNUT/DRT/drt_research_platform`

```
- README.md
- data/
    - demands/
        - S1_real_time.csv
        - S2_real_time.csv
        - S3_real_time.csv
        - base_real_time.csv
        - cleaned_knut_passenger_requests.csv
        - knut_passenger_requests.csv
        - knut_weekend_data_1.csv
        - knut_weekend_data_2.csv
        - knut_weekend_data_3.csv
        - preprocessed_base_real_time.csv
        - valid_requests_S1_real_time.csv
        - valid_requests_S2_real_time.csv
        - valid_requests_S3_real_time.csv
        - valid_requests_base_real_time.csv
    - networks/
        - hwaseong_drive.graphml
        - hwaseong_walk.graphml
        - knut_drive.graphml
        - knut_walk.graphml
        - nyc_network.geojson
- drt_sim/
    - __init__.py
    - algorithms/
        - __init__.py
        - base_interfaces/
            - __init__.py
            - cost_function_base.py
            - demand_predictor_base.py
            - dispatch_base.py
            - matching_base.py
            - routing_base.py
            - stop_selector_base.py
            - user_acceptance_base.py
        - cost_functions/
            - __init__.py
            - multi_objective_cost.py
            - simple_cost.py
            - user_experience_weighted.py
        - dispatch/
            - __init__.py
            - fcfs_dispatch.py
            - ga_dispatch.py
            - heuristic_dispatch.py
            - naive_dispatch.py
            - rl_dispatch.py
        - matching/
            - __init__.py
            - batch_matching.py
            - exact_optimizer.py
            - heuristic_matching.py
        - plugin_loader.py
        - routing/
            - __init__.py
            - dijkstra_routing.py
            - genetic_routing.py
            - time_dependent_routing.py
        - user_acceptance_models/
            - __init__.py
            - logit_acceptance.py
            - ml_acceptance.py
    - analysis/
        - __init__.py
        - data_extraction.py
        - metrics_analysis.py
        - pareto_analysis.py
        - scenario_comparison.py
        - statistics.py
        - visualization.py
    - config/
        - __init__.py
        - config.py
        - parameters.py
    - core/
        - __init__.py
        - demand/
            - __init__.py
            - generators.py
            - manager.py
            - pattern_models.py
            - prebooking_manager.py
            - user_acceptance.py
            - user_profiles.py
        - events/
            - dispatcher.py
            - manager.py
            - processor.py
            - validator.py
        - hooks.py
        - logging_config.py
        - monitoring/
            - metrics_collector.py
            - resource_monitor.py
        - persistence/
            - state_store.py
        - simulation/
            - context.py
            - engine.py
            - orchestrator.py
        - state/
            - base.py
            - manager.py
            - workers.py
    - handlers/
        - __init__.py
        - base.py
        - request_handlers.py
        - system_handlers.py
        - vehicle_handlers.py
    - integration/
        - __init__.py
        - external_solvers.py
        - ml_integration.py
        - traffic_sim_integration.py
    - models/
        - base.py
        - event.py
        - experiment.py
        - location.py
        - metrics.py
        - passenger.py
        - request.py
        - route.py
        - simulation.py
        - state.py
        - stop.py
        - vehicle.py
    - network/
        - __init__.py
        - disruptions.py
        - graph_operations.py
        - multi_modal_integration.py
        - network_loader.py
        - transfer_points.py
    - runners/
        - __init__.py
        - experiment_runner.py
        - scenario_runner.py
        - study_runner.py
    - stops/
        - __init__.py
        - adaptive_stop_placement.py
        - stop_scoring.py
        - stop_selector.py
    - studies/
        - configs/
            - fleet_demand_study.yaml
            - user_behavior_study.yaml
    - utils/
        - __init__.py
        - caching.py
        - geometry.py
        - random_seed_manager.py
- requirements.txt
- run_study.py
- setup.py
- tools/


```

---

## Implementation Guidelines

- **Abstract Interfaces:**  
  Maintain strict interfaces for routing, dispatch, matching, cost, and user acceptance via `abc` abstractions. This ensures straightforward integration of new algorithms.

- **Configuration-Driven:**  
  Rely on external config files for scenario selection, parameter values, and algorithm choices. This speeds up experimentation and reduces code changes.

- **Logging & Debugging:**  
  Provide multi-level logging and event logs for debugging. Store system states for retrospective analysis.

- **Testing & Validation:**
  - Unit tests for each component.
  - Integration tests for scenario simulations.
  - Regression tests to ensure no breakage of existing features.
  - Performance profiling to identify and optimize bottlenecks.

---

## Example Use Cases

1. **Basic Scenario Execution:**  
   Run `scripts/run_experiment.py` with a simple scenario to test event flows and confirm dispatch logic.

2. **Parameter Sweeps:**
   Use `experiments/parameter_sweeps.py` to systematically vary fleet sizes or algorithm parameters. Analyze results with `analysis/scenario_comparison.py`.

3. **New RL-Based Dispatch (New in Version 1.1):**
   - Implement `RL_DispatchStrategy` in `algorithms/dispatch/rl_dispatch.py`.
   - Update scenario config to choose `RL_DispatchStrategy`.
   - Analyze changes in performance metrics against FCFS or GA-based dispatch strategies.

4. **User Acceptance Enhancement:**
   Integrate ML-based acceptance models in `algorithms/user_acceptance_models/ml_acceptance.py` and see how changes in user behavior affect KPIs.

---

## Best Practices for Research

- **Reproducibility:**  
  Record random seeds, version scenarios, and log parameters. Commit scenario configs to version control.

- **Statistical Validation:**
  Compare multiple runs with different seeds and apply significance tests to identify truly impactful changes in performance.

- **Comprehensive Reporting:**
  Use `analysis/visualization.py` for maps, charts, and route animations. Include these visuals in research reports or publications.

---

## Maintaining & Updating the Blueprint

1. **Versioning the Blueprint:**  
   - Keep a version number at the top of this document.
   - When making substantial architectural changes, increment the version and summarize modifications in the Change History section.

2. **Incremental Updates:**  
   After implementing new features or structural changes:
   - Update the relevant sections (e.g., add new dispatch algorithms, new directories, or modified scenario parameters).
   - Modify the Change History at the top to highlight what changed.
   - Commit the updated blueprint to version control alongside code changes.

3. **Providing the Blueprint to AI Assistants:**
   - Start new AI sessions by presenting the current version of the blueprint.
   - If AI-assisted coding is needed for a new feature, reference the relevant blueprint sections.
   - Summarize what has changed since the last session so the AI can adapt its suggestions accordingly.

---

## Future Extensions

- **Advanced ML Integration:**  
  Predict demand with advanced models and integrate deep RL for adaptive dispatch policies.

- **External Solvers & Tools:**  
  Connect to OR-Tools, Gurobi, or SUMO for richer optimization and traffic simulation capabilities.

- **Policy & Equity Analysis:**  
  Study how different fare structures, service areas, or energy-efficient fleets affect various user segments.

---

**End of Blueprint (Version 1.4)**

--- 
