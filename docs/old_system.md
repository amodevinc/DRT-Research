# DRT System Documentation

## System Overview

This codebase implements a Demand-Responsive Transit (DRT) system focused on vehicle dispatching and route optimization. The system was initially built as an operational prototype rather than a research platform, which has led to certain architectural limitations for conducting systematic research on dispatching strategies and system parameters.

## System Architecture

### Core Components

1. **DRT System (`drt_system.py`)**
   - Central coordinator managing the interaction between all system components
   - Handles request processing, assignment generation, and system state management
   - Implements basic batch processing for demand handling

2. **Cost Function (`cost_function.py`)**
   - Implements the vehicle assignment strategy
   - Evaluates feasibility and scores potential assignments
   - Uses a weighted multi-objective function considering:
     - Vehicle occupancy
     - Passenger impact
     - Vehicle access time
     - Wait times
     - Ride times
     - Vehicle proximity
     - Stop clustering
     - Detour factors

3. **Supporting Managers**
   - Route Manager: Handles route calculations and modifications
   - Vehicle Manager: Manages vehicle states and movements
   - Stop Manager: Handles virtual and physical stops
   - Passenger Manager: Tracks passenger states and requirements
   - Call Manager: Processes incoming ride requests
   - Report Manager: Collects system performance metrics

## Research Limitations

### 1. Rigid Architecture

- The system uses a tightly coupled architecture where components are heavily interdependent
- Difficult to isolate and modify individual components without affecting others
- Limited ability to plug in alternative implementations of key components

### 2. Dispatching Strategy Constraints

- The dispatching strategy is hardcoded within the cost function
- No framework for easily testing different dispatching algorithms
- Weights and scoring mechanisms are not easily configurable
- Limited support for comparing different strategies in controlled experiments

### 3. Simulation Limitations

- Basic discrete event simulation capabilities
- No proper simulation clock or event queue management
- Limited control over temporal aspects of the simulation
- Difficult to reproduce exact scenarios or control random elements

### 4. Parameter Tuning Challenges

- System parameters are scattered across multiple components
- No centralized configuration management
- Limited ability to perform systematic parameter sweeps
- Difficult to track the impact of parameter changes

### 5. Metrics and Analysis

- Basic metrics collection focused on operational aspects
- Limited support for research-oriented metrics
- No built-in statistical analysis capabilities
- Difficult to perform comparative analysis between different strategies

### 6. Experimental Control

- No proper framework for conducting controlled experiments
- Limited ability to isolate variables
- Difficult to ensure fair comparisons between different approaches
- No support for scenario generation or demand pattern modeling

## Research Requirements Not Met

1. **Strategy Comparison Framework**
   - Need for modular strategy implementation
   - Ability to swap strategies at runtime
   - Controlled environment for fair comparison
   - Standardized metrics for evaluation

2. **Parameter Experimentation**
   - Centralized parameter management
   - Automated parameter sweeping
   - Impact analysis tools
   - Configuration version control

3. **Simulation Control**
   - Proper discrete event simulation engine
   - Reproducible scenarios
   - Controlled randomness
   - Time manipulation capabilities

4. **Analysis Capabilities**
   - Comprehensive metrics collection
   - Statistical analysis tools
   - Performance visualization
   - Strategy effectiveness measures

## Goals for New System

1. **Modular Architecture**
   - Clear separation of concerns
   - Plugin system for strategies
   - Dependency injection
   - Interface-based design

2. **Research Framework**
   - Experiment configuration system
   - Scenario management
   - Automated testing framework
   - Results analysis tools

3. **Simulation Engine**
   - Proper discrete event simulation
   - Time control mechanisms
   - Reproducibility features
   - Scenario generation tools

4. **Configuration Management**
   - Centralized parameter management
   - Version control for configurations
   - Parameter validation
   - Impact tracking

5. **Analysis Tools**
   - Comprehensive metrics collection
   - Statistical analysis capabilities
   - Visualization tools
   - Comparative analysis features

This documentation is intended to guide the development of a new research-oriented DRT system that addresses these limitations and provides better support for systematic research on dispatching strategies and system optimization. 