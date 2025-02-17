# DRT Research Platform Documentation

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
## Overview

The DRT (Demand Responsive Transportation) Research Platform is a comprehensive simulation framework designed for conducting research experiments in transportation systems. The platform supports various dispatch strategies, demand patterns, and vehicle configurations while providing robust tools for experiment management and analysis.

### Key Features
- Hierarchical study management
- Parameter sweep capabilities
- Distributed execution support
- Comprehensive metrics collection
- MLflow experiment tracking
- Scenario comparison tools
- State management and persistence

## System Architecture

### Core Components

```
Study (StudyRunner)
└── Experiments (ExperimentRunner)
    └── Scenarios (ScenarioRunner)
        └── Simulations
```

Each level of the hierarchy serves a specific purpose:

1. **StudyRunner**: Top-level coordinator
   - Manages overall study execution
   - Handles experiment organization
   - Coordinates MLflow tracking
   - Manages distributed execution

2. **ExperimentRunner**: Mid-level executor
   - Runs multiple scenarios
   - Collects experiment-level metrics
   - Manages experiment state

3. **ScenarioRunner**: Low-level executor
   - Executes individual simulations
   - Handles replications
   - Collects detailed metrics
   - Manages simulation state
   - Handles core simulation logic
