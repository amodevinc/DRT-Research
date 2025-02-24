# Request Matching and Dispatch Flow

This document describes the complete flow of handling a passenger request in the DRT system, from initial request to vehicle dispatch.

## Overview

The DRT system follows a modular event-driven architecture where each component handles specific responsibilities in the request-to-dispatch pipeline. The main components involved are:

1. Request Handler
2. Stop Handler
3. Matching Handler
4. Vehicle Handler
5. Route Handler

## Detailed Flow

### 1. Initial Request Processing
When a passenger makes a request:
1. `RequestHandler` receives the request and validates it
2. Creates a `DETERMINE_VIRTUAL_STOPS` event
3. Request enters `VALIDATED` state

### 2. Virtual Stop Determination
The `StopHandler` processes the `DETERMINE_VIRTUAL_STOPS` event:

1. **Stop Selection Strategy**:
   - Uses configured stop selector (Coverage-based or Demand-based)
   - Attempts to find suitable existing stops first
   - Creates virtual stops if needed

2. **Stop Assignment Creation**:
   - Creates `StopAssignment` with:
     - Origin and destination stops
     - Walking distances and times
     - Expected passenger arrival time at origin stop
   - Publishes `MATCH_REQUEST_TO_VEHICLE` event

### 3. Vehicle Matching
The `MatchingHandler` processes the `MATCH_REQUEST_TO_VEHICLE` event:

1. **Initial Processing**:
   - Retrieves stop assignment and request details
   - Gets list of available vehicles

2. **Matching Strategy**:
   - Uses configured matching strategy (e.g., Insertion or Auction)
   - Current implementation uses `InsertionAssigner`:
     - Evaluates all available vehicles in parallel
     - Calculates insertion costs for each vehicle
     - Selects best feasible insertion based on costs

3. **Assignment Creation**:
   - Creates `Assignment` with:
     - Matched vehicle
     - Updated route
     - Cost metrics
   - Immediately processes assignment and dispatches vehicle

### 4. Vehicle Dispatch
The `VehicleHandler` processes the `VEHICLE_DISPATCH_REQUEST` event:

1. Updates vehicle state and active route
2. Creates movement events for route segments
3. Vehicle immediately starts moving to pickup location

### 5. Limitations of Current Flow

The current implementation has several limitations:

1. **Immediate Dispatch**:
   - Vehicles are dispatched as soon as a match is made
   - No consideration of when the passenger will actually arrive
   - Can lead to excessive vehicle waiting time at pickup points

2. **Timing Issues**:
   - Passenger might take longer to reach the origin stop
   - Vehicle might wait unnecessarily
   - Potential timeout if passenger is significantly delayed

3. **Resource Inefficiency**:
   - Higher empty vehicle mileage
   - Increased vehicle dwell time at stops
   - Suboptimal vehicle utilization

4. **No Consideration For**:
   - Passenger walking time to origin stop
   - Historical behavior patterns

## Metrics and Monitoring

The system tracks several metrics to evaluate performance:

1. **Service Quality**:
   - `PASSENGER_WAIT_TIME`
   - `PASSENGER_RIDE_TIME`
   - `PASSENGER_WALK_TIME_TO_ORIGIN_STOP`
   - `PASSENGER_WALK_TIME_FROM_DESTINATION_STOP`
   - `PASSENGER_TOTAL_JOURNEY_TIME`
   - `PASSENGER_NO_SHOW`

2. **Vehicle Performance**:
   - `VEHICLE_UTILIZATION`
   - `VEHICLE_WAIT_TIME`
   - `VEHICLE_OCCUPIED_DISTANCE`
   - `VEHICLE_EMPTY_DISTANCE`
   - `VEHICLE_DWELL_TIME`
   - `VEHICLE_STOPS_SERVED`
   - `VEHICLE_PASSENGERS_SERVED`

## Research Capabilities

The system is designed for research with:

1. **Pluggable Components**:
   - Multiple stop selection strategies
   - Different matching algorithms
   - Various dispatch approaches

2. **Configuration Options**:
   - Adjustable parameters for each component
   - Different optimization objectives
   - Various constraint settings

3. **Data Collection**:
   - Comprehensive metrics
   - Detailed event logging
   - Performance analysis capabilities

## Potential Improvements

The current flow could be enhanced with:

1. **Smart Dispatch Timing**:
   - Calculate optimal dispatch times based on passenger arrival estimates
   - Consider traffic and time-of-day factors
   - Add buffer times for uncertainties

2. **Predictive Elements**:
   - Passenger behavior modeling

3. **Dynamic Optimization**:
   - Real-time route adjustments
   - Continuous system state optimization
   - Proactive vehicle positioning 