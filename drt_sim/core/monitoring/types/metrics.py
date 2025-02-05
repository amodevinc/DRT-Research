from enum import Enum

class MetricName(Enum):
    PASSENGER_WAIT_TIME = 'passenger.wait_time'
    PASSENGER_RIDE_TIME = 'passenger.ride_time'
    PASSENGER_WALK_TIME_TO_ORIGIN_STOP = 'passenger.walk_time_to_origin_stop'
    PASSENGER_WALK_TIME_FROM_DESTINATION_STOP = 'passenger.walk_time_from_destination_stop'
    PASSENGER_TOTAL_JOURNEY_TIME = 'passenger.total_journey_time'
    PASSENGER_NO_SHOW = 'passenger.no_show'

    REQUEST_RECEIVED = 'request.received'
    REQUEST_ASSIGNED = 'request.assigned'
    REQUEST_REJECTED = 'request.rejected'

    VEHICLE_UTILIZATION = 'vehicle.utilization'
    VEHICLE_WAIT_TIME = 'vehicle.wait_time'
    VEHICLE_OCCUPIED_DISTANCE = 'vehicle.occupied_distance'
    VEHICLE_EMPTY_DISTANCE = 'vehicle.empty_distance'
    VEHICLE_DWELL_TIME = 'vehicle.dwell_time'
    VEHICLE_STOPS_SERVED = 'vehicle.stops_served'

    STOP_OCCUPANCY = 'stop.occupancy'

    SIMULATION_STEP_DURATION = 'simulation.step_duration'
    SIMULATION_TOTAL_STEPS = 'simulation.total_steps'
    SIMULATION_EVENT_PROCESSING_TIME = 'simulation.event_processing_time'
    SIMULATION_REPLICATION_TIME = 'simulation.replication_time'

    MATCHING_SUCCESS_RATE = 'matching.success_rate'
    MATCHING_FAILURE_REASON = 'matching.failure_reason'

    ROUTE_COMPLETION_TIME = 'route.completion_time'
    ROUTE_DEVIATION = 'route.deviation'

    SERVICE_VIOLATIONS = 'service.violations'
    SERVICE_ON_TIME_RATE = 'service.on_time_rate'
    SERVICE_CAPACITY_UTILIZATION = 'service.capacity_utilization'

    DEMAND_GENERATED = 'demand.generated'
