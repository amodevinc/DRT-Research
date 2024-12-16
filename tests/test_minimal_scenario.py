# tests/test_minimal_scenario.py
import pytest
from datetime import datetime, timedelta
from pathlib import Path
import yaml

from drt_sim.core.simulation_engine import SimulationEngine, SimulationStatus
from drt_sim.core.event_manager import EventType, SimulationEvent
from drt_sim.models.vehicle import Vehicle, VehicleState, VehicleStatus
from drt_sim.models.passenger import Passenger, PassengerStatus
from drt_sim.models.request import Request, RequestStatus
from drt_sim.models.location import Location
from drt_sim.config.parameters import SimulationParameters

class TestMinimalScenario:
    @pytest.fixture
    def minimal_config(self):
        """Create a minimal configuration for testing"""
        return {
            'simulation': {
                'start_time': datetime(2024, 1, 1, 8, 0),
                'end_time': datetime(2024, 1, 1, 9, 0),
                'time_step': timedelta(minutes=1),
                'warm_up_period': timedelta(minutes=5),
                'cool_down_period': timedelta(minutes=5),
                'random_seed': 42
            },
            'vehicle': {
                'fleet_size': 1,
                'vehicle_capacity': 4,
                'depot_locations': [
                    Location(lat=52.5200, lon=13.4050)  # Example location
                ]
            }
        }

    @pytest.fixture
    def simulation_engine(self, minimal_config):
        """Create and initialize simulation engine"""
        config = SimulationParameters(**minimal_config)
        engine = SimulationEngine(config)
        engine.initialize()
        return engine

    def test_minimal_scenario(self, simulation_engine, minimal_config):
        """Test a minimal scenario with one vehicle and one passenger"""
        # 1. Setup initial state
        depot_location = minimal_config['vehicle']['depot_locations'][0]
        pickup_location = Location(lat=52.5300, lon=13.4150)
        dropoff_location = Location(lat=52.5400, lon=13.4250)

        # Create vehicle
        vehicle = Vehicle(
            id="vehicle_1",
            type="standard",
            capacity=4,
            depot_location=depot_location,
            current_state=VehicleState(
                vehicle_id="vehicle_1",
                current_location=depot_location,
                status=VehicleStatus.IDLE
            )
        )

        # Create passenger and request
        passenger = Passenger(
            id="passenger_1",
            pickup_location=pickup_location,
            dropoff_location=dropoff_location,
            requested_pickup_time=minimal_config['simulation']['start_time'] + timedelta(minutes=15),
            status=PassengerStatus.WAITING
        )

        request = Request(
            id="request_1",
            passenger_id=passenger.id,
            pickup_location=pickup_location,
            dropoff_location=dropoff_location,
            requested_time=minimal_config['simulation']['start_time'] + timedelta(minutes=15),
            status=RequestStatus.PENDING
        )

        # 2. Add entities to simulation state
        simulation_engine.state.add_vehicle(vehicle)
        simulation_engine.state.add_passenger(passenger)
        simulation_engine.state.add_request(request)

        # 3. Schedule initial events
        # Request arrival event
        simulation_engine.event_manager.schedule_event(
            SimulationEvent(
                event_type=EventType.REQUEST_ARRIVAL,
                timestamp=request.requested_time,
                priority=1,
                data={'request_id': request.id}
            )
        )

        # 4. Run simulation
        simulation_engine.run()

        # 5. Verify results
        final_state = simulation_engine.state

        # Verify vehicle state
        final_vehicle_state = final_state.vehicle_states[vehicle.id]
        assert final_vehicle_state.status == VehicleStatus.IDLE
        assert final_vehicle_state.current_location == depot_location

        # Verify passenger state
        final_passenger = final_state.passengers[passenger.id]
        assert final_passenger.status == PassengerStatus.COMPLETED

        # Verify request state
        final_request = final_state.requests[request.id]
        assert final_request.status == RequestStatus.COMPLETED

        # Verify events were processed in correct order
        event_sequence = [
            (event.event_type, event.timestamp) 
            for event in simulation_engine.event_manager.get_processed_events()
        ]

        expected_sequence = [
            (EventType.SIMULATION_START, minimal_config['simulation']['start_time']),
            (EventType.REQUEST_ARRIVAL, request.requested_time),
            (EventType.VEHICLE_DEPARTURE, request.requested_time + timedelta(minutes=1)),
            (EventType.PASSENGER_PICKUP, request.requested_time + timedelta(minutes=5)),
            (EventType.PASSENGER_DROPOFF, request.requested_time + timedelta(minutes=15)),
            (EventType.SIMULATION_END, minimal_config['simulation']['end_time'])
        ]

        assert event_sequence == expected_sequence

        # Verify metrics
        metrics = simulation_engine.state.get_metrics()
        assert metrics['total_requests'] == 1
        assert metrics['completed_requests'] == 1
        assert metrics['total_distance'] > 0
        assert metrics['average_wait_time'] > 0

def test_error_handling(self, simulation_engine):
    """Test error handling in minimal scenario"""
    # Test invalid vehicle
    with pytest.raises(ValueError):
        simulation_engine.state.add_vehicle(None)

    # Test invalid request
    with pytest.raises(ValueError):
        simulation_engine.state.add_request(None)

    # Test duplicate request
    request = Request(
        id="duplicate_request",
        passenger_id="passenger_1",
        pickup_location=Location(lat=0, lon=0),
        dropoff_location=Location(lat=1, lon=1),
        requested_time=datetime.now(),
        status=RequestStatus.PENDING
    )
    simulation_engine.state.add_request(request)
    with pytest.raises(ValueError):
        simulation_engine.state.add_request(request)

def test_simulation_interruption(self, simulation_engine):
    """Test simulation pause and resume functionality"""
    simulation_engine.run_until(
        simulation_engine.config.start_time + timedelta(minutes=30)
    )
    simulation_engine.pause()
    assert simulation_engine.status == SimulationStatus.PAUSED

    current_time = simulation_engine.current_time
    simulation_engine.resume()
    simulation_engine.run()
    assert simulation_engine.current_time > current_time