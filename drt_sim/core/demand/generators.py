from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Tuple
import random
import pandas as pd
from drt_sim.models.event import Event, EventType, EventPriority
from drt_sim.models.request import Request, RequestType, RequestStatus
from drt_sim.models.location import Location
from drt_sim.config.config import (
    DemandConfig, CSVDemandGeneratorConfig, RandomDemandGeneratorConfig
)
from drt_sim.models.location import haversine_distance
from shapely.geometry import Point, Polygon
import logging
from math import cos, radians
logger = logging.getLogger(__name__)

class BaseDemandGenerator(ABC):
    """Abstract base class for demand generators"""
    def __init__(self, config: DemandConfig):
        self.config = config
        self.request_counter = 0
        self.event_counter = 0

    @abstractmethod
    def generate(
        self,
        start_time: datetime,
        end_time: datetime | timedelta,
        time_scale_factor: float = 1.0
    ) -> List[Event]:
        """Generate demand events for the simulation period"""
        pass

    def _create_request(
        self,
        request_time: datetime,
        origin: Location,
        destination: Location
    ) -> Request:
        """Create a new request"""
        self.request_counter += 1
        distance = haversine_distance(origin, destination)
        if distance < 200:
            return None
        return Request(
            id=f"R{self.request_counter}",
            user_id=f"U{self.request_counter}",
            passenger_id=f"P{self.request_counter}",
            origin=origin,
            destination=destination,
            request_time=request_time,
            type=RequestType.IMMEDIATE,
            status=RequestStatus.PENDING
        )

    def _create_event(self, request: Request) -> Event:
        """Create event from request with a unique event ID"""
        self.event_counter += 1
        event_id = f"E{self.event_counter}_REQ_{request.id}"
        
        return Event(
            id=event_id,
            event_type=EventType.REQUEST_RECEIVED,
            priority=EventPriority.HIGH,  # Requests are high priority
            timestamp=request.request_time,
            data={"request": request},
            request_id=request.id
        )

class RandomDemandGenerator(BaseDemandGenerator):
    """Generates random demand based on configured patterns"""
    def __init__(self, config: RandomDemandGeneratorConfig):
        super().__init__(config)
        self.config: RandomDemandGeneratorConfig = config

    def generate(
        self,
        start_time: datetime,
        end_time: datetime | timedelta,
        time_scale_factor: float = 1.0
    ) -> List[Event]:
        events = []
        current_time = start_time
        
        # Handle end_time as either datetime or timedelta
        if isinstance(end_time, timedelta):
            actual_end_time = start_time + end_time
        else:
            actual_end_time = end_time
        
        # Adjust request rate for time scaling
        scaled_demand = self.config.demand_level * time_scale_factor
        
        while current_time < actual_end_time:
            # Time until next request (exponential distribution)
            hours_until_next = random.expovariate(scaled_demand)
            current_time += timedelta(hours=hours_until_next)
            
            if current_time < actual_end_time:
                origin = self._generate_location()
                destination = self._generate_location()
                
                request = self._create_request(
                    request_time=current_time,
                    origin=origin,
                    destination=destination
                )
                if request:
                    events.append(self._create_event(request))
        
        return events

    def _generate_location(self) -> Location:
        """Generate location based on spatial distribution"""
        if self.config.spatial_distribution == "uniform":
            return self._uniform_location()
        elif self.config.spatial_distribution == "hotspot":
            return self._hotspot_location()
        else:
            return self._uniform_location()

    def _uniform_location(self) -> Location:
        """Generate uniform random location within service area"""
        (min_lat, min_lon), (max_lat, max_lon) = self.config.service_area
        return Location(
            lat=random.uniform(min_lat, max_lat),
            lon=random.uniform(min_lon, max_lon)
        )

    def _hotspot_location(self) -> Location:
        """Generate location considering hotspots"""
        if not self.config.hotspots:
            return self._uniform_location()
            
        # Randomly choose between hotspot and uniform
        if random.random() < 0.7:  # 70% chance of hotspot
            hotspot = random.choice(self.config.hotspots)
            center = hotspot["location"]
            intensity = hotspot["intensity"]
            
            # Generate normally distributed offset based on intensity
            radius = 0.01 / intensity  # Higher intensity = smaller radius
            lat = random.gauss(center[0], radius)
            lon = random.gauss(center[1], radius)
            
            return Location(lat=lat, lon=lon)
        else:
            return self._uniform_location()

    def cleanup(self):
        """Cleanup resources"""
        logger.info("Random demand generator cleaned up")

class CSVDemandGenerator(BaseDemandGenerator):
    """Generates demand from multiple CSV files with weights"""
    def __init__(self, config: CSVDemandGeneratorConfig):
        super().__init__(config)
        self.config: CSVDemandGeneratorConfig = config
        self.dataframes = []
        self._load_data()

    def _is_point_valid(self, lat: float, lon: float, service_area_polygon: List[Tuple[float, float]], buffer_distance: float = 100) -> bool:
        """
        Check if a point is within the service area polygon or within buffer_distance meters of it.
        
        Args:
            lat: Latitude of the point
            lon: Longitude of the point
            service_area_polygon: List of (lat, lon) tuples defining the polygon
            buffer_distance: Distance in meters to buffer the polygon
            
        Returns:
            bool: True if point is valid, False otherwise
        """
        if not service_area_polygon:
            return True
            
        try:
            # Create point and polygon using (lon, lat) order for Shapely
            point = Point(lon, lat)
            polygon = Polygon([(lat, lon) for lat, lon in service_area_polygon])
            
            # Add some debug logging
            
            # If point is inside polygon, return True
            if polygon.contains(point):
                return True
                
            # If point is within buffer distance, return True
            # Convert buffer distance from meters to degrees (approximate at the equator)
            # Different scaling for lat/lon due to projection
            lat_buffer = buffer_distance / 111000  # 1 degree lat ≈ 111km
            lon_buffer = buffer_distance / (111000 * cos(radians(lat)))  # Adjust for latitude
            buffered_polygon = polygon.buffer(max(lat_buffer, lon_buffer))
            
            result = buffered_polygon.contains(point)
            return result
            
        except Exception as e:
            logger.error(f"Error in point validation: {str(e)}. Point: ({lat}, {lon})")
            # If there's an error in validation, we'll be conservative and include the point
            return True

    def _load_data(self):
        """Load and prepare data from multiple CSV files"""
        try:
            for file_config in self.config.files:
                logger.info(f"Loading CSV file: {file_config.file_path}")
                df = pd.read_csv(file_config.file_path)
                
                # Convert timestamp column
                df[self.config.columns["request_time"]] = pd.to_datetime(
                    df[self.config.columns["request_time"]],
                    format=self.config.datetime_format
                )
                
                # Apply time filters if specified
                if file_config.start_time:
                    start = pd.to_datetime(file_config.start_time, format=self.config.datetime_format)
                    df = df[df[self.config.columns["request_time"]] >= start]
                
                if file_config.end_time:
                    end = pd.to_datetime(file_config.end_time, format=self.config.datetime_format)
                    df = df[df[self.config.columns["request_time"]] <= end]
                
                # Apply service area polygon filter if specified
                service_area = self.config.service_area_polygon
                if service_area:
                    # Create mask for valid pickup locations
                    pickup_mask = df.apply(
                        lambda row: self._is_point_valid(
                            row[self.config.columns["pickup_lat"]],
                            row[self.config.columns["pickup_lon"]],
                            service_area
                        ),
                        axis=1
                    )
                    
                    # Create mask for valid dropoff locations
                    dropoff_mask = df.apply(
                        lambda row: self._is_point_valid(
                            row[self.config.columns["dropoff_lat"]],
                            row[self.config.columns["dropoff_lon"]],
                            service_area
                        ),
                        axis=1
                    )
                    
                    # Filter dataframe to only include rows where both pickup and dropoff are valid
                    df = df[pickup_mask & dropoff_mask]
                    
                    logger.info(
                        f"Filtered {len(pickup_mask) - len(df)} requests outside service area "
                        f"for file {file_config.file_path}"
                    )
                
                # Store weight with the dataframe
                self.dataframes.append((df, file_config.weight))
                
                logger.info(
                    f"Loaded {len(df)} records from {file_config.file_path} "
                    f"with weight {file_config.weight}"
                )
                
        except Exception as e:
            logger.error(f"Error loading CSV data: {str(e)}")
            raise

    def generate(
        self,
        start_time: datetime,
        end_time: datetime | timedelta,
        time_scale_factor: float = 1.0
    ) -> List[Event]:
        try:
            all_events = []
            
            # Handle end_time as either datetime or timedelta
            if isinstance(end_time, timedelta):
                actual_end_time = start_time + end_time
            else:
                actual_end_time = end_time
                
            # Convert datetime inputs to pandas Timestamp for consistent comparison
            pd_start_time = pd.Timestamp(start_time)
            pd_end_time = pd.Timestamp(actual_end_time)
            
            for df, weight in self.dataframes:
                # Filter data for simulation period
                mask = (
                    (df[self.config.columns["request_time"]] >= pd_start_time) &
                    (df[self.config.columns["request_time"]] <= pd_end_time)
                )
                period_data = df[mask].copy()
                
                # Apply time scaling
                if time_scale_factor != 1.0:
                    # Convert to timedelta first, scale it, then add back to start_time
                    time_diff = (period_data[self.config.columns["request_time"]] - pd_start_time)
                    scaled_diff = time_diff / time_scale_factor
                    period_data[self.config.columns["request_time"]] = pd_start_time + scaled_diff
                
                # Generate events for this file
                file_events = []
                for _, row in period_data.iterrows():
                    # Convert pandas Timestamp back to python datetime for Request object
                    request_time = row[self.config.columns["request_time"]].to_pydatetime()
                    
                    request = self._create_request(
                        request_time=request_time,
                        origin=Location(
                            lat=row[self.config.columns["pickup_lat"]],
                            lon=row[self.config.columns["pickup_lon"]]
                        ),
                        destination=Location(
                            lat=row[self.config.columns["dropoff_lat"]],
                            lon=row[self.config.columns["dropoff_lon"]]
                        )
                    )
                    if request:
                        file_events.append(self._create_event(request))
                
                # Apply weight to determine how many events to include
                if self.config.combine_method == "weighted_sum":
                    num_events = int(len(file_events) * weight)
                    file_events = random.sample(file_events, num_events)
                
                all_events.extend(file_events)
            
            # Sort all events by time
            all_events.sort(key=lambda x: x.timestamp)
            
            return all_events
            
        except Exception as e:
            import traceback
            logger.error(f"Error generating events: {str(e)}\n{traceback.format_exc()}")
            raise
    def cleanup(self):
        """Cleanup resources"""
        self.dataframes = []
        logger.info("CSV demand generator cleaned up")