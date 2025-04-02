from typing import Dict, Any, Optional
import logging
from datetime import datetime, time

from drt_sim.algorithms.base_interfaces.pricing_base import PricingModel
from drt_sim.models.request import Request
from drt_sim.models.vehicle import Vehicle
from drt_sim.models.route import Route

logger = logging.getLogger(__name__)

class DynamicPricingModel(PricingModel):
    """
    A dynamic pricing model that uses different flat rates for peak vs off-peak hours,
    with optional distance-based pricing.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the dynamic pricing model.
        
        Args:
            config: Configuration parameters including:
                - peak_flat_fare: Flat fare during peak hours
                - off_peak_flat_fare: Flat fare during off-peak hours
                - enable_distance_pricing: Whether to add distance-based pricing
                - per_km_rate: Rate per kilometer (only used if distance pricing is enabled)
                - peak_hours: Dictionary with start and end times for peak hours
                - min_price: Minimum price
                - max_price: Maximum price
        """
        super().__init__(config)
        
        # Set default values
        self.peak_flat_fare = self.config.get("peak_flat_fare", 7.0)
        self.off_peak_flat_fare = self.config.get("off_peak_flat_fare", 5.0)
        self.enable_distance_pricing = self.config.get("enable_distance_pricing", False)
        self.per_km_rate = self.config.get("per_km_rate", 0.5) if self.enable_distance_pricing else 0.0
        self.min_price = self.config.get("min_price", 3.0)
        self.max_price = self.config.get("max_price", 50.0)
        
        # Set peak hours (default: 7-9 AM and 5-7 PM)
        peak_hours = self.config.get("peak_hours", [
            [7, 9],  # 7 AM to 9 AM
            [17, 19]  # 5 PM to 7 PM
        ])
        
        # Convert hour integers to time objects
        self.peak_hours = []
        for start_hour, end_hour in peak_hours:
            start = time(start_hour, 0)  # Start at minute 0
            end = time(end_hour, 0)      # End at minute 0
            self.peak_hours.append((start, end))
        
        logger.info(f"Initialized DynamicPricingModel with peak_fare={self.peak_flat_fare}, "
                   f"off_peak_fare={self.off_peak_flat_fare}, "
                   f"distance_pricing={'enabled' if self.enable_distance_pricing else 'disabled'}")
    
    def calculate_price(self, 
                        request: Request, 
                        vehicle: Optional[Vehicle] = None,
                        route: Optional[Route] = None,
                        service_attributes: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate price based on time of day and optionally distance.
        
        Args:
            request: The transportation request
            vehicle: Optional vehicle assigned to the request
            route: Optional route for the request
            service_attributes: Optional service attributes
            
        Returns:
            float: The calculated price
        """
        # Start with appropriate flat fare based on time
        price = self._get_flat_fare(request.request_time)
        
        # Add distance-based pricing if enabled
        if self.enable_distance_pricing:
            distance_km = 0.0
            if service_attributes:
                distance_km = service_attributes.get("distance", 0.0) / 1000.0  # Convert m to km
            elif route:
                distance_km = route.total_distance / 1000.0  # Convert m to km
            
            distance_charge = distance_km * self.per_km_rate
            price += distance_charge
        
        # Apply min/max constraints
        price = max(min(price, self.max_price), self.min_price)
        
        logger.debug(f"Calculated dynamic price for request {request.id}: {price:.2f} "
                    f"(flat_fare={price - (distance_charge if self.enable_distance_pricing else 0):.2f})")
        
        return price
    
    def _get_flat_fare(self, request_time: datetime) -> float:
        """
        Get the appropriate flat fare based on time of day.
        
        Args:
            request_time: The time to get the fare for
            
        Returns:
            float: The flat fare for the given time
        """
        request_time_only = request_time.time()
        
        for start_time, end_time in self.peak_hours:
            # Handle periods that cross midnight
            if start_time > end_time:
                if request_time_only >= start_time or request_time_only <= end_time:
                    return self.peak_flat_fare
            else:
                if start_time <= request_time_only <= end_time:
                    return self.peak_flat_fare
        
        return self.off_peak_flat_fare
    
    def get_price_breakdown(self, 
                           request: Request,
                           vehicle: Optional[Vehicle] = None,
                           route: Optional[Route] = None,
                           service_attributes: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Get a breakdown of the price components.
        
        Args:
            request: The transportation request
            vehicle: Optional vehicle assigned to the request
            route: Optional route for the request
            service_attributes: Optional service attributes
            
        Returns:
            Dict[str, float]: Price components
        """
        # Get base flat fare
        flat_fare = self._get_flat_fare(request.request_time)
        
        breakdown = {
            "total": flat_fare,
            "flat_fare": flat_fare,
            "distance_charge": 0.0,
            "distance_km": 0.0,
            "is_peak": flat_fare == self.peak_flat_fare
        }
        
        # Add distance-based pricing if enabled
        if self.enable_distance_pricing:
            distance_km = 0.0
            if service_attributes:
                distance_km = service_attributes.get("distance", 0.0) / 1000.0
            elif route:
                distance_km = route.total_distance / 1000.0
            
            distance_charge = distance_km * self.per_km_rate
            breakdown["distance_charge"] = distance_charge
            breakdown["distance_km"] = distance_km
            breakdown["total"] += distance_charge
        
        # Apply min/max constraints
        breakdown["total"] = max(min(breakdown["total"], self.max_price), self.min_price)
        
        return breakdown