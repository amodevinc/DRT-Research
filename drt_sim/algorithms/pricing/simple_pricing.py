from typing import Dict, Any, Optional
import logging

from drt_sim.algorithms.base_interfaces.pricing_base import PricingModel
from drt_sim.models.request import Request
from drt_sim.models.vehicle import Vehicle
from drt_sim.models.route import Route

logger = logging.getLogger(__name__)

class SimplePricingModel(PricingModel):
    """
    A simple pricing model based on flat fee with optional distance-based pricing.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the simple pricing model.
        
        Args:
            config: Configuration parameters including:
                - flat_fare: Base flat fare for any trip
                - enable_distance_pricing: Whether to add distance-based pricing
                - per_km_rate: Rate per kilometer (only used if distance pricing is enabled)
                - min_price: Minimum price
                - max_price: Maximum price
        """
        super().__init__(config)
        
        # Set default values if not in config
        self.flat_fare = self.config.get("flat_fare", 5.0)
        self.enable_distance_pricing = self.config.get("enable_distance_pricing", False)
        self.per_km_rate = self.config.get("per_km_rate", 0.5) if self.enable_distance_pricing else 0.0
        self.min_price = self.config.get("min_price", 3.0)
        self.max_price = self.config.get("max_price", 50.0)
        
        logger.info(f"Initialized SimplePricingModel with flat_fare={self.flat_fare}, "
                   f"distance_pricing={'enabled' if self.enable_distance_pricing else 'disabled'}")
    
    def calculate_price(self, 
                        request: Request, 
                        vehicle: Optional[Vehicle] = None,
                        route: Optional[Route] = None,
                        service_attributes: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate price based on flat fee and optionally distance.
        
        Args:
            request: The transportation request
            vehicle: Optional vehicle assigned to the request
            route: Optional route for the request
            service_attributes: Optional service attributes
            
        Returns:
            float: The calculated price
        """
        # Start with flat fare
        price = self.flat_fare
        
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
        
        logger.debug(f"Calculated price for request {request.id}: {price:.2f} "
                    f"(flat_fare={self.flat_fare})")
        
        return price
    
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
        breakdown = {
            "total": self.flat_fare,
            "flat_fare": self.flat_fare,
            "distance_charge": 0.0,
            "distance_km": 0.0
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