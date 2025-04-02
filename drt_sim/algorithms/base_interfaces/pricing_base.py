from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from drt_sim.models.request import Request
from drt_sim.models.vehicle import Vehicle
from drt_sim.models.route import Route

logger = logging.getLogger(__name__)

class PricingModel(ABC):
    """
    Abstract base class for pricing models.
    
    This class defines the interface that all pricing models
    must implement.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the pricing model.
        
        Args:
            config: Configuration parameters
        """
        logger.info("Initializing PricingModel")
        logger.info(f"Received config: {config}")
        self.config = config or {}
        logger.info(f"Final config: {self.config}")
    
    @abstractmethod
    def calculate_price(self, 
                        request: Request, 
                        vehicle: Optional[Vehicle] = None,
                        route: Optional[Route] = None,
                        service_attributes: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate the price for a given request.
        
        Args:
            request: The transportation request
            vehicle: Optional vehicle assigned to the request
            route: Optional route for the request
            service_attributes: Optional service attributes
            
        Returns:
            float: The calculated price
        """
        pass
    
    def adjust_price(self, 
                    base_price: float, 
                    factors: Dict[str, float]) -> float:
        """
        Adjust a base price using various factors.
        
        Args:
            base_price: The base price to adjust
            factors: Dictionary of adjustment factors
            
        Returns:
            float: The adjusted price
        """
        logger.info(f"Adjusting base price: {base_price}")
        logger.info(f"Adjustment factors: {factors}")
        
        # Default implementation that can be overridden
        adjusted_price = base_price
        for factor_name, factor_value in factors.items():
            if factor_name in self.config.get("adjustments", {}):
                weight = self.config["adjustments"][factor_name]
                logger.info(f"Applying factor {factor_name} with weight {weight} and value {factor_value}")
                adjusted_price *= (1 + factor_value * weight)
        
        logger.info(f"Final adjusted price: {adjusted_price}")
        return max(adjusted_price, 0.0)  # Ensure price is non-negative
    
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
        logger.info("Calculating price breakdown")
        logger.info(f"Request: {request}")
        if vehicle:
            logger.info(f"Vehicle: {vehicle}")
        if route:
            logger.info(f"Route: {route}")
        if service_attributes:
            logger.info(f"Service attributes: {service_attributes}")
            
        # Default implementation that can be overridden
        breakdown = {
            "total": self.calculate_price(request, vehicle, route, service_attributes),
            "base_fare": 0.0,
            "distance_charge": 0.0,
            "time_charge": 0.0,
            "surge_charge": 0.0,
            "discounts": 0.0
        }
        
        logger.info(f"Price breakdown: {breakdown}")
        return breakdown