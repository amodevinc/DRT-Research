from typing import Dict, Any, Optional, Type
import logging
import importlib
import traceback
from drt_sim.algorithms.base_interfaces.pricing_base import PricingModel
from drt_sim.models.request import Request
from drt_sim.models.vehicle import Vehicle
from drt_sim.models.route import Route
from drt_sim.config.config import PricingConfig, Currency

logger = logging.getLogger(__name__)

class PricingManager:
    """
    Manages pricing models and pricing operations in the DRT system.
    """
    
    def __init__(self, config: PricingConfig):
        """
        Initialize the pricing manager.
        
        Args:
            config: Configuration for pricing
        """
        self.config = config
        self.pricing_model = self._initialize_pricing_model()
        
        # Initialize metrics tracking
        self.pricing_metrics = {
            "total_requests_priced": 0,
            "average_price": 0.0,
            "min_price": float('inf'),
            "max_price": 0.0,
            "price_distribution": {},
            "by_time_of_day": {},
            "by_zone": {}
        }
        
        logger.info(f"Pricing manager initialized with model: {self.pricing_model.__class__.__name__}")
    
    def _initialize_pricing_model(self) -> PricingModel:
        """
        Initialize the pricing model based on configuration.
        
        Returns:
            PricingModel: The initialized pricing model
        """
        model_type = self.config.model_type
        model_params = self.config.model_params
        
        try:
            # Try to dynamically import the model class
            module_path = f"drt_sim.algorithms.pricing.{model_type}_pricing"
            class_name = "".join(word.capitalize() for word in model_type.split("_")) + "PricingModel"
            
            logger.debug(f"Attempting to import {class_name} from {module_path}")
            
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            
            # Initialize the model with config parameters
            model = model_class(model_params)
            logger.info(f"Successfully initialized {class_name} pricing model")
            
            return model
            
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to load pricing model '{model_type}': {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            
            # Fall back to a simple pricing model
            from drt_sim.algorithms.pricing.simple_pricing import SimplePricingModel
            logger.warning("Falling back to SimplePricingModel")
            return SimplePricingModel(model_params)
    
    def calculate_price(self, 
                        request: Request, 
                        vehicle: Optional[Vehicle] = None,
                        route: Optional[Route] = None,
                        service_attributes: Optional[Dict[str, Any]] = None,
                        currency: Optional[Currency] = None) -> float:
        """
        Calculate the price for a request using the pricing model.
        
        Args:
            request: The transportation request
            vehicle: Optional vehicle assigned to the request
            route: Optional route for the request
            service_attributes: Optional service attributes
            currency: Optional currency to use (defaults to the configured currency)
            
        Returns:
            float: The calculated price
        """
        try:
            # Calculate price using model
            price = self.pricing_model.calculate_price(
                request=request,
                vehicle=vehicle,
                route=route,
                service_attributes=service_attributes
            )
            
            # Convert currency if needed
            target_currency = currency or self.config.currency
            if request.currency and request.currency != target_currency.value:
                # Convert from the model's base currency to the requested currency
                source_currency = Currency(request.currency)
                price = self.config.convert_price(price, self.config.currency, source_currency)
            
            # Format the price with appropriate currency symbol
            formatted_price = self.config.format_price(price, target_currency)
            
            # Update the request with currency information
            request.estimated_price = price
            request.currency = target_currency.value
            request.formatted_price = formatted_price
            
            # Update metrics
            self._update_metrics(request, price, service_attributes)
            
            return round(price, 2)
            
        except Exception as e:
            logger.error(f"Error calculating price for request {request.id}: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            
            # Return default price in case of error
            default_price = self.config.default_price
            logger.warning(f"Using default price {default_price} due to error")
            
            # Set formatted default price
            request.estimated_price = default_price
            request.currency = self.config.currency.value
            request.formatted_price = self.config.format_price(default_price)
            
            return default_price
    
    def get_price_breakdown(self, 
                           request: Request,
                           vehicle: Optional[Vehicle] = None,
                           route: Optional[Route] = None,
                           service_attributes: Optional[Dict[str, Any]] = None,
                           currency: Optional[Currency] = None) -> Dict[str, float]:
        """
        Get a detailed breakdown of price components.
        
        Args:
            request: The transportation request
            vehicle: Optional vehicle assigned to the request
            route: Optional route for the request
            service_attributes: Optional service attributes
            currency: Optional currency to use (defaults to the configured currency)
            
        Returns:
            Dict[str, float]: Breakdown of price components
        """
        try:
            # Get price breakdown from model
            breakdown = self.pricing_model.get_price_breakdown(
                request=request,
                vehicle=vehicle,
                route=route,
                service_attributes=service_attributes
            )
            
            # Convert currency if needed
            target_currency = currency or self.config.currency
            if request.currency and request.currency != target_currency.value:
                source_currency = Currency(request.currency)
                
                # Convert each price in the breakdown
                for key, value in breakdown.items():
                    if isinstance(value, (int, float)):
                        breakdown[key] = self.config.convert_price(value, self.config.currency, source_currency)
            
            # Add currency information to the breakdown
            breakdown["currency"] = target_currency.value
            breakdown["currency_symbol"] = self.config.currency_symbol
            
            # Add formatted total price
            if "total" in breakdown:
                breakdown["formatted_total"] = self.config.format_price(breakdown["total"], target_currency)
            
            # Update request with the breakdown
            request.price_breakdown = breakdown
            
            return breakdown
            
        except Exception as e:
            logger.error(f"Error getting price breakdown for request {request.id}: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            
            # Return simple breakdown in case of error
            price = self.config.default_price
            formatted_price = self.config.format_price(price)
            
            breakdown = {
                "total": price,
                "base_fare": price,
                "currency": self.config.currency.value,
                "currency_symbol": self.config.currency_symbol,
                "formatted_total": formatted_price,
                "error": "Failed to get detailed breakdown"
            }
            
            request.price_breakdown = breakdown
            return breakdown
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get pricing metrics.
        
        Returns:
            Dict[str, Any]: Pricing metrics
        """
        return self.pricing_metrics
    
    def _update_metrics(self, 
                        request: Request, 
                        price: float, 
                        service_attributes: Optional[Dict[str, Any]] = None) -> None:
        """
        Update pricing metrics.
        
        Args:
            request: The transportation request
            price: The calculated price
            service_attributes: Optional service attributes
        """
        metrics = self.pricing_metrics
        
        # Update basic metrics
        metrics["total_requests_priced"] += 1
        
        # Update min/max price
        metrics["min_price"] = min(metrics["min_price"], price) if metrics["min_price"] != float('inf') else price
        metrics["max_price"] = max(metrics["max_price"], price)
        
        # Update average price
        total_requests = metrics["total_requests_priced"]
        metrics["average_price"] = ((metrics["average_price"] * (total_requests - 1)) + price) / total_requests
        
        # Update price distribution
        price_range = int(price / 5) * 5  # Round to nearest 5
        price_bucket = f"{price_range}-{price_range + 5}"
        if price_bucket not in metrics["price_distribution"]:
            metrics["price_distribution"][price_bucket] = 0
        metrics["price_distribution"][price_bucket] += 1
        
        # Update by time of day
        hour = request.request_time.hour
        hour_range = f"{hour:02d}:00-{(hour+1)%24:02d}:00"
        if hour_range not in metrics["by_time_of_day"]:
            metrics["by_time_of_day"][hour_range] = {"count": 0, "total": 0.0, "average": 0.0}
        
        metrics["by_time_of_day"][hour_range]["count"] += 1
        metrics["by_time_of_day"][hour_range]["total"] += price
        metrics["by_time_of_day"][hour_range]["average"] = (
            metrics["by_time_of_day"][hour_range]["total"] / 
            metrics["by_time_of_day"][hour_range]["count"]
        )
        
        # Update by zone if available
        if service_attributes and "zone_id" in service_attributes:
            zone_id = service_attributes["zone_id"]
            if zone_id not in metrics["by_zone"]:
                metrics["by_zone"][zone_id] = {"count": 0, "total": 0.0, "average": 0.0}
            
            metrics["by_zone"][zone_id]["count"] += 1
            metrics["by_zone"][zone_id]["total"] += price
            metrics["by_zone"][zone_id]["average"] = (
                metrics["by_zone"][zone_id]["total"] / 
                metrics["by_zone"][zone_id]["count"]
            )