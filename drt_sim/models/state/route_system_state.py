from dataclasses import dataclass
from typing import Dict, List, Any
from ..base import ModelBase
from ..route import Route, RouteStatus

@dataclass
class RouteSystemState(ModelBase):
    """Represents the state of the routing system"""
    active_routes: Dict[str, Route]
    routes_by_status: Dict[RouteStatus, List[str]]
    routes_by_vehicle: Dict[str, str]  # vehicle_id -> route_id
    passenger_route_mapping: Dict[str, str]  # passenger_id -> route_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_routes": {
                rid: route.to_dict() for rid, route in self.active_routes.items()
            },
            "routes_by_status": {
                status.value: rids for status, rids in self.routes_by_status.items()
            },
            "routes_by_vehicle": self.routes_by_vehicle,
            "passenger_route_mapping": self.passenger_route_mapping,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RouteSystemState':
        return cls(
            active_routes={
                rid: Route.from_dict(route) 
                for rid, route in data["active_routes"].items()
            },
            routes_by_status={
                RouteStatus(status): rids 
                for status, rids in data["routes_by_status"].items()
            },
            routes_by_vehicle=data["routes_by_vehicle"],
            passenger_route_mapping=data["passenger_route_mapping"],
        ) 