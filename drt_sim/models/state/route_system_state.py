from dataclasses import dataclass
from typing import Dict, List, Any
from ..base import ModelBase
from ..route import Route, RouteStatus

@dataclass
class RouteSystemState(ModelBase):
    """Represents the state of the routing system"""
    routes_by_status: Dict[RouteStatus, List[str]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "routes_by_status": {
                status.value: rids for status, rids in self.routes_by_status.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RouteSystemState':
        return cls(
            routes_by_status={
                RouteStatus(status): rids 
                for status, rids in data["routes_by_status"].items()
            }
        ) 