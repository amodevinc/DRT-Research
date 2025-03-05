from dataclasses import dataclass, field
from typing import Dict, List, Any
from ..base import ModelBase
from ..vehicle import VehicleState, VehicleStatus

@dataclass
class VehicleSystemState(ModelBase):
    """Represents the state of the vehicle system"""
    vehicles: Dict[str, VehicleState]
    vehicles_by_status: Dict[VehicleStatus, List[str]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vehicles": {
                vid: vstate.to_dict() for vid, vstate in self.vehicles.items()
            },
            "vehicles_by_status": {
                status.value: vids for status, vids in self.vehicles_by_status.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VehicleSystemState':
        return cls(
            vehicles={
                vid: VehicleState.from_dict(vstate) 
                for vid, vstate in data["vehicles"].items()
            },
            vehicles_by_status={
                VehicleStatus(status): vids 
                for status, vids in data["vehicles_by_status"].items()
            }
        ) 