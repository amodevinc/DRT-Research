from dataclasses import dataclass
from typing import Dict, List, Any
from ..base import ModelBase
from ..passenger import PassengerState, PassengerStatus

@dataclass
class PassengerSystemState(ModelBase):
    """Represents the state of the passenger system"""
    active_passengers: Dict[str, PassengerState]
    passengers_by_status: Dict[PassengerStatus, List[str]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_passengers": {
                pid: pstate.to_dict() for pid, pstate in self.active_passengers.items()
            },
            "passengers_by_status": {
                status.value: pids for status, pids in self.passengers_by_status.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PassengerSystemState':
        return cls(
            active_passengers={
                pid: PassengerState.from_dict(pstate) 
                for pid, pstate in data["active_passengers"].items()
            },
            passengers_by_status={
                PassengerStatus(status): pids 
                for status, pids in data["passengers_by_status"].items()
            },
        ) 