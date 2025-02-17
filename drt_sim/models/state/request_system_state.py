from dataclasses import dataclass
from typing import Dict, List, Any
from ..base import ModelBase
from ..request import Request, RequestStatus

@dataclass
class RequestSystemState(ModelBase):
    """Represents the state of the request system"""
    active_requests: Dict[str, Request]
    historical_requests: Dict[str, Request]
    requests_by_status: Dict[RequestStatus, List[str]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_requests": {
                rid: req.to_dict() for rid, req in self.active_requests.items()
            },
            "historical_requests": {
                rid: req.to_dict() for rid, req in self.historical_requests.items()
            },
            "requests_by_status": {
                status.value: rids for status, rids in self.requests_by_status.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RequestSystemState':
        return cls(
            active_requests={
                rid: Request.from_dict(req) 
                for rid, req in data["active_requests"].items()
            },
            historical_requests={
                rid: Request.from_dict(req) 
                for rid, req in data["historical_requests"].items()
            },
            requests_by_status={
                RequestStatus(status): rids 
                for status, rids in data["requests_by_status"].items()
            },
        ) 