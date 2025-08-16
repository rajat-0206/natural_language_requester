from dataclasses import dataclass
from typing import Optional, Dict, Any
from .api_response import ApiResponse
from .datamodel import DataModel
from enum import Enum

@dataclass
class StepResult(DataModel):
    step_number: int
    step_description: str
    api_description: str
    status: str
    api_response: Optional[ApiResponse] = None
    error: Optional[str] = None


    def to_dict(self):
        return {
            'step_number': self.step_number,
            'description': self.step_description,
            'api_description': self.api_description,
            'result': self.api_response,
            'status': self.status,
            'error': self.error
        }

class StepResultStatus(Enum):
    SUCCESS = 'success'
    FAILED = 'failed'
    MISSING_FIELDS = 'missing_fields'