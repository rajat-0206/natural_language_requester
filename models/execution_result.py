from dataclasses import dataclass
from typing import List, Optional
from .step_result import StepResult
from .datamodel import DataModel
from enum import Enum


class ExecutionStatus(Enum):
    COMPLETED = 'completed'
    FAILED = 'failed'
    MISSING_FIELDS = 'missing_fields'


@dataclass
class ExecutionResult(DataModel):
    plan_description: str
    final_result: Optional[str] = None
    api_description: Optional[str] = None
    step_results: List[StepResult] = None
    status: ExecutionStatus = ExecutionStatus.COMPLETED
    
    def __post_init__(self):
        if self.step_results is None:
            self.step_results = [] 

