from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from .datamodel import DataModel

@dataclass
class ExecutionStep(DataModel):
    step_number: int
    description: str
    api_description: str
    result_key: Optional[str] = None

@dataclass
class ExecutionPlan(DataModel):
    description: str
    steps: List[ExecutionStep]
    final_result: Optional[str] = None