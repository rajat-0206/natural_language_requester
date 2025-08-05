from dataclasses import dataclass
from typing import List, Dict, Any
from .datamodel import DataModel

@dataclass
class MissingFieldsData(DataModel):
    step_number: int
    step_description: str
    api_description: str
    missing_fields: List[str]
    current_params: Dict[str, Any]
    matched_api: Dict[str, Any]
    plan: Dict[str, Any]
    user_input: str
    request_method: str
    previous_results: Dict[str, Any]
    completed_steps: List[Dict[str, Any]] 