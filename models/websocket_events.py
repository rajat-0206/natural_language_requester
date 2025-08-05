from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from .datamodel import DataModel

@dataclass
class StatusEvent(DataModel):
    message: str
    status: str

@dataclass
class ErrorEvent(DataModel):
    message: str

@dataclass
class NextBestItemsEvent(DataModel):
    suggestions: List[str]

@dataclass
class StepCompletedEvent(DataModel):
    step_number: int
    description: str
    api_description: str
    result: Optional[Dict[str, Any]] = None
    status: str = 'success'
    error: Optional[str] = None

@dataclass
class ExecutionPlanEvent(DataModel):
    plan: Dict[str, Any]
    user_input: str
    status: str = 'pending_approval'

@dataclass
class MultipleRequestsCompleteEvent(DataModel):
    results: Dict[str, Any]
    plan: Dict[str, Any]
    status: str = 'completed'

@dataclass
class VisualizationResultsEvent(DataModel):
    query: str
    results: Dict[str, Any]

@dataclass
class UploadSuccessEvent(DataModel):
    message: str
    output: Optional[str] = None

@dataclass
class UploadErrorEvent(DataModel):
    message: str 