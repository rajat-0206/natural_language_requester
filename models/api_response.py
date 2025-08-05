from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List
from .datamodel import DataModel


class ApiResponseStatus(Enum):
    NOT_STARTED = 'not_started'
    SUCCESS = 'success'
    MISSING_FIELDS = 'missing_fields'
    ERROR = 'error'


@dataclass
class ApiResponse(DataModel):
    '''
    This model should be able to handle all api responses
    1. Success response
    2. Missing fields response
    3. Error response
    '''
    action: Optional[str] = None
    user_input: Optional[str] = None
    api_path: Optional[str] = None
    method: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    response: Optional[Dict[str, Any]] = None
    status_code: Optional[int] = None
    status: ApiResponseStatus = ApiResponseStatus.NOT_STARTED
    missing_fields: Optional[List[str]] = None
    current_params: Optional[Dict[str, Any]] = None
    matched_api: Optional[Dict[str, Any]] = None
    request_method: Optional[str] = None



    
    
    
    