
from typing import Dict, Any, List
from models.api_response import ApiResponse


class APIResponseBuilder:
    @staticmethod
    def build_success_response(action: str, api_path: str, method: str, payload: Dict[str, Any], 
                              response: Dict[str, Any], status_code: int, status: str = 'success') -> ApiResponse:
        return ApiResponse(
            action=action,
            api_path=api_path,
            method=method,
            payload=payload,
            response=response,
            status_code=status_code,
            status=status
        )
    
    @staticmethod
    def build_missing_fields_response(missing_fields: List[str], current_params: Dict[str, Any], 
                                    matched_api: Dict[str, Any], request_method: str, 
                                    action: str, user_input: str) -> ApiResponse:
        return ApiResponse(
            missing_fields=missing_fields,
            current_params=current_params,
            matched_api=matched_api,
            request_method=request_method,
            action=action,
            user_input=user_input
        )
    
    @staticmethod
    def build_error_response(error: str, status_code: int, status: str = 'failed', 
                           action: str = None, api_path: str = None, 
                           method: str = None, payload: Dict[str, Any] = None) -> ApiResponse:
        return ApiResponse(
            error=error,
            status_code=status_code,
            status=status,
            action=action,
            api_path=api_path,
            method=method,
            payload=payload
        )
    
