from typing import Dict, Any, Optional
from models.step_result import StepResult, StepResultStatus
from models.api_response import ApiResponse


class StepResponseBuilder:
    @staticmethod
    def build_step_response(api_response: ApiResponse, step_number: int, step_description: str, api_description: str) -> StepResult:
        """Build StepResult from API response result."""
        if api_response.missing_fields:
            return StepResponseBuilder.build_step_missing_fields_response(
                api_response, step_number, step_description, api_description
            )
        elif api_response.error:
            return StepResponseBuilder.build_step_error_response(
                api_response.error, 'failed', step_number, step_description, api_description
            )
        else:
            return StepResponseBuilder.build_step_success_response(
                api_response, step_number, step_description, api_description
            )

    @staticmethod
    def build_step_success_response(api_response: ApiResponse, step_number: int, step_description: str, api_description: str) -> StepResult:
        return StepResult(
            step_number=step_number,
            step_description=step_description,
            api_description=api_description,
            status=StepResultStatus.SUCCESS,
            api_response=api_response
        )
    
    @staticmethod
    def build_step_error_response(error: str, step_number: int, step_description: str, api_description: str) -> StepResult:
        return StepResult(
            step_number=step_number,
            step_description=step_description,
            api_description=api_description,
            status=StepResultStatus.FAILED,
            error=error
        )
    
    @staticmethod
    def build_step_missing_fields_response(api_response: ApiResponse, step_number: int, step_description: str, api_description: str) -> StepResult:
        return StepResult(
            step_number=step_number,
            step_description=step_description,
            api_description=api_description,
            status=StepResultStatus.MISSING_FIELDS,
            api_response=api_response
        )