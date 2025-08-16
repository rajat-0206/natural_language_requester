from typing import List, Dict, Any
from models.step_result import StepResult
from models.execution_plan import ExecutionPlan
from models.missing_fields_data import MissingFieldsData

class MissingFieldsDataBuilder:

    
    @staticmethod
    def build_from_step_result(step_result: StepResult, step_number: int, 
                              step_description: str, api_description: str,
                              plan: ExecutionPlan, user_input: str,
                              previous_results: List[StepResult], 
                              completed_steps: List[Dict[str, Any]]) -> MissingFieldsData:
        """Build MissingFieldsData from step result."""
        return MissingFieldsData(
            step_number=step_number,
            step_description=step_description,
            api_description=api_description,
            missing_fields=step_result.api_response.missing_fields,
            current_params=step_result.api_response.current_params,
            matched_api=step_result.api_response.matched_api,
            plan=plan,
            user_input=user_input,
            request_method=step_result.api_response.request_method,
            previous_results=previous_results,
            completed_steps=completed_steps
        )
    
    @staticmethod
    def build_missing_fields_data(step_result: StepResult, plan: ExecutionPlan, user_input: str, previous_results: List[StepResult], completed_steps: List[StepResult]) -> MissingFieldsData:
        """Build MissingFieldsData from components."""
        return MissingFieldsData(
            step_number=step_result.step_number,
            step_description=step_result.step_description,
            api_description=step_result.api_description,
            missing_fields=step_result.api_response.missing_fields,
            matched_api=step_result.api_response.matched_api,
            request_method=step_result.api_response.request_method,
            current_params=step_result.api_response.current_params,
            previous_results=previous_results,
            completed_steps=completed_steps,
            plan=plan,
            user_input=user_input,
        ) 