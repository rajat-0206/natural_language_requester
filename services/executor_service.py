"""
Executor Service - Handles complex multi-step API requests with dataclass support.
"""

import json
import traceback
from typing import Dict, Any, List, Optional
from utils import call_model, extract_json_from_response
from utils.prompts import GENERATE_EXECUTION_PLAN_PROMPT, MODIFY_EXECUTION_PLAN_PROMPT
from api_requester.services.api_service import APIService
from builders.step_response import StepResponseBuilder
from builders.missing_fields_data import MissingFieldsDataBuilder
from builders.execution_result import ExecutionResultBuilder
from models.execution_result import Execution, ExecutionStatus
from models.step_result import StepResult, StepResultStatus
from models.execution_plan import ExecutionPlan, ExecutionStep
from builders.execution_plan import ExecutionPlanBuilder
from services.socket_response_service import WebSocketResponseService
from models.api_response import ApiResponse

class ExecutorService:
    """Service for handling multiple API requests with execution plans."""
    
    def __init__(self, api_service: APIService):
        self.api_service: APIService = api_service
        
    def generate_execution_plan(self, user_input: str) -> ExecutionPlan:
        """Generate an execution plan for multiple API requests."""
        prompt = GENERATE_EXECUTION_PLAN_PROMPT.format(user_input=user_input)
        
        response = call_model(prompt)
        try:
            json_str = extract_json_from_response(response)
            return ExecutionPlanBuilder.from_dict(json.loads(json_str))
        except Exception as e:
            print(f"Failed to parse execution plan: {e}")
            return None
    
    def modify_execution_plan(self, original_plan: ExecutionPlan, user_feedback: str, user_input: str) -> ExecutionPlan:
        """Modify an execution plan based on user feedback."""
        prompt = MODIFY_EXECUTION_PLAN_PROMPT.format(
            user_input=user_input,
            original_plan=original_plan.to_json(),
            user_feedback=user_feedback
        )
        
        response = call_model(prompt)
        try:
            json_str = extract_json_from_response(response)
            return ExecutionPlanBuilder.from_dict(json.loads(json_str))
        except Exception as e:
            print(f"Failed to parse modified plan: {e}")
            return None
    
    def execute_single_step(self, step: ExecutionStep, previous_results: List[StepResult], user_input: str) -> StepResult:
        """Execute a single step in the plan using the search service."""
        try:
            step_number = step.step_number
            step_description = step.description
            api_description = step.api_description
            
            # Enhance the API request with context from previous results
            enhanced_request = self.api_service.enhance_request_with_context(api_description, previous_results, user_input)
            print(f"Enhanced request: {enhanced_request}")
            
            # Use the search service to process the API request
            api_response: ApiResponse = self.api_service.process_api_request(enhanced_request)

            step_result: StepResult = StepResponseBuilder.build_step_response(api_response, step_number, step_description, api_description)
            print(f"Step {step_number} result: {step_result}")
            return step_result
            
        except Exception as e:
            print(f"Error in execute_single_step: {e}")
            print(traceback.format_exc())
            return StepResponseBuilder.build_step_error_response(
                error=str(e),
                step_number=step.step_number,
                step_description=step.description,
                api_description=step.api_description
            )
    
    def execute_plan_steps(self, plan: ExecutionPlan, user_input: str, websocket_service: WebSocketResponseService, 
                          start_from_step: int = 1, previous_results: Optional[Dict[str, ApiResponse]] = None, 
                          completed_steps: Optional[List[StepResult]] = None) -> Execution:
        """Execute the steps in the plan sequentially with WebSocket communication."""
        if previous_results is None:
            previous_results = {}
        if completed_steps is None:
            completed_steps = []
        
        # Get steps starting from the specified step number
        steps_to_execute = [step for step in plan.steps if step.step_number >= start_from_step]
        
        for step in steps_to_execute:
            try:
                print(f"Executing step {step.step_number}: {step.description}")
                websocket_service.emit_executing_step_status(step.step_number, step.description)
                
                # Execute the step
                step_result: StepResult = self.execute_single_step(step, previous_results, user_input)
                
                # Check if step failed due to missing fields
                if step_result.status == StepResultStatus.MISSING_FIELDS and step_result.api_response and step_result.api_response.missing_fields:
                    # Return the missing fields info for the websocket to handle
                    missing_fields_data = MissingFieldsDataBuilder.build_missing_fields_data(
                        step_result=step_result,
                        plan=plan,
                        user_input=user_input,
                        previous_results=previous_results,
                        completed_steps=completed_steps
                    )
                    # if we find missing fields, we will raise websocket event from here itself
                    websocket_service.emit_missing_fields(missing_fields_data)
                    return ExecutionResultBuilder.build_execution_result(
                        plan_description=plan.description,
                        step_results=completed_steps,
                        final_result=plan.final_result,
                        status=ExecutionStatus.MISSING_FIELDS
                    )
                
                # Store result for future steps
                if step.result_key and step_result.status == StepResultStatus.SUCCESS and step_result.api_response:
                    previous_results[step.result_key] = step_result.api_response
                
                completed_steps.append(step_result)
                
                # Send step completion update
                websocket_service.emit_step_completed({
                    'step_number': step.step_number,
                    'description': step.description,
                    'api_description': step.api_description,
                    'result': step_result.api_response,
                    'status': step_result.status,
                    'error': step_result.error
                })
                
            except Exception as e:
                print(f"Error executing step {step.step_number}: {e}")
                completed_steps.append(StepResponseBuilder.build_step_error_response(
                    error=str(e),
                    status='failed',
                    step_number=step.step_number,
                    step_description=step.description,
                    api_description=step.api_description
                ))
                
                websocket_service.emit_step_completed(step_result)
        
        # Return final results using builders
        step_results = []
        for step_data in completed_steps:
            step_result = StepResult(
                step_number=step_data.step_number,
                step_description=step_data.description,
                api_description=step_data.api_description,
                status=step_data.status,
                api_response=step_data.api_response,
                error=step_data.error
            )
            step_results.append(step_result)
        
        final_result = ExecutionResultBuilder.build_execution_result(
            plan_description=plan.description,
            step_results=step_results,
            final_result=plan.final_result,
            status=ExecutionStatus.COMPLETED
        )
        
        return final_result