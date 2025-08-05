from typing import List, Dict, Any
from models.execution_result import Execution, ExecutionStatus
from models.step_result import StepResult
from builders.step_response import StepResponseBuilder


class ExecutionResultBuilder:

    
    @staticmethod
    def build_from_step_results(plan_description: str, step_results: List[StepResult], 
                               final_result: str = None, api_description: str = None) -> Execution:
        """Build ExecutionResult from step results."""
        return Execution(
            plan_description=plan_description,
            final_result=final_result,
            api_description=api_description,
            step_results=step_results
        )
    
    @staticmethod
    def build_execution_result(plan_description: str, step_results: List[StepResult] = None,
                              final_result: str = None, api_description: str = None,
                              status: ExecutionStatus = ExecutionStatus.COMPLETED) -> Execution:
        """Build ExecutionResult from components."""
        
        return Execution(
            plan_description=plan_description,
            final_result=final_result,
            api_description=api_description,
            step_results=step_results,
            status=status
        ) 