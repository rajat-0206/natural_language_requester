from typing import List, Dict, Any
from models.execution_plan import ExecutionPlan, ExecutionStep


class ExecutionPlanBuilder:
    """Builder for execution plan data structures."""
    
    @staticmethod
    def build_execution_plan(description: str, steps: List[ExecutionStep], 
                           final_result: str = None, api_description: str = None) -> ExecutionPlan:
        """Build ExecutionPlan from components."""
        return ExecutionPlan(
            description=description,
            steps=steps,
            final_result=final_result,
            api_description=api_description
        )
    
    @staticmethod
    def build_execution_step(step_number: int, description: str, api_description: str, 
                           result_key: str = None) -> ExecutionStep:
        """Build ExecutionStep from components."""
        return ExecutionStep(
            step_number=step_number,
            description=description,
            api_description=api_description,
            result_key=result_key
        )
    
    def from_dict(plan: Dict[str, Any]) -> ExecutionPlan:
        plan_description = plan.get('description', '')
        steps = plan.get('steps', [])
        final_result = plan.get('final_result', '')
        execution_steps = []
        for step_number, step in enumerate(steps):
            execution_steps.append(ExecutionPlanBuilder.build_execution_step(step_number, step.get('description', ''), step.get('api_description', ''), step.get('result_key', '')))
        return ExecutionPlanBuilder.build_execution_plan(plan_description, execution_steps, final_result)

    
    
 