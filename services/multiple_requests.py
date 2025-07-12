"""
Multiple Requests Service - Handles complex multi-step API requests.
"""

import json
import traceback
from utils import call_model, extract_json_from_response
from utils.prompts import GENERATE_EXECUTION_PLAN, MODIFY_EXECUTION_PLAN
from services.search import APISearchService

class MultipleRequestsService:
    """Service for handling multiple API requests with execution plans."""
    
    def __init__(self, search_service):
        self.search_service: APISearchService = search_service
    
    def generate_execution_plan(self, user_input):
        """Generate an execution plan for multiple API requests."""
        prompt = GENERATE_EXECUTION_PLAN.format(user_input=user_input)
        
        response = call_model(prompt)
        try:
            json_str = extract_json_from_response(response)
            return json.loads(json_str)
        except Exception as e:
            print(f"Failed to parse execution plan: {e}")
            return None
    
    def modify_execution_plan(self, original_plan, user_feedback, user_input):
        """Modify an execution plan based on user feedback."""
        prompt = MODIFY_EXECUTION_PLAN.format(
            user_input=user_input,
            original_plan=json.dumps(original_plan, indent=2),
            user_feedback=user_feedback
        )
        
        response = call_model(prompt)
        try:
            json_str = extract_json_from_response(response)
            return json.loads(json_str)
        except Exception as e:
            print(f"Failed to parse modified plan: {e}")
            return None
    
    def execute_single_step(self, step, previous_results, user_input):
        """Execute a single step in the plan using the search service."""
        try:
            step_number = step.get('step_number', 0)
            step_description = step.get('description', '')
            api_description = step.get('api_description', '')
            
            # Enhance the API request with context from previous results
            enhanced_request = self.search_service.enhance_request_with_context(api_description, previous_results, user_input)
            print(f"Enhanced request: {enhanced_request}")
            
            # Use the search service to process the API request
            result = self.search_service.process_api_request(enhanced_request)
            
            # Add step metadata to the result
            result['step_number'] = step_number
            result['step_description'] = step_description
            result['api_description'] = api_description
            
            print(f"Step {step_number} result: {result}")
            return result
            
        except Exception as e:
            print(f"Error in execute_single_step: {e}")
            print(traceback.format_exc())
            return {
                'error': str(e),
                'status': 'failed',
                'step_number': step.get('step_number', 0),
                'step_description': step.get('description', ''),
                'api_description': step.get('api_description', '')
            }
    
    def execute_plan_steps(self, plan, user_input):
        """Execute the steps in the plan sequentially."""
        results = {}
        step_results = []
        
        for step in plan.get('steps', []):
            try:
                step_num = step.get('step_number', 0)
                step_desc = step.get('description', '')
                api_desc = step.get('api_description', '')
                
                print(f"Executing step {step_num}: {step_desc}")
                
                # Execute the step
                step_result = self.execute_single_step(step, results, user_input)
                
                # Check if step failed due to missing fields
                if step_result and step_result.get('missing_fields'):
                    # Return the missing fields info for the websocket to handle
                    return {
                        'missing_fields': step_result['missing_fields'],
                        'current_params': step_result['current_params'],
                        'matched_api': step_result['matched_api'],
                        'request_method': step_result['request_method'],
                        'step_number': step_num,
                        'step_description': step_desc,
                        'api_description': api_desc,
                        'plan': plan,
                        'user_input': user_input,
                        'previous_results': results,
                        'completed_steps': step_results
                    }
                
                # Store result for future steps
                result_key = step.get('result_key')
                if result_key and step_result and step_result.get('status') == 'success':
                    results[result_key] = step_result
                
                step_results.append({
                    'step_number': step_num,
                    'description': step_desc,
                    'api_description': api_desc,
                    'status': 'success' if step_result and step_result.get('status') == 'success' else 'failed',
                    'result': step_result,
                })
                
            except Exception as e:
                print(f"Error executing step {step_num}: {e}")
                step_results.append({
                    'step_number': step_num,
                    'description': step_desc,
                    'api_description': api_desc,
                    'status': 'failed',
                    'error': str(e),
                })
        
        return {
            'plan_description': plan.get('description', ''),
            'final_result': plan.get('final_result', ''),
            'api_description': api_desc,
            'step_results': step_results,
            'overall_status': 'completed'
        } 