from typing import Dict, Any, Optional, List

from api_requester.models.execution_plan import ExecutionPlan
from api_requester.models.execution_result import Execution
from api_requester.models.step_result import StepResult
from api_requester.models.missing_fields_data import MissingFieldsData


class WebSocketResponseService:
    """Centralized service for handling all WebSocket emit operations."""
    
    def __init__(self, socketio):
        self.socketio = socketio
    
    def emit_status(self, message: str, status: str):
        """Emit status update."""
        self.socketio.emit('status', {'message': message, 'status': status})
    
    def emit_error(self, message: str):
        """Emit error message."""
        self.socketio.emit('error', {'message': message})
    
    def emit_connection_status(self, message: str = 'Connected to API Requester Server'):
        """Emit connection status."""
        self.emit_status(message, 'connected')
    
    def emit_next_best_items(self, suggestions: Dict[str, List[str]]):
        """Emit next best items suggestions."""
        self.socketio.emit('next_best_items', suggestions)
    
    def emit_missing_fields(self, missing_fields_data: MissingFieldsData):
        """Emit missing fields request."""
        self.socketio.emit('missing_fields', missing_fields_data.to_dict())
    
    def emit_api_response(self, response_data: Dict[str, Any]):
        """Emit API response."""
        self.socketio.emit('api_response', response_data)
    
    def emit_execution_plan(self, plan: ExecutionPlan, user_input: str, status: str = 'pending_approval'):
        """Emit execution plan."""
        self.socketio.emit('execution_plan', {
            'plan': plan.to_dict(),
            'user_input': user_input,
            'status': status
        })
    
    def emit_multiple_requests_missing_fields(self, missing_fields_data: Dict[str, Any]):
        """Emit multiple requests missing fields with consistent structure."""
        required_fields = [
            'step_number', 'step_description', 'api_description', 
            'missing_fields', 'current_params', 'matched_api', 
            'plan', 'user_input', 'request_method', 
            'previous_results', 'completed_steps'
        ]
        
        # Ensure all required fields are present
        emit_data = {}
        for field in required_fields:
            emit_data[field] = missing_fields_data.get(field)
        
        self.socketio.emit('multiple_requests_missing_fields', emit_data)
    
    def emit_step_completed(self, step_data: StepResult):
        """Emit step completion status."""
        self.socketio.emit('step_completed', step_data.to_dict())
    
    def emit_multiple_requests_complete(self, results: Execution, plan: ExecutionPlan):
        """Emit multiple requests completion."""
        self.socketio.emit('multiple_requests_complete', {
            'results': results,
            'plan': plan,
            'status': 'completed'
        })
    
    def emit_visualization_results(self, query: str, results: Dict[str, Any]):
        """Emit visualization results."""
        self.socketio.emit('visualization_results', {
            'query': query,
            'results': results
        })
    
    def emit_upload_success(self, success_data: Dict[str, Any]):
        """Emit upload success."""
        self.socketio.emit('upload_success', success_data)
    
    def emit_upload_error(self, message: str):
        """Emit upload error."""
        self.socketio.emit('upload_error', {'message': message})
    
    # Processing status methods
    def emit_processing_status(self):
        """Emit processing status."""
        self.emit_status('Processing your request...', 'processing')
    
    def emit_searching_status(self):
        """Emit searching status."""
        self.emit_status('Finding matching API...', 'searching')
    
    def emit_found_api_status(self, api_path: str):
        """Emit found API status."""
        self.emit_status(f'Found API: {api_path}', 'found_api')
    
    def emit_generating_status(self):
        """Emit generating status."""
        self.emit_status('Generating API payload...', 'generating')
    
    def emit_analyzing_status(self):
        """Emit analyzing status."""
        self.emit_status('Analyzing your complex request...', 'analyzing')
    
    def emit_executing_status(self):
        """Emit executing status."""
        self.emit_status('Starting execution of approved plan...', 'executing')
    
    def emit_modifying_status(self):
        """Emit modifying status."""
        self.emit_status('Modifying plan based on your feedback...', 'modifying')
    
    def emit_executing_step_status(self, step_number: int, step_description: str):
        """Emit executing step status."""
        self.emit_status(f'Executing step {step_number}: {step_description}', 'executing_step')
    
    def emit_step_completed_status(self, step_number: int):
        """Emit step completed status."""
        self.emit_status(f'Step {step_number} completed. Continuing execution...', 'executing_step')
    
    def emit_visualizing_status(self):
        """Emit visualizing status."""
        self.emit_status('Generating visualization...', 'visualizing') 