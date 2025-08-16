from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import json
import traceback

from services.api_service import APIService
from services.executor_service import ExecutorService
from services.visualization_service import VisualizationService
from services.socket_response_service import WebSocketResponseService
from models.execution_plan import ExecutionPlan
from builders.execution_plan import ExecutionPlanBuilder
from models.execution_result import ExecutionResult, ExecutionStatus
from utils import (
    sanitize_api_url,
    sanitize_api_params,
    make_api_call,
    update_nested_dict,
)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

_default_knowledge_base = """logged in user details: organization_id: 16d330dd-57ca-42f2-ab12-fb500c51beb9, user_id: 87cf268e-8049-4e54-ab2d-61d67134c1d2"""

# Initialize services
search_service = APIService(search_model="faiss")
multiple_requests_service = ExecutorService(search_service)
visualization_service = VisualizationService(search_service)
websocket_response_service = WebSocketResponseService(socketio)

@app.route('/')
def index_page():
    """Serve the main page."""
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print(f"Client connected: {request.sid}")
    websocket_response_service.emit_connection_status()
    # suggest initial suggestions like: Create event for tomorrow 5pm with title hello world
    next_best_items = {
        "suggestions": [
            "Create event for tomorrow 5pm with title hello world",
            "Get a users magiclink",
            "Get all event users for an event with id 123",
        ]
    }
    websocket_response_service.emit_next_best_items(next_best_items)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print(f"Client disconnected: {request.sid}")

@socketio.on('multiple_requests')
def handle_multiple_requests(data):
    try:
        user_input = data.get('query', '').strip()
        
        if not user_input:
            websocket_response_service.emit_error('No query provided')
            return
        
        user_input_with_knowledge_base = f"{user_input} {_default_knowledge_base}"
        
        print(f"Received multiple requests: {user_input}")
        websocket_response_service.emit_analyzing_status()
        
        # Step 1: Generate execution plan
        execution_plan: ExecutionPlan = multiple_requests_service.generate_execution_plan(user_input_with_knowledge_base)
        
        if not execution_plan or not execution_plan.steps:
            websocket_response_service.emit_error('Could not generate execution plan. Please try rephrasing your request.')
            return
        
        # Step 2: Send plan for user verification
        websocket_response_service.emit_execution_plan(execution_plan, user_input_with_knowledge_base)
        
    except Exception as e:
        print(f"Error processing multiple requests: {e}")
        print(traceback.format_exc())
        websocket_response_service.emit_error(f'An error occurred: {str(e)}')

@socketio.on('approve_execution_plan')
def handle_approve_execution_plan(data):
    """Handle user approval of execution plan and start execution."""
    try:
        plan: ExecutionPlan = ExecutionPlanBuilder.from_dict(data.get('plan', {}))
        user_input = data.get('user_input', '')
        
        if not plan or not plan.steps:
            websocket_response_service.emit_error('Invalid execution plan')
            return
        
        websocket_response_service.emit_executing_status()

        execution_result: ExecutionResult = multiple_requests_service.execute_plan_steps(
            plan, user_input, websocket_response_service
        )
        
        if execution_result.status == ExecutionStatus.COMPLETED:
            websocket_response_service.emit_multiple_requests_complete(execution_result, plan)
        
    except Exception as e:
        print(f"Error executing plan: {e}")
        print(traceback.format_exc())
        websocket_response_service.emit_error(f'An error occurred during execution: {str(e)}')

@socketio.on('modify_execution_plan')
def handle_modify_execution_plan(data):
    """Handle user request to modify the execution plan."""
    try:
        original_plan = data.get('plan', {})
        user_feedback = data.get('feedback', '')
        user_input = data.get('user_input', '')
        
        if not user_feedback:
            websocket_response_service.emit_error('No modification feedback provided')
            return
        
        websocket_response_service.emit_modifying_status()
        
        # Generate modified plan
        modified_plan = multiple_requests_service.modify_execution_plan(original_plan, user_feedback, user_input)
        
        if not modified_plan or not modified_plan.get('steps'):
            websocket_response_service.emit_error('Could not generate modified plan. Please try again.')
            return
        
        # Send modified plan for user verification
        websocket_response_service.emit_execution_plan(modified_plan, user_input)
        
    except Exception as e:
        print(f"Error modifying plan: {e}")
        print(traceback.format_exc())
        websocket_response_service.emit_error(f'An error occurred: {str(e)}')

@socketio.on('provide_multiple_requests_missing_fields')
def handle_multiple_requests_missing_fields(data):
    """Handle missing fields provided by user during multiple requests execution."""
    try:
        provided_fields = data.get('fields', {})
        current_params = data.get('current_params', {})
        matched_api = data.get('matched_api', {})
        step_number = data.get('step_number', 0)
        plan = data.get('plan', {})
        request_method = data.get('request_method', 'GET')
        user_input = data.get('user_input', '')
        previous_results = data.get('previous_results', {})
        completed_steps = data.get('completed_steps', [])
        
        print(f"Providing missing fields for step {step_number}")
        print("Provided fields:", provided_fields)
        
        # Use the same logic as handle_missing_fields
        updated_params = update_nested_dict(current_params, provided_fields)
        
        if request_method == "GET":
            api = matched_api + "?" + "&".join([f"{key}={value}" for key, value in updated_params.items()])
        else:
            api = matched_api
        
        # Make the API call
        sanitized_path = sanitize_api_url(api)
        sanitized_params = json.loads(sanitize_api_params(updated_params))
        
        response_data, status_code = make_api_call(
            request_method, 
            sanitized_path, 
            sanitized_params
        )
        
        if status_code > 299:
            websocket_response_service.emit_error(f'API call failed for step {step_number}: {status_code}: {response_data}')
            return
        
        # Create step result
        step_result = {
            'action': f'Step {step_number}',
            'api_path': api,
            'method': request_method,
            'payload': sanitized_params,
            'response': response_data,
            'status_code': status_code,
            'status': 'success',
            'step_number': step_number,
            'step_description': data.get('step_description', ''),
            'api_description': data.get('api_description', '')
        }
        
        # Store result for future steps
        step = next((s for s in plan.get('steps', []) if s.get('step_number') == step_number), None)
        if step and step.get('result_key'):
            previous_results[step.get('result_key')] = step_result
        
        # Add to completed steps
        completed_steps.append({
            'step_number': step_number,
            'description': data.get('step_description', ''),
            'api_description': data.get('api_description', ''),
            'status': 'success',
            'result': step_result
        })
        
        # Send step completion update
        websocket_response_service.emit_step_completed({
            'step_number': step_number,
            'description': data.get('step_description', ''),
            'api_description': data.get('api_description', ''),
            'result': step_result,
            'status': 'success'
        })
        
        # Continue execution from the next step
        websocket_response_service.emit_step_completed_status(step_number)
        
        # Resume execution from the next step using centralized logic
        next_step_number = step_number + 1
        execution_result = multiple_requests_service.execute_plan_steps(
            plan, user_input, websocket_response_service, 
            start_from_step=next_step_number, 
            previous_results=previous_results, 
            completed_steps=completed_steps
        )
        
        if execution_result.status == ExecutionStatus.COMPLETED:
            websocket_response_service.emit_multiple_requests_complete(execution_result, plan)
        
    except Exception as e:
        print(f"Error handling multiple requests missing fields: {e}")
        print(traceback.format_exc())
        websocket_response_service.emit_error(f'An error occurred: {str(e)}')

def get_final_response(action, api_path, method, payload):
    return {
            'action': action,
            'api_path': api_path,
            'method': method,
            'payload': payload,
            'status': 'success'
        }

if __name__ == '__main__':
    print("✅ WebSocket Server is ready!")
    socketio.run(app, debug=True, host='0.0.0.0', port=8010)