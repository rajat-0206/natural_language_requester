from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import json
import subprocess
import traceback
import time

from services.search import APISearchService
from services.multiple_requests import MultipleRequestsService
from services.visualization import VisualizationService
from services.single_request import SingleRequestService
from handlers import SingleRequestHandlers

from utils import (
    sanitize_api_url,
    sanitize_api_params,
    make_api_call,
    update_nested_dict,
)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize services
search_service = APISearchService(search_model="faiss")
multiple_requests_service = MultipleRequestsService(search_service)
visualization_service = VisualizationService(search_service)
single_request_service = SingleRequestService(search_service)
single_request_handlers = SingleRequestHandlers(single_request_service)

@app.route('/')
def index_page():
    """Serve the main page."""
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print(f"Client connected: {request.sid}")
    emit('status', {'message': 'Connected to API Requester Server', 'status': 'connected'})
    # suggest initial suggestions like: Create event for tomorrow 5pm with title hello world
    next_best_items = {
        "suggestions": [
            "Create event for tomorrow 5pm with title hello world",
            "Get a users magiclink",
            "Get all event users for an event with id 123",
        ]
    }
    emit('next_best_items', next_best_items)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print(f"Client disconnected: {request.sid}")

@socketio.on('api_request')
def handle_api_request(data):
    """
    Handler for single API requests (previous version of the agent).
    Delegates to the SingleRequestHandlers class.
    """
    single_request_handlers.handle_api_request(data)

@socketio.on('multiple_requests')
def handle_multiple_requests(data):
    '''
    1.Can handle multiple request in a single user input.
    2. Call AI to generate a plan where it breaks the user input
    into seperate actions.
    3. Return the plan to user for approval or modification.
    '''
    try:
        user_input = data.get('query', '').strip()
        
        if not user_input:
            emit('error', {'message': 'No query provided'})
            return
        
        print(f"Received multiple requests: {user_input}")
        emit('status', {'message': 'Analyzing your complex request...', 'status': 'analyzing'})
        
        # Step 1: Generate execution plan
        execution_plan = multiple_requests_service.generate_execution_plan(user_input)
        
        if not execution_plan or not execution_plan.get('steps'):
            emit('error', {'message': 'Could not generate execution plan. Please try rephrasing your request.'})
            return
        
        # Step 2: Send plan for user verification
        emit('execution_plan', {
            'plan': execution_plan,
            'user_input': user_input,
            'status': 'pending_approval'
        })
        
    except Exception as e:
        print(f"Error processing multiple requests: {e}")
        print(traceback.format_exc())
        emit('error', {'message': f'An error occurred: {str(e)}'})

@socketio.on('approve_execution_plan')
def handle_approve_execution_plan(data):
    """
    1. Handle user approval of execution plan and start execution.
    2. Plan execution is similar to executing single API request user input.
    3. We first ask AI to break the step into data and action.
    4. For the given action we find the best API using swaggers vector embedding.
    5. Then we give the best matched API and data from step 3 to AI again to return a payload.
    6. If some field is missing we ask user to provide them.
    7. If field is not missing we execute the API using admin token and store the result.
    8. The stored result is used as context for next step of plan.
    """
    try:
        plan = data.get('plan', {})
        user_input = data.get('user_input', '')
        
        if not plan or not plan.get('steps'):
            emit('error', {'message': 'Invalid execution plan'})
            return
        
        emit('status', {'message': 'Starting execution of approved plan...', 'status': 'executing'})
        
        # Create a callback function to emit WebSocket messages
        def emit_callback(event, data):
            emit(event, data)
        
        # Execute the plan step by step with real-time updates
        results = multiple_requests_service.execute_plan_steps(plan, user_input, emit_callback)
        
        # Check if execution was paused due to missing fields
        if results.get('missing_fields'):
            emit('multiple_requests_missing_fields', {
                'step_number': results['step_number'],
                'step_description': results['step_description'],
                'api_description': results['api_description'],
                'missing_fields': results['missing_fields'],
                'current_params': results['current_params'],
                'matched_api': results['matched_api'],
                'plan': results['plan'],
                'user_input': results['user_input'],
                'request_method': results['request_method'],
                'previous_results': results['previous_results'],
                'completed_steps': results['completed_steps']
            })
            return
        
        # Send final results
        emit('multiple_requests_complete', {
            'results': results,
            'plan': plan,
            'status': 'completed'
        })
        
    except Exception as e:
        print(f"Error executing plan: {e}")
        print(traceback.format_exc())
        emit('error', {'message': f'An error occurred during execution: {str(e)}'})

@socketio.on('modify_execution_plan')
def handle_modify_execution_plan(data):
    """Handle user request to modify the execution plan."""
    try:
        original_plan = data.get('plan', {})
        user_feedback = data.get('feedback', '')
        user_input = data.get('user_input', '')
        
        if not user_feedback:
            emit('error', {'message': 'No modification feedback provided'})
            return
        
        emit('status', {'message': 'Modifying plan based on your feedback...', 'status': 'modifying'})
        
        # Generate modified plan
        modified_plan = multiple_requests_service.modify_execution_plan(original_plan, user_feedback, user_input)
        
        if not modified_plan or not modified_plan.get('steps'):
            emit('error', {'message': 'Could not generate modified plan. Please try again.'})
            return
        
        # Send modified plan for user verification
        emit('execution_plan', {
            'plan': modified_plan,
            'user_input': user_input,
            'status': 'pending_approval'
        })
        
    except Exception as e:
        print(f"Error modifying plan: {e}")
        print(traceback.format_exc())
        emit('error', {'message': f'An error occurred: {str(e)}'})

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
        
        # Find the current step
        current_step = next((s for s in plan.get('steps', []) if s.get('step_number') == step_number), None)
        if not current_step:
            emit('error', {'message': f'Step {step_number} not found in plan'})
            return
        
        # Execute the step with provided fields using the service
        step_result = multiple_requests_service.execute_step_with_missing_fields(
            current_step, provided_fields, current_params, matched_api, 
            request_method, previous_results, user_input
        )
        
        # Add to completed steps
        completed_steps.append({
            'step_number': step_number,
            'description': data.get('step_description', ''),
            'api_description': data.get('api_description', ''),
            'status': 'success' if step_result and step_result.get('status') == 'success' else 'failed',
            'result': step_result
        })
        
        # Send step completion update
        emit('step_completed', {
            'step_number': step_number,
            'description': data.get('step_description', ''),
            'api_description': data.get('api_description', ''),
            'result': step_result,
            'status': 'success' if step_result and step_result.get('status') == 'success' else 'failed'
        })
        
        # Continue execution from the next step
        emit('status', {'message': f'Step {step_number} completed. Continuing execution...', 'status': 'executing_step'})
        
        # Resume execution from the next step with real-time updates
        remaining_steps = [s for s in plan.get('steps', []) if s.get('step_number') > step_number]
        
        # Create a callback function to emit WebSocket messages
        def emit_callback(event, data):
            emit(event, data)
        
        # Execute remaining steps with real-time updates
        for step in remaining_steps:
            try:
                step_num = step.get('step_number', 0)
                step_desc = step.get('description', '')
                api_desc = step.get('api_description', '')
                
                print(f"Executing step {step_num}: {step_desc}")
                emit_callback('status', {'message': f'Executing step {step_num}: {step_desc}', 'status': 'executing_step'})
                
                # Execute the step with updated previous results
                step_result = multiple_requests_service.execute_single_step(step, previous_results, user_input)
                
                # Check if step failed due to missing fields
                if step_result and step_result.get('missing_fields'):
                    # Pause execution and request missing fields from user
                    emit('multiple_requests_missing_fields', {
                        'step_number': step_num,
                        'step_description': step_desc,
                        'api_description': api_desc,
                        'missing_fields': step_result['missing_fields'],
                        'current_params': step_result['current_params'],
                        'matched_api': step_result['matched_api'],
                        'plan': plan,
                        'user_input': user_input,
                        'previous_results': previous_results,
                        'completed_steps': completed_steps
                    })
                    return  # Pause execution
                
                # Store result for future steps
                result_key = step.get('result_key')
                if result_key and step_result and step_result.get('status') == 'success':
                    previous_results[result_key] = step_result
                
                # Create step completion data
                step_completion_data = {
                    'step_number': step_num,
                    'description': step_desc,
                    'api_description': api_desc,
                    'status': 'success' if step_result and step_result.get('status') == 'success' else 'failed',
                    'result': step_result
                }
                
                if step_result and step_result.get('error'):
                    step_completion_data['error'] = step_result['error']
                
                completed_steps.append(step_completion_data)
                
                # Send step completion update
                emit_callback('step_completed', step_completion_data)
                
            except Exception as e:
                print(f"Error executing step {step_num}: {e}")
                error_step_data = {
                    'step_number': step_num,
                    'description': step_desc,
                    'api_description': api_desc,
                    'status': 'failed',
                    'error': str(e)
                }
                
                completed_steps.append(error_step_data)
                emit_callback('step_completed', error_step_data)
        
        # Send final results
        emit('multiple_requests_complete', {
            'results': {
                'plan_description': plan.get('description', ''),
                'final_result': plan.get('final_result', ''),
                'api_description': plan.get('api_description', ''),
                'step_results': completed_steps,
                'overall_status': 'completed'
            },
            'plan': plan,
            'status': 'completed'
        })
        
    except Exception as e:
        print(f"Error handling multiple requests missing fields: {e}")
        print(traceback.format_exc())
        emit('error', {'message': f'An error occurred: {str(e)}'})


@socketio.on('provide_missing_fields')
def handle_missing_fields(data):
    """Handler for missing fields provided by user (previous version of the agent)."""
    single_request_handlers.handle_missing_fields(data)


if __name__ == '__main__':
    print("✅ WebSocket Server is ready!")
    socketio.run(app, debug=True, host='0.0.0.0', port=8010)