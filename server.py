from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import json
import subprocess
import traceback
import time

from api_requester.services.api_service import APIService
from api_requester.services.executor_service import ExecutorService
from api_requester.services.visualization_service import VisualizationService
from api_requester.services.socket_response_service import WebSocketResponseService
from api_requester.models.execution_plan import ExecutionPlan
from api_requester.builders.execution_plan import ExecutionPlanBuilder
from api_requester.models.execution_result import Execution, ExecutionStatus
from utils import (
    sanitize_api_url,
    sanitize_api_params,
    make_api_call,
    suggest_next_best_item,
    update_nested_dict,
    extract_action_data,
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

@socketio.on('api_request')
def handle_api_request(data):
    """Handle API request from frontend."""
    try:
        user_input = data.get('query', '').strip()
        
        if not user_input:
            websocket_response_service.emit_error('No query provided')
            return
        
        print(f"Received request: {user_input}")
        websocket_response_service.emit_processing_status()
        
        # Extract action/data/details using Claude
        extracted = extract_action_data(user_input)
        print("Improved user query is", extracted)
        
        if extracted:
            search_text = f"{extracted.get('action', '')} {extracted.get('data', '')}".strip()
            if not search_text:
                search_text = user_input
        else:
            search_text = user_input  # fallback

        websocket_response_service.emit_searching_status()

        # Search for matching API
        matched_api = search_service.search_api(user_input)
        print("Matched API: ", matched_api["method"], matched_api["path"])
        websocket_response_service.emit_found_api_status(matched_api["path"])
        
        websocket_response_service.emit_generating_status()
        
        # Use the search service to process the request
        result = search_service.process_api_request(user_input)
        
        if result.get('missing_fields'):
            websocket_response_service.emit_missing_fields({
                'message': 'Some required fields are missing',
                'missing_fields': result['missing_fields'],
                'current_params': result['current_params'],
                'matched_api': result['matched_api'],
                'request_method': result['request_method'],
                'action': result['action'],
                'user_input': result['user_input']
            })
            return
        
        if result.get('error'):
            websocket_response_service.emit_error(result['error'])
            return
        
        if result.get('status') == 'success':
            final_response = get_final_response(
                result['action'], 
                result['api_path'], 
                result['method'], 
                result['payload']
            )
            websocket_response_service.emit_api_response(final_response)

            time.sleep(1)
            next_best_items = suggest_next_best_item(result['action'], user_input)
            if next_best_items:
                websocket_response_service.emit_next_best_items(next_best_items)
        else:
            websocket_response_service.emit_error('Failed to process request')
            
    except Exception as e:
        print(f"Error processing request: {e}")
        print(traceback.format_exc())
        websocket_response_service.emit_error(f'An error occurred: {str(e)}')

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
        
        # Execute the plan step by step with WebSocket communication
        execution_result: Execution = multiple_requests_service.execute_plan_steps(
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

@socketio.on('visualize_search')
def handle_visualize_search(data):
    """Handle search visualization request."""
    try:
        query = data.get('query', '').strip()
        use_pca = data.get('use_pca', False)
        
        if not query:
            websocket_response_service.emit_error('No query provided for visualization')
            return
        
        print(f"Visualizing search for: {query}")
        websocket_response_service.emit_visualizing_status()
        
        # Get search analysis with visualization
        analysis = visualization_service.get_search_analysis(query, top_k=5)
        
        if analysis.get('error'):
            websocket_response_service.emit_error(analysis['error'])
            return
        
        # Send visualization results
        websocket_response_service.emit_visualization_results(query, analysis)
        
    except Exception as e:
        print(f"Error generating visualization: {e}")
        print(traceback.format_exc())
        websocket_response_service.emit_error(f'An error occurred: {str(e)}')

@socketio.on('provide_missing_fields')
def handle_missing_fields(data):
    """Handle missing fields provided by user."""
    try:
        provided_fields = data.get('fields', {})
        current_params = data.get('current_params', {})
        matched_api = data.get('matched_api', {})
        request_method = data.get('request_method', "GET")
        action = data.get('action', "")
        user_input = data.get('user_input', "")

        print("request method", request_method)
        print("matched api", matched_api)
        print("current params", current_params)
        print("provided fields", provided_fields)
        print("action", action)
        print("user input", user_input)
        
        # Update params with provided fields (handling nested structures)
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
            websocket_response_service.emit_error(f'API failed: {status_code}: {response_data}')
            return
        
        # Send final response
        final_response = get_final_response(action, sanitized_path, request_method, sanitized_params)
        websocket_response_service.emit_api_response(final_response)
        time.sleep(1)
        next_best_items = suggest_next_best_item(action, user_input)
        if next_best_items:
            websocket_response_service.emit_next_best_items(next_best_items)
        
    except Exception as e:
        print(f"Error handling missing fields: {e}")
        print(traceback.format_exc())
        websocket_response_service.emit_error(f'An error occurred: {str(e)}')

@socketio.on('make_api_call')
def handle_make_api_call(data):
    '''
    Make an API call to the API server
    using requests module
    '''
    try:
        response_data, status_code = make_api_call(
            data['method'], 
            data['api_path'], 
            data.get('payload')
        )
        if status_code < 300:
            websocket_response_service.emit_api_response({
                'response': response_data,
                'status': 'success'
            })
        else:
            websocket_response_service.emit_error(f'Failed to make API call: {status_code}: {response_data}')
    except Exception as e:
        print(f"Error making API call: {e}")
        websocket_response_service.emit_error(f'An error occurred: {str(e)}')

@socketio.on('upload_csv')
def handle_upload_csv(data):
    '''
    Upload a csv file to the server
    '''
    try:
        csv_content = data.get('csv_file')
        event_id = data.get('event_id')

        print("Uploading CSV to event", event_id)

        if not csv_content:
            websocket_response_service.emit_upload_error('No CSV content provided')
            return

        if not event_id:
            websocket_response_service.emit_upload_error('No event ID provided')
            return

        # Create a temporary CSV file
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            temp_file.write(csv_content)
            temp_file_path = temp_file.name

        try:
            print("Calling csv_uploader.js")
            # Use subprocess to call the csv_uploader.js file
            result = subprocess.run(
                ['node', 'csv_uploader.js', temp_file_path, event_id], 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True, 
                timeout=30
            )
            print("Result", result)
            
            # Clean up the temporary file
            os.unlink(temp_file_path)
            
            if result.returncode == 0:
                websocket_response_service.emit_upload_success({
                    'message': f'CSV uploaded successfully for event {event_id}!',
                    'output': result.stdout
                })
            else:
                websocket_response_service.emit_upload_error(f'Failed to upload CSV: {result.stderr}')
                
        except subprocess.TimeoutExpired:
            # Clean up the temporary file
            os.unlink(temp_file_path)
            websocket_response_service.emit_upload_error('Upload timed out. Please try again.')
        except FileNotFoundError:
            # Clean up the temporary file
            os.unlink(temp_file_path)
            websocket_response_service.emit_upload_error('csv_uploader.js not found. Please ensure the file exists.')
            
    except Exception as e:
        print(f"Error uploading CSV: {e}")
        print(traceback.format_exc())
        websocket_response_service.emit_upload_error(f'An error occurred: {str(e)}')

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