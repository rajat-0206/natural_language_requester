from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import json
import subprocess
import traceback
from api_requester import (
    build_prompt, 
    build_index, 
    load_cached_index, 
    save_index_to_cache,
    embed
)
from get_top_apis import (
    build_scann_index,
    load_cached_scann_index,
    save_scann_index_to_cache,
    search_apis_scann,
    search_apis
)
import time
from utils import (
    sanitize_api_url,
    sanitize_api_params,
    call_model, 
    parse_response, 
    extract_action_data,
    handle_get_request_params,
    generate_curl_command,
    make_api_call,
    suggest_next_best_item,
    get_current_time_for_prompt
)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
API_SCHEMA = None
index = None
metadata = None
scann_index = None
scann_metadata = None
SEARCH_MODEL = "faiss"  # or "scann"

def initialize_api_system(search_model="faiss"):
    """Initialize the API schema and search index."""
    global API_SCHEMA, index, metadata, scann_index, scann_metadata, SEARCH_MODEL
    SEARCH_MODEL = search_model
    try:
        with open("schema.json", "r") as f:
            API_SCHEMA = json.load(f)
    except FileNotFoundError:
        print("Error: `schema.json` not found. Please make sure the file is in the same directory.")
        return False

    if SEARCH_MODEL == "faiss":
        # Initialize the semantic search index (FAISS)
        index, metadata, _, _, _ = load_cached_index()
        if index is None:
            print("Building new index...")
            index, metadata, _, _, _ = build_index(API_SCHEMA)
            save_index_to_cache(index, metadata, None, None)
            print("Index built and cached successfully!")
        else:
            print("Using cached index...")
    elif SEARCH_MODEL == "scann":
        # Initialize the semantic search index (SCANN)
        scann_index, scann_metadata, _, _, _ = load_cached_scann_index()
        if scann_index is None:
            print("Building new SCANN index...")
            scann_index, scann_metadata, _, _, _ = build_scann_index(API_SCHEMA)
            save_scann_index_to_cache(scann_index, scann_metadata, None, None)
            print("SCANN index built and cached successfully!")
        else:
            print("Using cached SCANN index...")
    else:
        raise ValueError(f"Unknown search model: {SEARCH_MODEL}")
    return True

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
    """Handle API request from frontend."""
    try:
        user_input = data.get('query', '').strip()
        
        if not user_input:
            emit('error', {'message': 'No query provided'})
            return
        
        print(f"Received request: {user_input}")
        emit('status', {'message': 'Processing your request...', 'status': 'processing'})
        
        # Extract action/data/details using Claude
        extracted = extract_action_data(user_input)
        print("Improved user query is", extracted)
        
        if extracted:
            search_text = f"{extracted.get('action', '')} {extracted.get('data', '')}".strip()
            if not search_text:
                search_text = user_input
        else:
            search_text = user_input  # fallback

        emit('status', {'message': 'Finding matching API...', 'status': 'searching'})

        # Use the selected search model
        if SEARCH_MODEL == "faiss":
            query_emb = embed([search_text])
            _, top_ids = index.search(query_emb, 1)
            matched_api = metadata[top_ids[0][0]]
        elif SEARCH_MODEL == "scann":
            results = search_apis_scann(search_text, scann_index, scann_metadata, None, None, top_k=1)
            matched_api = results[0]['api']
        else:
            raise ValueError(f"Unknown search model: {SEARCH_MODEL}")

        print("Matched API: ", matched_api)
        emit('status', {'message': f'Found API: {matched_api["path"]}', 'status': 'found_api'})
        
        prompt = build_prompt(
            user_input, 
            api_path=matched_api["path"], 
            api_method=matched_api["method"], 
            api_parameters=matched_api["parameters"], 
            api_request_body=matched_api["requestBody"]
        )
        
        emit('status', {'message': 'Generating API payload...', 'status': 'generating'})
        llm_response = call_model(prompt)
        
        action, params, api = parse_response(llm_response)

        if matched_api["method"] == "GET":
            api, params, missing_fields = handle_get_request_params(api, params)
            if missing_fields:
                emit('missing_fields', {
                    'message': 'Some required fields are missing',
                    'missing_fields': missing_fields,
                    'current_params': params,
                    'matched_api': api,
                    'request_method': matched_api['method'],
                    'action': action,
                    'user_input': user_input
                })
                return


        if action and action != 'more_info_needed':
            # Check if there are any required fields missing (including nested ones)
            missing_fields = find_missing_fields_nested(params)
            if missing_fields:
                emit('missing_fields', {
                    'message': 'Some required fields are missing',
                    'missing_fields': missing_fields,
                    'current_params': params,
                    'matched_api': api,
                    'request_method': matched_api['method'],
                    'action': action,
                    'user_input': user_input
                })
                return
            
            final_response = get_final_response(action, api, matched_api['method'], params)

            emit('api_response', final_response)

            time.sleep(1)
            next_best_items = suggest_next_best_item(action, user_input)
            if next_best_items:
                emit('next_best_items', next_best_items)
            
        elif action == 'more_info_needed':
            emit('error', {'message': f'More information needed: {params.get("text", "No details provided.")}'})
        else:
            emit('error', {'message': "I couldn't determine the correct action. Please try rephrasing your request."})
            
    except Exception as e:
        print(f"Error processing request: {e}")
        print(traceback.format_exc())
        emit('error', {'message': f'An error occurred: {str(e)}'})

@socketio.on('multiple_requests')
def handle_multiple_requests(data):
    try:
        user_input = data.get('query', '').strip()
        
        if not user_input:
            emit('error', {'message': 'No query provided'})
            return
        
        print(f"Received multiple requests: {user_input}")
        emit('status', {'message': 'Analyzing your complex request...', 'status': 'analyzing'})
        
        # Step 1: Generate execution plan
        execution_plan = generate_execution_plan(user_input)
        
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
    """Handle user approval of execution plan and start execution."""
    try:
        plan = data.get('plan', {})
        user_input = data.get('user_input', '')
        
        if not plan or not plan.get('steps'):
            emit('error', {'message': 'Invalid execution plan'})
            return
        
        emit('status', {'message': 'Starting execution of approved plan...', 'status': 'executing'})
        
        # Execute the plan step by step
        results = execute_plan_steps(plan, user_input)
        
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
        modified_plan = modify_execution_plan(original_plan, user_feedback, user_input)
        
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
        user_input = data.get('user_input', '')
        previous_results = data.get('previous_results', {})
        completed_steps = data.get('completed_steps', [])
        
        print(f"Providing missing fields for step {step_number}")
        print("Provided fields:", provided_fields)
        
        # Use the same logic as handle_missing_fields
        updated_params = update_nested_dict(current_params, provided_fields)
        
        if matched_api.get('method') == "GET":
            api = matched_api + "?" + "&".join([f"{key}={value}" for key, value in updated_params.items()])
        else:
            api = matched_api
        
        # Make the API call
        sanitized_path = sanitize_api_url(api)
        sanitized_params = json.loads(sanitize_api_params(updated_params))
        
        response_data, status_code = make_api_call(
            matched_api.get('method', 'GET'), 
            sanitized_path, 
            sanitized_params
        )
        
        if response_data is None:
            emit('error', {'message': f'API call failed for step {step_number}'})
            return
        
        # Create step result
        step_result = {
            'action': f'Step {step_number}',
            'api_path': api,
            'method': matched_api.get('method', 'GET'),
            'payload': sanitized_params,
            'response': response_data,
            'status_code': status_code,
            'status': 'success',
            'step_number': step_number,
            'step_description': data.get('step_description', ''),
            'api_request': data.get('api_request', '')
        }
        
        # Store result for future steps
        step = next((s for s in plan.get('steps', []) if s.get('step_number') == step_number), None)
        if step and step.get('result_key'):
            previous_results[step.get('result_key')] = step_result
        
        # Add to completed steps
        completed_steps.append({
            'step_number': step_number,
            'description': data.get('step_description', ''),
            'status': 'success',
            'result': step_result,
            'api_request': data.get('api_request', '')
        })
        
        # Send step completion update
        emit('step_completed', {
            'step_number': step_number,
            'description': data.get('step_description', ''),
            'result': step_result,
            'status': 'success'
        })
        
        # Continue execution from the next step
        emit('status', {'message': f'Step {step_number} completed. Continuing execution...', 'status': 'executing_step'})
        
        # Resume execution from the next step
        remaining_steps = [s for s in plan.get('steps', []) if s.get('step_number') > step_number]
        
        for step in remaining_steps:
            try:
                step_num = step.get('step_number', 0)
                step_desc = step.get('description', '')
                api_request = step.get('api_request', '')
                
                print(f"Executing step {step_num}: {step_desc}")
                emit('status', {'message': f'Executing step {step_num}: {step_desc}', 'status': 'executing_step'})
                
                # Execute the step with updated previous results
                step_result = execute_single_step(step, previous_results, user_input)
                
                # Check if step failed due to missing fields
                if step_result and step_result.get('missing_fields'):
                    # Pause execution and request missing fields from user
                    emit('multiple_requests_missing_fields', {
                        'step_number': step_num,
                        'step_description': step_desc,
                        'missing_fields': step_result['missing_fields'],
                        'current_params': step_result['current_params'],
                        'matched_api': step_result['matched_api'],
                        'api_request': api_request,
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
                
                completed_steps.append({
                    'step_number': step_num,
                    'description': step_desc,
                    'status': 'success' if step_result and step_result.get('status') == 'success' else 'failed',
                    'result': step_result,
                    'api_request': api_request
                })
                
                # Send step completion update
                emit('step_completed', {
                    'step_number': step_num,
                    'description': step_desc,
                    'result': step_result,
                    'status': 'success' if step_result and step_result.get('status') == 'success' else 'failed'
                })
                
            except Exception as e:
                print(f"Error executing step {step_num}: {e}")
                completed_steps.append({
                    'step_number': step_num,
                    'description': step_desc,
                    'status': 'failed',
                    'error': str(e),
                    'api_request': api_request
                })
                
                emit('step_completed', {
                    'step_number': step_num,
                    'description': step_desc,
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Send final results
        emit('multiple_requests_complete', {
            'results': {
                'plan_description': plan.get('description', ''),
                'final_result': plan.get('final_result', ''),
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

def extract_json_from_response(response):
    """
    Extract the last valid JSON object from a string by finding the first '{' and the last matching '}'.
    Handles nested braces and ignores text before/after the JSON.
    """
    if not response:
        return "{}"
    
    start = response.find('{')
    if start == -1:
        return "{}"
    
    # Scan for the last valid JSON object using brace counting
    brace_count = 0
    end = -1
    for i in range(start, len(response)):
        if response[i] == '{':
            brace_count += 1
        elif response[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                end = i
                # Don't break, keep looking for a later valid object
    if end == -1:
        return "{}"
    return response[start:end+1].strip()

def generate_execution_plan(user_input):
    """Generate an execution plan for multiple API requests."""
    prompt = f"""
You are an expert at breaking down complex API requests into sequential steps.
Given the user's request, create a detailed execution plan with multiple API calls.
The plan should only contain API execution steps. Any data manipulation will be done at later step when we call the api.
ONLY RETURN  VALID JSON OBJECT. IT SHOULD BE JSON PARSABLE.


**USER REQUEST:** {user_input}

**TASK:** Create a JSON plan with the following structure:
{{
  "description": "Brief description of what this plan accomplishes",
  "steps": [
    {{
      "step_number": 1,
      "description": "What this step does",
      "api_request": "Natural language description of the API call needed",
      "depends_on": [], // List of step numbers this step depends on
      "expected_result": "What we expect to get from this step",
      "result_key": "key_name" // Key to store the result for future steps
    }}
  ],
  "final_result": "Description of the final outcome"
}}

**RULES:**
1. Break down complex requests into logical sequential steps
2. Each step should be a single API call
3. Use depends_on to indicate step dependencies
4. Use result_key to store results for use in later steps
5. Make sure steps are in the correct order
6. Each step should be clear and actionable

**EXAMPLES:**
- "Create an event and add 5 attendees" → 2 steps: create event, then add attendees
- "Get all events and create a broadcast for the first one" → 2 steps: get events, then create broadcast
- "Create event, add speakers, and send invitations" → 3 steps: create event, add speakers, send invitations
"""
    
    response = call_model(prompt)
    try:
        json_str = extract_json_from_response(response)
        return json.loads(json_str)
    except Exception as e:
        print(f"Failed to parse execution plan: {e}")
        return None

def modify_execution_plan(original_plan, user_feedback, user_input):
    """Modify an execution plan based on user feedback."""
    prompt = f"""
You are modifying an execution plan based on user feedback.

**ORIGINAL USER REQUEST:** {user_input}

**ORIGINAL PLAN:** {json.dumps(original_plan, indent=2)}

**USER FEEDBACK:** {user_feedback}

**TASK:** Modify the plan according to the user's feedback and return the updated plan in the same JSON format.

**RULES:**
1. Keep the same structure as the original plan
2. Modify steps based on user feedback
3. Ensure dependencies are still correct
4. Make sure the plan still accomplishes the original goal

Return only the JSON object, no additional text.
"""
    
    response = call_model(prompt)
    try:
        json_str = extract_json_from_response(response)
        return json.loads(json_str)
    except Exception as e:
        print(f"Failed to parse modified plan: {e}")
        return None

def execute_plan_steps(plan, user_input):
    """Execute the steps in the plan sequentially."""
    results = {}
    step_results = []
    
    for step in plan.get('steps', []):
        try:
            step_num = step.get('step_number', 0)
            step_desc = step.get('description', '')
            api_request = step.get('api_request', '')
            
            print(f"Executing step {step_num}: {step_desc}")
            emit('status', {'message': f'Executing step {step_num}: {step_desc}', 'status': 'executing_step'})
            
            # Execute the step
            step_result = execute_single_step(step, results, user_input)
            
            # Check if step failed due to missing fields
            if step_result and step_result.get('missing_fields'):
                # Pause execution and request missing fields from user
                emit('multiple_requests_missing_fields', {
                    'step_number': step_num,
                    'step_description': step_desc,
                    'missing_fields': step_result['missing_fields'],
                    'current_params': step_result['current_params'],
                    'matched_api': step_result['matched_api'],
                    'api_request': api_request,
                    'plan': plan,
                    'user_input': user_input,
                    'previous_results': results,
                    'completed_steps': step_results
                })
                return None  # Pause execution
            
            # Store result for future steps
            result_key = step.get('result_key')
            if result_key and step_result and step_result.get('status') == 'success':
                results[result_key] = step_result
            
            step_results.append({
                'step_number': step_num,
                'description': step_desc,
                'status': 'success' if step_result and step_result.get('status') == 'success' else 'failed',
                'result': step_result,
                'api_request': api_request
            })
            
            # Send step completion update
            emit('step_completed', {
                'step_number': step_num,
                'description': step_desc,
                'result': step_result,
                'status': 'success' if step_result and step_result.get('status') == 'success' else 'failed'
            })
            
        except Exception as e:
            print(f"Error executing step {step_num}: {e}")
            step_results.append({
                'step_number': step_num,
                'description': step_desc,
                'status': 'failed',
                'error': str(e),
                'api_request': api_request
            })
            
            emit('step_completed', {
                'step_number': step_num,
                'description': step_desc,
                'status': 'failed',
                'error': str(e)
            })
    
    return {
        'plan_description': plan.get('description', ''),
        'final_result': plan.get('final_result', ''),
        'step_results': step_results,
        'overall_status': 'completed'
    }

def process_api_request(user_input, previous_results=None):
    """
    Process a single API request using the same logic as handle_api_request.
    Returns the result without emitting WebSocket events.
    """
    try:
        if not user_input:
            return {'error': 'No query provided'}
        
        print(f"Processing API request: {user_input}")
        
        # Extract action/data/details using Claude
        extracted = extract_action_data(user_input)
        print("Improved user query is", extracted)
        
        if extracted:
            search_text = f"{extracted.get('action', '')} {extracted.get('data', '')}".strip()
            if not search_text:
                search_text = user_input
        else:
            search_text = user_input  # fallback

        # Use the selected search model
        if SEARCH_MODEL == "faiss":
            query_emb = embed([search_text])
            _, top_ids = index.search(query_emb, 1)
            matched_api = metadata[top_ids[0][0]]
        elif SEARCH_MODEL == "scann":
            results = search_apis_scann(search_text, scann_index, scann_metadata, None, None, top_k=1)
            matched_api = results[0]['api']
        else:
            raise ValueError(f"Unknown search model: {SEARCH_MODEL}")

        print("Matched API: ", matched_api)
        
        # Build prompt with context if available
        if previous_results:
            prompt = build_prompt_with_context(
                user_input, 
                matched_api["path"], 
                matched_api["method"], 
                matched_api["parameters"], 
                matched_api["requestBody"],
                previous_results
            )
        else:
            prompt = build_prompt(
                user_input, 
                api_path=matched_api["path"], 
                api_method=matched_api["method"], 
                api_parameters=matched_api["parameters"], 
                api_request_body=matched_api["requestBody"]
            )
        
        llm_response = call_model(prompt)
        action, params, api = parse_response(llm_response)

        if matched_api["method"] == "GET":
            api, params, missing_fields = handle_get_request_params(api, params)
            if missing_fields:
                return {
                    'missing_fields': missing_fields,
                    'current_params': params,
                    'matched_api': api,
                    'request_method': matched_api['method'],
                    'action': action,
                    'user_input': user_input
                }

        if action and action != 'more_info_needed':
            # Check if there are any required fields missing (including nested ones)
            missing_fields = find_missing_fields_nested(params)
            if missing_fields:
                return {
                    'missing_fields': missing_fields,
                    'current_params': params,
                    'matched_api': api,
                    'request_method': matched_api['method'],
                    'action': action,
                    'user_input': user_input
                }
            
            # Make the API call
            sanitized_path = sanitize_api_url(api)
            sanitized_params = json.loads(sanitize_api_params(params))
            
            response_data, status_code = make_api_call(
                matched_api['method'], 
                sanitized_path, 
                sanitized_params
            )
            
            if response_data is None:
                return {
                    'error': 'API call failed',
                    'status': 'failed',
                    'action': action,
                    'api_path': api,
                    'method': matched_api['method'],
                    'payload': sanitized_params
                }
            
            return {
                'action': action,
                'api_path': api,
                'method': matched_api['method'],
                'payload': sanitized_params,
                'response': response_data,
                'status_code': status_code,
                'status': 'success'
            }
            
        elif action == 'more_info_needed':
            return {
                'error': f'More information needed: {params.get("text", "No details provided.")}'
            }
        else:
            return {
                'error': "I couldn't determine the correct action. Please try rephrasing your request."
            }
            
    except Exception as e:
        print(f"Error processing API request: {e}")
        print(traceback.format_exc())
        return {'error': f'An error occurred: {str(e)}'}

def execute_single_step(step, previous_results, user_input):
    """Execute a single step in the plan using the process_api_request helper."""
    try:
        api_request = step.get('api_request', '')
        step_number = step.get('step_number', 0)
        step_description = step.get('description', '')
        
        print(f"Executing step {step_number}: {step_description}")
        print(f"API request: {api_request}")
        
        # Enhance the API request with context from previous results
        enhanced_request = enhance_request_with_context(api_request, previous_results, user_input)
        print(f"Enhanced request: {enhanced_request}")
        
        # Use the process_api_request helper (same logic as handle_api_request)
        result = process_api_request(enhanced_request, previous_results)
        
        # Add step metadata to the result
        result['step_number'] = step_number
        result['step_description'] = step_description
        result['api_request'] = api_request
        
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
            'api_request': step.get('api_request', '')
        }

def enhance_request_with_context(api_request, previous_results, user_input):
    """Enhance the API request with context from previous results."""
    if not previous_results:
        return api_request
    
    # Extract useful information from previous results
    context_info = []
    for key, result in previous_results.items():
        if isinstance(result, dict) and result.get('status') == 'success':
            response_data = result.get('response', {})
            if isinstance(response_data, dict):
                # Extract common fields that might be useful
                if 'id' in response_data:
                    context_info.append(f"{key}_id: {response_data['id']}")
                if 'event_id' in response_data:
                    context_info.append(f"event_id: {response_data['event_id']}")
                if 'broadcast_id' in response_data:
                    context_info.append(f"broadcast_id: {response_data['broadcast_id']}")
                if 'user_id' in response_data:
                    context_info.append(f"user_id: {response_data['user_id']}")
                # Add the full response for complex cases
                context_info.append(f"{key}_response: {json.dumps(response_data, indent=2)}")
    
    context_prompt = f"""
Enhance the following API request with context from previous results.

**ORIGINAL REQUEST:** {api_request}
**USER INPUT:** {user_input}

**PREVIOUS RESULTS CONTEXT:**
{chr(10).join(context_info)}

**TASK:** Enhance the API request by incorporating relevant information from previous results.
For example:
- If a previous step created an event with ID "123", use that ID in subsequent requests
- If a previous step returned user data, use that data in the current request
- Replace placeholders like "the event", "the broadcast", "the user" with actual IDs or data from previous results
- Use specific IDs like event_id, broadcast_id, user_id when available

**RULES:**
1. Keep the request natural and clear
2. Replace generic references with specific IDs when available
3. Maintain the original intent of the request
4. Don't add unnecessary information

Return only the enhanced request text, no additional formatting or explanations.
"""
    
    response = call_model(context_prompt)
    enhanced = response.strip() if response else api_request
    
    print(f"Enhanced request from '{api_request}' to '{enhanced}'")
    return enhanced

def build_prompt_with_context(user_input, api_path, api_method, api_parameters, api_request_body, previous_results):
    """Build a prompt with context from previous results."""
    current_time = get_current_time_for_prompt()
    
    context_info = ""
    if previous_results:
        context_info = f"""
**CONTEXT FROM PREVIOUS STEPS:**
{json.dumps(previous_results, indent=2)}

Use the information from previous steps to fill in the current request.
For example, if a previous step created an event with ID "123", use that ID in the current request.
"""

    prompt = f"""
You are an expert AI assistant that fills in API request details based on user input and context from previous steps.
Your task is to analyze the user's request and the matched API endpoint to generate a valid JSON payload.

**CONTEXT:**
- The current date and time is: {current_time}. You MUST use this to resolve relative times like "tomorrow" or "next week".
- All dates and times in the final JSON MUST be in ISO 8601 format (e.g., "YYYY-MM-DDTHH:MM:SS").
{context_info}

**MATCHED API:**
```json
{{
  "path": "{api_path}",
  "method": "{api_method}",
  "parameters": {json.dumps(api_parameters, indent=2)},
  "requestBody": {json.dumps(api_request_body, indent=2)}
}}
```

**INSTRUCTIONS:**
1. Analyze the user's request and extract all necessary information.
2. Use context from previous steps to fill in missing information (IDs, data, etc.).
3. Fill in the API path parameters if required (replace {{param}} with actual values).
4. If a required parameter is missing and cannot be inferred, use "REQUIRED_FIELD_MISSING".
5. If request method is GET only add payload as query parameters and take default values as limit=10 and offset=0
6. For dates and times:
   - Convert relative times (e.g., "tomorrow", "next week") to actual dates
   - Use the current timezone (Asia/Kolkata) if not specified
   - Format all dates in ISO 8601 format
7. PLEASE DO NOT RETURN ANYTHING OTHER THAN THE JSON OBJECT. IF YOU WANT TO PASS A NOTE, ADD IT IN JSON OBJECT AS "note": "..." SINCE 
WE ARE USING THIS JSON OBJECT TO PARSE RESPONSE AND DISPLAY IT TO THE USER.
8. If logged_in_user is present then use it as owner field in payload if required.
9. If organization is present then use it as organization field in payload if required.
10. You will also get a currentPage object, if it is present it will contain either (or all) event_id, broadcast_id. Use that info while building
the request payload.

**CURRENT TASK:**
User: {user_input}
Response:"""

    return prompt.strip()

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
        
        # Generate curl command
        sanitized_path = sanitize_api_url(api)
        sanitized_params = json.loads(sanitize_api_params(updated_params))
        final_response = get_final_response("API Request", sanitized_path, request_method, sanitized_params)
        # Send final response
        emit('api_response', final_response)
        time.sleep(1)
        next_best_items = suggest_next_best_item(action, user_input)
        if next_best_items:
            emit('next_best_items', next_best_items)
        
    except Exception as e:
        print(f"Error handling missing fields: {e}")
        print(traceback.format_exc())
        emit('error', {'message': f'An error occurred: {str(e)}'})

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
        if response_data:
            emit('api_response', {
                'response': response_data,
                'status': 'success'
            })
        else:
            emit('error', {'message': 'Failed to make API call'})
    except Exception as e:
        print(f"Error making API call: {e}")
        emit('error', {'message': f'An error occurred: {str(e)}'})

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
            emit('upload_error', {'message': 'No CSV content provided'})
            return

        if not event_id:
            emit('upload_error', {'message': 'No event ID provided'})
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
                emit('upload_success', {
                    'message': f'CSV uploaded successfully for event {event_id}!',
                    'output': result.stdout
                })
            else:
                emit('upload_error', {
                    'message': f'Failed to upload CSV: {result.stderr}',
                    'details': result.stdout
                })
                
        except subprocess.TimeoutExpired:
            # Clean up the temporary file
            os.unlink(temp_file_path)
            emit('upload_error', {'message': 'Upload timed out. Please try again.'})
        except FileNotFoundError:
            # Clean up the temporary file
            os.unlink(temp_file_path)
            emit('upload_error', {'message': 'csv_uploader.js not found. Please ensure the file exists.'})
            
    except Exception as e:
        print(f"Error uploading CSV: {e}")
        print(traceback.format_exc())
        emit('upload_error', {'message': f'An error occurred: {str(e)}'})

def get_final_response(action, api_path, method, payload):
    return {
            'action': action,
            'api_path': api_path,
            'method': method,
            'payload': payload,
            'status': 'success'
        }

def flatten_nested_dict(data, parent_key='', sep='.'):
    """Flatten a nested dictionary, creating dot-notation keys for nested objects."""
    items = []
    for k, v in data.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def unflatten_dict(data, sep='.'):
    """Convert flattened dictionary back to nested structure."""
    result = {}
    for key, value in data.items():
        keys = key.split(sep)
        current = result
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    return result

def find_missing_fields_nested(params):
    """Find all missing fields in a nested structure and return flattened keys."""
    missing_fields = []
    
    def check_nested(obj, prefix=''):
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_key = f"{prefix}.{key}" if prefix else key
                if value == "REQUIRED_FIELD_MISSING":
                    missing_fields.append(current_key)
                elif isinstance(value, dict):
                    check_nested(value, current_key)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                current_key = f"{prefix}[{i}]" if prefix else f"[{i}]"
                if isinstance(item, dict):
                    check_nested(item, current_key)
    
    check_nested(params)
    return missing_fields

def update_nested_dict(original_dict, updates):
    """Update a nested dictionary with flattened key-value pairs."""
    result = original_dict.copy()
    
    for key, value in updates.items():
        keys = key.split('.')
        current = result
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    
    return result


if __name__ == '__main__':
    # Initialize the API system
    if initialize_api_system():
        print("✅ WebSocket Server is ready!")
        socketio.run(app, debug=True, host='0.0.0.0', port=8010)
    else:
        print("❌ Failed to initialize API system. Exiting.")