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
    suggest_next_best_item
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
SEARCH_MODEL = "scann"  # or "scann"

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
            # Check if there are any required fields missing
            if params and any(value == "REQUIRED_FIELD_MISSING" for value in params.values()):
                missing_fields = [key for key, value in params.items() if value == "REQUIRED_FIELD_MISSING"]
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
        
        # Update params with provided fields
        updated_params = current_params.copy()
        updated_params.update(provided_fields)

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


if __name__ == '__main__':
    # Initialize the API system
    if initialize_api_system():
        print("✅ WebSocket Server is ready!")
        socketio.run(app, debug=True, host='0.0.0.0', port=8010)
    else:
        print("❌ Failed to initialize API system. Exiting.")