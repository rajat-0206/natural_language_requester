from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import json
import traceback
from api_requester import (
    build_prompt, 
    build_index, 
    load_cached_index, 
    save_index_to_cache, 
    embed
)
from utils import (
    sanitize_api_url,
    sanitize_api_params,
    call_claude, 
    parse_response, 
    extract_action_data,
    handle_get_request_params,
    generate_curl_command,
    make_api_call
)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
API_SCHEMA = None
index = None
metadata = None

def initialize_api_system():
    """Initialize the API schema and search index."""
    global API_SCHEMA, index, metadata
    
    try:
        with open("schema.json", "r") as f:
            API_SCHEMA = json.load(f)
    except FileNotFoundError:
        print("Error: `schema.json` not found. Please make sure the file is in the same directory.")
        return False

    # Initialize the semantic search index
    index, metadata, _, _, _ = load_cached_index()
    if index is None:
        print("Building new index...")
        index, metadata, _, _, _ = build_index(API_SCHEMA)
        save_index_to_cache(index, metadata, None, None)
        print("Index built and cached successfully!")
    else:
        print("Using cached index...")
    
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
        
        # Find the best matching API using semantic search
        query_emb = embed([search_text])
        _, top_ids = index.search(query_emb, 1)  # Get top 1 match
        matched_api = metadata[top_ids[0][0]]

        print("Matched API: ", matched_api)
        emit('status', {'message': f'Found API: {matched_api["path"]}', 'status': 'found_api'})
        
        # Use Claude to fill in the details
        prompt = build_prompt(
            user_input, 
            api_path=matched_api["path"], 
            api_method=matched_api["method"], 
            api_parameters=matched_api["parameters"], 
            api_request_body=matched_api["requestBody"]
        )
        
        emit('status', {'message': 'Generating API payload...', 'status': 'generating'})
        llm_response = call_claude(prompt)
        
        action, params, api = parse_response(llm_response)

        if matched_api["method"] == "GET":
            api, params, missing_fields = handle_get_request_params(api, params)
            if missing_fields:
                emit('missing_fields', {
                    'message': 'Some required fields are missing',
                    'missing_fields': missing_fields,
                    'current_params': params,
                    'matched_api': matched_api
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
                    'matched_api': matched_api
                })
                return
            
            # Generate curl command
            sanitized_path = sanitize_api_url(matched_api['path'])
            sanitized_params = json.loads(sanitize_api_params(params))

            curl_command = generate_curl_command(matched_api, sanitized_params, sanitized_path)
            
            # Send final response
            emit('api_response', {
                'action': action,
                'api_path': sanitized_path,
                'method': matched_api['method'],
                'payload': sanitized_params,
                'curl_command': curl_command,
                'status': 'success'
            })
            
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
        
        # Update params with provided fields
        updated_params = current_params.copy()
        updated_params.update(provided_fields)
        
        # Generate curl command
        sanitized_path = sanitize_api_url(matched_api['path'])
        sanitized_params = json.loads(sanitize_api_params(updated_params))
        curl_command = generate_curl_command(matched_api, sanitized_params, sanitized_path)
        
        # Send final response
        emit('api_response', {
            'action': 'API Request',
            'api_path': sanitized_path,
            'method': matched_api['method'],
            'payload': sanitized_params,
            'curl_command': curl_command,
            'status': 'success'
        })
        
    except Exception as e:
        print(f"Error handling missing fields: {e}")
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

if __name__ == '__main__':
    # Initialize the API system
    if initialize_api_system():
        print("✅ WebSocket Server is ready!")
        socketio.run(app, debug=True, host='0.0.0.0', port=8010)
    else:
        print("❌ Failed to initialize API system. Exiting.")