from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import json
import urllib.parse
import os
from datetime import datetime, timedelta
import pytz
from api_requester import (
    build_prompt, 
    call_claude, 
    parse_response, 
    collect_missing_fields,
    extract_action_data,
    build_index, 
    load_cached_index, 
    save_index_to_cache, 
    embed
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
            parsed_url = urllib.parse.urlparse(api)
            query_params = urllib.parse.parse_qs(parsed_url.query)
            missing_fields = []
            for key, value in query_params.items():
                if value[0] == "REQUIRED_FIELD_MISSING":
                    missing_fields.append(key)
                    query_params[key] = value[0]
                else:
                    query_params[key] = value[0]
                    
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
            curl_command = generate_curl_command(matched_api, params)
            
            # Send final response
            emit('api_response', {
                'action': action,
                'api_path': matched_api['path'],
                'method': matched_api['method'],
                'payload': params,
                'curl_command': curl_command,
                'status': 'success'
            })
            
        elif action == 'more_info_needed':
            emit('error', {'message': f'More information needed: {params.get("text", "No details provided.")}'})
        else:
            emit('error', {'message': "I couldn't determine the correct action. Please try rephrasing your request."})
            
    except Exception as e:
        print(f"Error processing request: {e}")
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
        curl_command = generate_curl_command(matched_api, updated_params)
        
        # Send final response
        emit('api_response', {
            'action': 'API Request',
            'api_path': matched_api['path'],
            'method': matched_api['method'],
            'payload': updated_params,
            'curl_command': curl_command,
            'status': 'success'
        })
        
    except Exception as e:
        print(f"Error handling missing fields: {e}")
        emit('error', {'message': f'An error occurred: {str(e)}'})

@socketio.on('make_api_call')
def make_api_call(data):
    '''
    Make an API call to the API server
    using requests module
    '''
    try:
        response = requests.request(data['method'], data['api_path'], json=data['payload'])
        emit('api_response', {
            'response': response.json(),
            'status': 'success'
        })
    except Exception as e:
        print(f"Error making API call: {e}")
        emit('error', {'message': f'An error occurred: {str(e)}'})

def sanitize_api_url(path):
    '''
    Sanitize the API path and parameters to convert any random value to correct type like tomorrow becomes actual data
    '''
    prompt = f"""
    Sanitize the API path {path} to convert any random value to correct type like tomorrow becomes actual data. Only return the sanitized text.
    If path is already sanitized, return the same path.
    """
    response = call_claude(prompt)
    return response

def sanitize_api_params(params):
    '''
    Sanitize the API parameters to convert any random value to correct type like tomorrow becomes actual data
    '''
    prompt = f"""
    Sanitize the API parameters {params} to convert any random value to correct type like tomorrow becomes actual data. Return the sanitized parameters in JSON format.
    DO NOT ADD ANYTHING ELSE TO THE RESPONSE.
    If parameters are already sanitized, return the same parameters.
    """
    response = call_claude(prompt)
    return response

def generate_curl_command(api, params):
    """Generate a curl command from the API and parameters."""
    # use llm to again sanitize the api and params to convert any random value to correct type like tomorrow becomes actual data
    method = api['method'].upper()
    path = api['path']

    sanitized_path = sanitize_api_url(path)
    sanitized_params = json.loads(sanitize_api_params(params))

    print("sanitized_path", sanitized_path)
    print("sanitized_params", sanitized_params)

    
    # Replace path parameters
    for param_name, param_value in sanitized_params.items():
        if f"{{{param_name}}}" in sanitized_path:
            sanitized_path = sanitized_path.replace(f"{{{param_name}}}", str(param_value))
    
    curl_parts = [f"curl -X {method}"]
    
    # Add headers
    curl_parts.append('-H "Content-Type: application/json"')
    curl_parts.append('-H "Authorization: Bearer YOUR_API_KEY"')
    
    # Add URL
    base_url = "https://api.example.com"  # Replace with your actual base URL
    url = f"{base_url}{sanitized_path}"
    curl_parts.append(f'"{url}"')
    
    # Add body for non-GET requests
    if method != "GET" and sanitized_params:
        # Remove path parameters from body
        body_params = {k: v for k, v in sanitized_params.items() if f"{{{k}}}" not in sanitized_path}
        if body_params:
            curl_parts.append(f"-d '{json.dumps(body_params)}'")
    
    return " ".join(curl_parts)

if __name__ == '__main__':
    # Initialize the API system
    if initialize_api_system():
        print("✅ WebSocket Server is ready!")
        socketio.run(app, debug=True, host='0.0.0.0', port=8010)
    else:
        print("❌ Failed to initialize API system. Exiting.")