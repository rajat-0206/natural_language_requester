# API Requester WebSocket Server

A real-time WebSocket server that converts natural language requests into API calls using Claude AI. Users can describe what they want to do in plain English, and the system will generate the appropriate API call with a curl command.

## Features

- 🔗 **Real-time WebSocket communication** with the frontend
- 🤖 **Claude AI integration** for intelligent API matching and parameter generation
- 📝 **Interactive missing field collection** when required parameters are not provided
- 🎯 **Semantic search** to find the most relevant API endpoints
- 📋 **cURL command generation** for easy API testing
- 🎨 **Modern, responsive UI** with real-time status updates
- 🔄 **Multiple Requests Support** for complex workflows with step-by-step execution
- 📋 **Execution Planning** with user approval and modification capabilities
- 🚀 **Context-Aware Execution** that uses results from previous steps

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

Create a `.env` file in the project directory:

```bash
CLAUDE_API_KEY=your_claude_api_key_here
API_KEY=your_api_key_here
API_HOST=https://your-api-domain.com
```

### 3. Prepare API Schema

Ensure you have a `schema.json` file in the project directory with your API schema. The schema should contain:

```json
{
  "paths": {
    "/api/endpoint": {
      "get": {
        "parameters": [...],
        "requestBody": {...}
      }
    }
  }
}
```

### 4. Run the Server

```bash
python websocket_server.py
```

The server will start on `http://localhost:5000`

## Usage

### Web Interface

#### Single API Request
1. Open your browser and navigate to `http://localhost:5000`
2. Enter your request in natural language (e.g., "Create an event for tomorrow at 3 PM called 'Team Meeting' for 1 hour")
3. Click "Generate API Call" or press Ctrl+Enter
4. If required fields are missing, the system will prompt you to provide them
5. View the generated API call with the curl command

#### Multiple API Requests (Complex Workflows)
1. Enter a complex request that requires multiple steps (e.g., "Create an event and add 5 attendees")
2. Click "Multiple Requests" button
3. Review the generated execution plan with all steps
4. Choose to:
   - **Approve & Execute**: Run all steps automatically
   - **Modify Plan**: Provide feedback to change the plan
5. Watch real-time progress as each step executes
6. View final results with all step outcomes

**Example Complex Requests:**
- "Create an event for tomorrow, add speakers, and send invitations"
- "Get all events and create a broadcast for the first one"
- "Create event, add attendees, and create a discussion group"

### WebSocket Events

#### Client to Server Events

- `api_request`: Send a natural language query
  ```javascript
  socket.emit('api_request', { query: 'Create an event for tomorrow' });
  ```

- `provide_missing_fields`: Provide missing required fields
  ```javascript
  socket.emit('provide_missing_fields', {
    fields: { organization: 'my-org', owner: 'user123' },
    current_params: {...},
    matched_api: {...}
  });
  ```

- `multiple_requests`: Request multiple API calls for complex workflows
  ```javascript
  socket.emit('multiple_requests', { query: 'Create event and add attendees' });
  ```

- `approve_execution_plan`: Approve and execute a generated plan
  ```javascript
  socket.emit('approve_execution_plan', {
    plan: {...},
    user_input: 'Create event and add attendees'
  });
  ```

- `modify_execution_plan`: Request modification of an execution plan
  ```javascript
  socket.emit('modify_execution_plan', {
    plan: {...},
    feedback: 'Add a broadcast step after creating the event',
    user_input: 'Create event and add attendees'
  });
  ```

- `provide_multiple_requests_missing_fields`: Provide missing fields during plan execution
  ```javascript
  socket.emit('provide_multiple_requests_missing_fields', {
    fields: { organization: 'my-org', owner: 'user123' },
    current_params: {...},
    matched_api: {...},
    step_number: 2,
    plan: {...},
    user_input: 'Create event and add attendees',
    previous_results: {...},
    completed_steps: [...]
  });
  ```

#### Server to Client Events

- `status`: Status updates during processing
  ```javascript
  socket.on('status', (data) => {
    console.log(data.message); // Status message
    console.log(data.status);  // 'connected', 'processing', 'searching', etc.
  });
  ```

- `missing_fields`: When required fields are missing
  ```javascript
  socket.on('missing_fields', (data) => {
    console.log(data.missing_fields); // Array of missing field names
    console.log(data.current_params); // Current parameters
    console.log(data.matched_api);    // Matched API details
  });
  ```

- `api_response`: Final API call result
  ```javascript
  socket.on('api_response', (data) => {
    console.log(data.action);      // Action description
    console.log(data.api_path);    // API endpoint
    console.log(data.method);      // HTTP method
    console.log(data.payload);     // Request payload
    console.log(data.curl_command); // Generated curl command
  });
  ```

- `error`: Error messages
  ```javascript
  socket.on('error', (data) => {
    console.log(data.message); // Error message
  });
  ```

- `execution_plan`: Generated execution plan for multiple requests
  ```javascript
  socket.on('execution_plan', (data) => {
    console.log(data.plan);        // Execution plan object
    console.log(data.user_input);  // Original user input
    console.log(data.status);      // 'pending_approval'
  });
  ```

- `step_completed`: Real-time updates for each step execution
  ```javascript
  socket.on('step_completed', (data) => {
    console.log(data.step_number);  // Step number
    console.log(data.description);  // Step description
    console.log(data.status);       // 'success' or 'failed'
    console.log(data.result);       // Step result data
  });
  ```

- `multiple_requests_complete`: Final results of all steps
  ```javascript
  socket.on('multiple_requests_complete', (data) => {
    console.log(data.results);      // Complete execution results
    console.log(data.plan);         // Original execution plan
    console.log(data.status);       // 'completed'
  });
  ```

- `multiple_requests_missing_fields`: Missing fields during plan execution
  ```javascript
  socket.on('multiple_requests_missing_fields', (data) => {
    console.log(data.step_number);      // Current step number
    console.log(data.step_description); // Step description
    console.log(data.missing_fields);   // Array of missing field names
    console.log(data.current_params);   // Current parameters
    console.log(data.matched_api);      // Matched API details
    console.log(data.plan);             // Original execution plan
    console.log(data.previous_results); // Results from previous steps
  });
  ```

## Architecture

### Components

1. **WebSocket Server** (`websocket_server.py`): Flask-SocketIO server handling real-time communication
2. **API Requester** (`api_requester.py`): Core logic for API matching and parameter generation
3. **Frontend** (`templates/index.html`): Modern, responsive web interface
4. **Semantic Search** (`get_top_apis.py`): Vector search for finding relevant APIs

### Flow

#### Single Request Flow
1. **User Input**: User enters natural language request
2. **Query Enhancement**: Claude extracts action, data, and details
3. **API Matching**: Semantic search finds relevant API endpoints
4. **Parameter Generation**: Claude generates API parameters from user input
5. **Missing Fields**: If required fields are missing, prompt user
6. **Final Response**: Generate complete API call with curl command

#### Multiple Requests Flow
1. **Complex User Input**: User enters multi-step request
2. **Plan Generation**: Claude creates execution plan with sequential steps
3. **Plan Review**: User reviews and can modify the plan
4. **Step Execution**: Execute each step sequentially with context from previous steps
5. **Real-time Updates**: Show progress for each step execution
6. **Final Results**: Display comprehensive results from all steps

## Configuration

### API Configuration

Set the following environment variables for API calls:

```bash
# API Configuration
API_KEY=your_api_key_here
API_HOST=https://your-api-domain.com
```

The system will automatically use these values when making API calls during plan execution.

### Model Configuration

You can modify the Claude model and parameters in `websocket_server.py`:

```python
"model": "claude-3-5-sonnet-20241022",  # Change model as needed
"max_tokens": 1024,                     # Adjust token limit
```

## Error Handling

The system handles various error scenarios:

- **Missing API Key**: Prompts user to set environment variable
- **Invalid Schema**: Shows clear error message
- **Network Issues**: Graceful degradation with retry logic
- **Missing Fields**: Interactive collection of required information
- **Invalid Responses**: Fallback to error messages

## Development

### Adding New Features

1. **New API Endpoints**: Add to `schema.json`
2. **Custom Prompts**: Modify `build_prompt()` in `api_requester.py`
3. **UI Enhancements**: Update `templates/index.html`
4. **WebSocket Events**: Add new event handlers in `websocket_server.py`

### Testing

```bash
# Test the WebSocket server
python websocket_server.py

# Test with curl
curl -X POST http://localhost:5000/api_request \
  -H "Content-Type: application/json" \
  -d '{"query": "Create an event for tomorrow"}'

# Test multiple requests functionality
python test_multiple_requests.py
```

#### Multiple Requests Testing

The `test_multiple_requests.py` script tests all aspects of the multiple requests functionality:

- **Execution Plan Generation**: Tests creating plans for complex requests
- **Plan Modification**: Tests modifying plans based on user feedback
- **Context Enhancement**: Tests enhancing requests with previous step results
- **Prompt Building**: Tests building prompts with context from previous steps

Run the test to verify everything works correctly:

```bash
python test_multiple_requests.py
```

## Troubleshooting

### Common Issues

1. **"CLAUDE_API_KEY not set"**: Ensure your `.env` file contains the API key
2. **"schema.json not found"**: Verify the schema file is in the project directory
3. **WebSocket connection failed**: Check if the server is running on the correct port
4. **Import errors**: Install all dependencies with `pip install -r requirements.txt`

### Logs

The server provides detailed logging for debugging:

- Connection/disconnection events
- API matching results
- Claude API responses
- Error details

## License

This project is open source and available under the MIT License. 