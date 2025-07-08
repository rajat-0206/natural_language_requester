# API Requester WebSocket Server

A real-time WebSocket server that converts natural language requests into API calls using Claude AI. Users can describe what they want to do in plain English, and the system will generate the appropriate API call with a curl command.

## Features

- 🔗 **Real-time WebSocket communication** with the frontend
- 🤖 **Claude AI integration** for intelligent API matching and parameter generation
- 📝 **Interactive missing field collection** when required parameters are not provided
- 🎯 **Semantic search** to find the most relevant API endpoints
- 📋 **cURL command generation** for easy API testing
- 🎨 **Modern, responsive UI** with real-time status updates

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

Create a `.env` file in the project directory:

```bash
CLAUDE_API_KEY=your_claude_api_key_here
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

1. Open your browser and navigate to `http://localhost:5000`
2. Enter your request in natural language (e.g., "Create an event for tomorrow at 3 PM called 'Team Meeting' for 1 hour")
3. Click "Generate API Call" or press Ctrl+Enter
4. If required fields are missing, the system will prompt you to provide them
5. View the generated API call with the curl command

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

## Architecture

### Components

1. **WebSocket Server** (`websocket_server.py`): Flask-SocketIO server handling real-time communication
2. **API Requester** (`api_requester.py`): Core logic for API matching and parameter generation
3. **Frontend** (`templates/index.html`): Modern, responsive web interface
4. **Semantic Search** (`get_top_apis.py`): Vector search for finding relevant APIs

### Flow

1. **User Input**: User enters natural language request
2. **Query Enhancement**: Claude extracts action, data, and details
3. **API Matching**: Semantic search finds relevant API endpoints
4. **Parameter Generation**: Claude generates API parameters from user input
5. **Missing Fields**: If required fields are missing, prompt user
6. **Final Response**: Generate complete API call with curl command

## Configuration

### Base URL

Update the base URL in `websocket_server.py`:

```python
base_url = "https://your-api-domain.com"  # Replace with your actual base URL
```

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