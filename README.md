# App Interaction Agent

An intelligent agent that helps users perform actions in your application through natural language. Instead of navigating complex interfaces or remembering specific API calls, users can simply describe what they want to do in plain English, and the agent will figure out the right actions to take.

## What It Does

The App Interaction Agent understands natural language requests and automatically performs the corresponding actions in your application. It works by:

1. **Understanding Intent**: Analyzes your request to understand what you want to accomplish
2. **Finding the Right Tools**: Uses AI-powered search to identify the appropriate API endpoints from your application's documentation
3. **Executing Actions**: Automatically calls the right APIs with the correct parameters
4. **Handling Complex Workflows**: Can break down complex requests into multiple steps and execute them sequentially

## Capabilities

### 🎯 **Single Actions**
Perform individual tasks with simple natural language:
- "Create an event for tomorrow at 3 PM called 'Team Meeting'"
- "Get a user's magic link"
- "Add a speaker to event ID 123"
- "Create a broadcast for the upcoming webinar"
- "Send invitations to all attendees"

### 🔄 **Complex Workflows**
Handle multi-step processes automatically:
- "Create an event, add speakers, and send invitations"
- "Get all events and create a broadcast for the first one"
- "Create event, add attendees, and create a discussion group"
- "Set up a webinar with speakers, create registration, and send reminders"

### 🤖 **Intelligent Features**
- **Context Awareness**: Uses results from previous steps to inform later actions
- **Smart Field Detection**: Automatically identifies missing required information
- **Interactive Refinement**: Asks for clarification when needed
- **Real-time Progress**: Shows step-by-step execution progress
- **Plan Review**: Lets you review and modify execution plans before running them

## How It Works

### For Users
1. **Describe Your Goal**: Tell the agent what you want to accomplish in natural language
2. **Review the Plan**: For complex requests, review the generated execution plan
3. **Provide Missing Info**: If needed, provide any additional required information
4. **Watch It Work**: The agent executes all the necessary actions automatically
5. **Get Results**: Receive confirmation and results of all completed actions

### Technical Architecture
- **Natural Language Processing**: Uses AI to understand user intent
- **Semantic Search**: Vector embeddings of your API documentation for intelligent endpoint matching
- **Context Management**: Maintains context across multiple steps in complex workflows
- **Real-time Communication**: WebSocket-based interface for live updates
- **Error Handling**: Graceful handling of missing information and API errors

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project directory:

```bash
CLAUDE_API_KEY=your_claude_api_key_here
API_KEY=your_api_key_here
API_HOST=https://your-api-domain.com
```

### 3. Prepare API Documentation

Ensure you have a `schema.json` file with your application's API documentation. The agent uses this to understand what actions are available.

### 4. Start the Agent

```bash
python server.py
```

The agent will be available at `http://localhost:8010`

## Usage Examples

### Simple Actions

**Create an Event:**
```
"Create an event for tomorrow at 3 PM called 'Team Meeting' for 1 hour"
```

**Add a Speaker:**
```
"Add John Doe as a speaker to event ID 123"
```

**Get User Information:**
```
"Get magic link for user john.doe@company.com"
```

### Complex Workflows

**Event Setup with Speakers and Invitations:**
```
"Create an event for next Friday at 2 PM called 'Product Launch', add Sarah Johnson and Mike Chen as speakers, and send invitations to the marketing team"
```

**Multi-Step Process:**
```
"Get all upcoming events, create a broadcast for the first one, add it to the homepage, and notify all registered users"
```

## Web Interface

### Single Action Mode
1. Enter your request in natural language
2. Click "Generate Action" or press Ctrl+Enter
3. If additional information is needed, the agent will ask for it
4. View the completed action and results

### Complex Workflow Mode
1. Enter a complex request that requires multiple steps
2. Click "Multiple Actions" button
3. Review the generated execution plan
4. Choose to:
   - **Approve & Execute**: Run all steps automatically
   - **Modify Plan**: Provide feedback to change the plan
5. Watch real-time progress as each step executes
6. View final results with all step outcomes

## API Integration

The agent communicates with your application through WebSocket events:

### Client to Agent Events

- `api_request`: Send a natural language request
- `multiple_requests`: Request complex multi-step workflows
- `approve_execution_plan`: Approve and execute a generated plan
- `modify_execution_plan`: Request modification of an execution plan
- `provide_missing_fields`: Provide additional required information

### Agent to Client Events

- `status`: Real-time status updates during processing
- `missing_fields`: When additional information is needed
- `api_response`: Results of completed actions
- `execution_plan`: Generated execution plan for complex workflows
- `step_completed`: Progress updates for each step
- `multiple_requests_complete`: Final results of all steps

## Configuration

### API Configuration

Set your application's API credentials:

```bash
API_KEY=your_api_key_here
API_HOST=https://your-api-domain.com
```

### AI Model Configuration

You can customize the AI model and parameters in `server.py`:

```python
"model": "claude-3-5-sonnet-20241022",  # Change model as needed
"max_tokens": 1024,                     # Adjust token limit
```

## Development

### Adding New Capabilities

1. **New API Endpoints**: Add to your `schema.json` file
2. **Custom Prompts**: Modify prompts in the service files
3. **UI Enhancements**: Update `templates/index.html`
4. **New Event Handlers**: Add handlers in `server.py`

### Architecture Overview

```
app_interaction_agent/
├── server.py (main WebSocket server)
├── handlers.py (single action handlers)
└── services/
    ├── single_request.py (single action service)
    ├── multiple_requests.py (complex workflow service)
    ├── search.py (API search and matching)
    └── visualization.py (data visualization)
```

## Error Handling

The agent handles various scenarios gracefully:

- **Missing Information**: Prompts for required details
- **API Errors**: Provides clear error messages and suggestions
- **Network Issues**: Graceful degradation with retry logic
- **Invalid Requests**: Helpful feedback for unclear requests

## Troubleshooting

### Common Issues

1. **"CLAUDE_API_KEY not set"**: Ensure your `.env` file contains the API key
2. **"schema.json not found"**: Verify the API documentation file is present
3. **WebSocket connection failed**: Check if the server is running on port 8010
4. **Import errors**: Install all dependencies with `pip install -r requirements.txt`

### Logs

The agent provides detailed logging for debugging:
- Connection events
- Action matching results
- AI responses
- Error details

## License

This project is open source and available under the MIT License. 