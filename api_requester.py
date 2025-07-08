import json
import requests
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pytz
import urllib.parse
from get_top_apis import build_index, load_cached_index, save_index_to_cache, embed

load_dotenv()

# Global variable for the API schema
API_SCHEMA = None
print_raw_response = True
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')
CLAUDE_API_HOST = 'https://api.anthropic.com/v1/messages'

def get_current_time_for_prompt():
    """
    Returns the current date, time, and timezone to provide context to the model.
    This is crucial for handling relative date/time requests like "tomorrow".
    """
    tz = pytz.timezone('Asia/Kolkata') 
    now = datetime.now(tz)
    return now.strftime('%Y-%m-%d %H:%M:%S %Z%z')

def build_prompt(user_input, api_path, api_method, api_parameters, api_request_body):
    """
    Builds a prompt to guide the language model in filling API details.
    """
    current_time = get_current_time_for_prompt()

    prompt = f"""
You are an expert AI assistant that fills in API request details based on user input.
Your task is to analyze the user's request and the matched API endpoint to generate a valid JSON payload.

**CONTEXT:**
- The current date and time is: {current_time}. You MUST use this to resolve relative times like "tomorrow" or "next week".
- All dates and times in the final JSON MUST be in ISO 8601 format (e.g., "YYYY-MM-DDTHH:MM:SS").

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
2. Fill in the API path parameters if required (replace {{param}} with actual values).
3. Generate a valid request payload based on the API's requestBody schema.
4. If a required parameter is missing and cannot be inferred, use "REQUIRED_FIELD_MISSING".
5. If request method is GET only add limit and offset parameters and take default values as limit=10 and offset=0
5. For dates and times:
   - Convert relative times (e.g., "tomorrow", "next week") to actual dates
   - Use the current timezone (Asia/Kolkata) if not specified
   - Format all dates in ISO 8601 format
6. Respond ONLY with a single, valid JSON object containing:
   - action: The action being performed
   - api: The complete API path with parameters filled
   - payload: The request payload
7. PLEASE DO NOT RETURN ANYTHING OTHER THAN THE JSON OBJECT. IF YOU WANT TO PASS A NOTE, ADD IT IN JSON OBJECT AS "note": "..." SINCE 
WE ARE USING THIS JSON OBJECT TO PARSE RESPONSE AND DISPLAY IT TO THE USER.

**EXAMPLES:**
---
User: Schedule a meeting called Team Sync on June 28th at 3 PM for 45 minutes
Response: {{
  "action": "Create event",
  "api": "/event/",
  "payload": {{
    "title": "Team Sync",
    "start_time": "2025-06-28T15:00:00",
    "end_time": "2025-06-28T15:45:00",
    "timezone": "Asia/Kolkata",
    "organization": "REQUIRED_FIELD_MISSING",
    "owner": "REQUIRED_FIELD_MISSING"
  }}
}}
---
User: Create an event of 1 hour for tomorrow 5pm IST with title "AI in todays world"
Response: {{
  "action": "Create event",
  "api": "/event/",
  "payload": {{
    "title": "AI in todays world",
    "start_time": "{ (datetime.now(pytz.timezone('Asia/Kolkata')) + timedelta(days=1)).strftime('%Y-%m-%d') }T17:00:00",
    "end_time": "{ (datetime.now(pytz.timezone('Asia/Kolkata')) + timedelta(days=1)).strftime('%Y-%m-%d') }T18:00:00",
    "timezone": "Asia/Kolkata",
    "organization": "REQUIRED_FIELD_MISSING",
    "owner": "REQUIRED_FIELD_MISSING"
  }}
}}
---
User: Get all events
Response: {{
  "action": "Get all events",
  "api": "/event/",
  "payload": {{
    "limit": 10,
    "offset": 0
  }}
}}
---

**CURRENT TASK:**
User: {user_input}
Response:"""

    return prompt.strip()

def call_claude(prompt):
    """Calls the Claude API with the given prompt."""
    try:
        # Get API key from environment variable
        api_key = os.getenv('CLAUDE_API_KEY')
        if not api_key or len(api_key) < 10:
            print("Error: CLAUDE_API_KEY environment variable not set.")
            print("Please set your Claude API key: export CLAUDE_API_KEY='your-api-key-here'")
            return None
        
        response = requests.post(
            CLAUDE_API_HOST,
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 1024,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
        )
        response.raise_for_status()
        return response.json()["content"][0]["text"]
    except requests.exceptions.RequestException as e:
        print(f"Error calling Claude API: {e}")
        print("Please ensure you have a valid API key and internet connection.")
        return None
    except KeyError as e:
        print(f"Error parsing Claude API response: {e}")
        print("Unexpected response format from Claude API.")
        return None

def call_ollama(prompt):
    """Calls the Ollama API with the given prompt."""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "top_p": 0.9
                }
            }
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama: {e}")
        print("Please ensure the Ollama server is running and accessible.")
        return None

def parse_response(llm_response):
    """Parses the JSON string from the LLM's response."""
    if not llm_response:
        return None, None, None
    try:
        if print_raw_response:
            print("Raw response was:\n", llm_response)
        if "```json" in llm_response:
            json_str = llm_response.split("```json")[-1].split("```")[0].strip()
        else:
            json_str = llm_response.strip()
            
        data = json.loads(json_str)
        action = data.get("action")
        params = data.get("payload")
        api = data.get("api")
        return action, params, api
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse LLM response as JSON: {e}")
        print("Raw response was:\n", llm_response)
        return None, None, None
    except Exception as e:
        print(f"❌ An unexpected error occurred during parsing: {e}")
        print("Raw response was:\n", llm_response)
        return None, None, None

def collect_missing_fields(params):
    """
    Collects missing required fields from user input.
    Returns a new dictionary with all required fields filled.
    """
    missing_fields = {}
    for key, value in params.items():
        if value == "REQUIRED_FIELD_MISSING":
            user_input = input(f"Please enter the {key}: ").strip()
            if not user_input:  # If user provides empty input
                return None
            missing_fields[key] = user_input
    
    # Create new params with missing fields filled
    updated_params = params.copy()
    updated_params.update(missing_fields)
    return updated_params

def extract_action_data(user_query):
    prompt = f"""
You are an expert at analyzing user requests for APIs. Given the following user query, extract:
- action: the main intent or verb (e.g., get, create, delete)
- data: the main object or resource (e.g., event, attendees, user)
- details: any other relevant information (e.g., filters, ids, time)

User query: \"{user_query}\"

Respond in JSON:
{{
  "action": "...",
  "data": "...",
  "details": "..."
}}
"""
    response = call_claude(prompt)
    try:
        # Parse the JSON from the response
        if "```json" in response:
            json_str = response.split("```json")[-1].split("```")[0].strip()
        else:
            json_str = response.strip()
        return json.loads(json_str)
    except Exception as e:
        print("Failed to parse Claude extraction:", e)
        return None

def main():
    """Main loop to run the API agent."""
    global API_SCHEMA
    try:
        with open("schema.json", "r") as f:
            API_SCHEMA = json.load(f)
    except FileNotFoundError:
        print("Error: `schema.json` not found. Please make sure the file is in the same directory.")
        return

    # Initialize the semantic search index
    index, metadata, _, _, _ = load_cached_index()
    if index is None:
        print("Building new index...")
        index, metadata, _, _, _ = build_index(API_SCHEMA)
        save_index_to_cache(index, metadata, None, None)
        print("Index built and cached successfully!")
    else:
        print("Using cached index...")
        
    print("✅ AI Agent is ready. Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        
        if not user_input:
            continue

        # Extract action/data/details using Mistral
        extracted = extract_action_data(user_input)
        print("Improved user query is", extracted)
        if extracted:
            search_text = f"{extracted.get('action', '')} {extracted.get('data', '')}".strip()
            if not search_text:
                search_text = user_input
        else:
            search_text = user_input  # fallback

        # First, find the best matching API using semantic search (use extracted action+data)
        query_emb = embed([search_text])
        _, top_ids = index.search(query_emb, 1)  # Get top 1 match
        matched_api = metadata[top_ids[0][0]]

        print("Matched API: ", matched_api)
        print("(Search text used for embedding:)", search_text)
        
        # Then use Claude to fill in the details (use full user_input)
        prompt = build_prompt(user_input, api_path=matched_api["path"], api_method=matched_api["method"], api_parameters=matched_api["parameters"], api_request_body=matched_api["requestBody"])
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
                print("Some required fields are missing. Please provide the following information:")
                updated_params = collect_missing_fields(query_params)
                if updated_params is None:
                    print("\n❌ Cannot determine correct API call without all required fields.")
                    continue
                query_params = updated_params
                api = parsed_url.path + "?" + "&".join([f"{key}={value}" for key, value in query_params.items()])

        if action and action != 'more_info_needed':
            # Check if there are any required fields missing
            if params and any(value == "REQUIRED_FIELD_MISSING" for value in params.values()):
                print("\nℹ️ Some required fields are missing. Please provide the following information:")
                updated_params = collect_missing_fields(params)
                if updated_params is None:
                    print("\n❌ Cannot determine correct API call without all required fields.")
                    continue
                params = updated_params
            
            print(f"\n➡️ Action: {action}")
            print(f"📦 API: {api}")
            print(f"📦 Method: {matched_api['method']}")
            print(f"📦 Payload: {json.dumps(params, indent=2)}")
            # Here you would add the logic to actually execute the API call
            # e.g., execute_api_call(action, params)
        elif action == 'more_info_needed':
            print(f"\nℹ️ More information needed: {params.get('text', 'No details provided.')}")
        else:
            print("\n🤔 I couldn't determine the correct action. Please try rephrasing your request.")

if __name__ == "__main__":
    main()
