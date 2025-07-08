import json
import os
from datetime import datetime, timedelta
import pytz
from get_top_apis import build_index, load_cached_index, save_index_to_cache, embed
from utils import (
    get_current_time_for_prompt,
    call_claude,
    parse_response,
    collect_missing_fields,
    extract_action_data,
    handle_get_request_params,
    generate_curl_command
)

# Global variable for the API schema
API_SCHEMA = None



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
8. If logged_in_user is present then use it as owner field in payload if required.
9. If organization is present then use it as organization field in payload if required.

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
            api, params, missing_fields = handle_get_request_params(api, params)
            if missing_fields:
                print("\n❌ Cannot determine correct API call without all required fields.")
                continue

        if action and action != 'more_info_needed':
            # Check if there are any required fields missing
            if params and any(value == "REQUIRED_FIELD_MISSING" for value in params.values()):
                print("\nℹ️ Some required fields are missing. Please provide the following information:")
                updated_params = collect_missing_fields(params)
                if updated_params is None:
                    print("\n❌ Cannot determine correct API call without all required fields.")
                    continue
                params = updated_params
            
            # Generate curl command
            curl_command = generate_curl_command(matched_api, params)
            
            print(f"\n➡️ Action: {action}")
            print(f"📦 API: {api}")
            print(f"📦 Method: {matched_api['method']}")
            print(f"📦 Payload: {json.dumps(params, indent=2)}")
            print(f"📋 cURL Command: {curl_command}")
            # Here you would add the logic to actually execute the API call
            # e.g., execute_api_call(action, params)
        elif action == 'more_info_needed':
            print(f"\nℹ️ More information needed: {params.get('text', 'No details provided.')}")
        else:
            print("\n🤔 I couldn't determine the correct action. Please try rephrasing your request.")

if __name__ == "__main__":
    main()
