# Prompt templates for the API Requester system

# --- SEARCH SERVICE PROMPTS ---

BUILD_API_PAYLOAD_PROMPT = """
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
  "parameters": {api_parameters},
  "requestBody": {api_request_body}
}}
```

**INSTRUCTIONS:**
1. Analyze the user's request and extract all necessary information.
2. Fill in the API path parameters if required (replace {{param}} with actual values).
3. If a required parameter is missing and cannot be inferred, use "REQUIRED_FIELD_MISSING".
4. If request method is GET only add payload as query parameters and take default values as limit=10 and offset=0
5. For dates and times:
   - Convert relative times (e.g., "tomorrow", "next week") to actual dates
   - Use the current timezone (Asia/Kolkata) if not specified
   - Format all dates in ISO 8601 format
6. PLEASE DO NOT RETURN ANYTHING OTHER THAN THE JSON OBJECT. IF YOU WANT TO PASS A NOTE, ADD IT IN JSON OBJECT AS "note": "..." SINCE 
WE ARE USING THIS JSON OBJECT TO PARSE RESPONSE AND DISPLAY IT TO THE USER.
8. If logged_in_user is present then use it as owner field in payload if required.
9. If organization is present then use it as organization field in payload if required.
10. You will also get a currentPage object, if it is present it will contain either (or all) event_id, broadcast_id. Use that info while building
the request payload.

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
User: Get all events
Response: {{
  "action": "Get all events",
  "api": "/event/?limit=10&offset=0",
  "payload": {{}}
}}
---

**CURRENT TASK:**
User: {user_input}
Response:"""

BUILD_CONTEXT_RICH_API_PAYLOAD_PROMPT = """
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
  "parameters": {api_parameters},
  "requestBody": {api_request_body}
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

ENHANCE_REQUEST_WITH_CONTEXT = """
Enhance the following API request with context from previous results.

**ORIGINAL REQUEST:** {api_request}
**USER INPUT:** {user_input}

**PREVIOUS RESULTS CONTEXT:**
{context_info}

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

GENERATE_EXECUTION_PLAN_PROMPT = """
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
      "api_description": "Natural language description of the API call needed. DO NOT GIVE API SIGNATURE, JUST THE NATURAL LANGUAGE DESCRIPTION.",
      "depends_on": [], // List of step numbers this step depends on
      "expected_result": "What we expect to get from this step, again IN NATURAL LANGUAGE.",
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

MODIFY_EXECUTION_PLAN_PROMPT = """
You are modifying an execution plan based on user feedback.

**ORIGINAL USER REQUEST:** {user_input}

**ORIGINAL PLAN:** {original_plan}

**USER FEEDBACK:** {user_feedback}

**TASK:** Modify the plan according to the user's feedback and return the updated plan in the same JSON format.

**RULES:**
1. Keep the same structure as the original plan
2. Modify steps based on user feedback
3. Ensure dependencies are still correct
4. Make sure the plan still accomplishes the original goal

Return only the JSON object, no additional text.
""" 