import json
import requests
import os
import urllib.parse
from datetime import datetime
import pytz
from dotenv import load_dotenv

load_dotenv()

# Global variables
API_SCHEMA = None
print_raw_response = True
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')
MODEL_TO_USE = "CLAUDE"
CLAUDE_API_HOST = 'https://api.anthropic.com/v1/messages'

NEXT_BEST_SUGGESTIONS = {
  "Event": {
    "next": [
      "Broadcast",
      "AgendaItem",
      "DiscussionGroup",
      "Track",
      "Booth",
      "Resource",
      "MedialiveRtmpInput",
      "EventEmailTemplate",
      "EmailConfig",
      "EventWidget",
      "TicketType",
      "EventAppModerators",
      "EventDefaultRole",
      "EventUsers",
      "EventRole"
    ]
  },
  "Broadcast": {
    "next": [
      "Resource",
      "TrackItems",
      "SlideAsset",
      "VideoAsset",
      "Staff"
    ]
  },
  "Track": {
    "next": [
      "TrackItems"
    ]
  },
  "DiscussionGroup": {
    "next": [
      "Booth"
    ]
  },
  "Booth": {
    "next": [
      "Resource"
    ]
  },
  "AgendaItem": {
    "next": []
  },
  "TrackItems": {
    "next": []
  },
  "Resource": {
    "next": []
  },
  "EmailConfig": {
    "next": []
  },
  "EventEmailTemplate": {
    "next": [
      "EmailConfig"
    ]
  },
  "EventUsers": {
    "next": [
      "EventRole"
    ]
  },
  "EventRole": {
    "next": []
  },
  "EventWidget": {
    "next": []
  },
  "MedialiveRtmpInput": {
    "next": [
      "Broadcast"
    ]
  },
  "TicketType": {
    "next": []
  },
  "EventAppModerators": {
    "next": []
  },
  "EventDefaultRole": {
    "next": []
  }
}


def get_current_time_for_prompt():
    """
    Returns the current date, time, and timezone to provide context to the model.
    This is crucial for handling relative date/time requests like "tomorrow".
    """
    tz = pytz.timezone('Asia/Kolkata') 
    now = datetime.now(tz)
    return now.strftime('%Y-%m-%d %H:%M:%S %Z%z')

def call_model(prompt):
    print("MODEL_TO_USE", MODEL_TO_USE)
    if MODEL_TO_USE == "CLAUDE":
        return call_claude(prompt)
    elif MODEL_TO_USE == "OPENAI":
        return call_openai(prompt)
    else:
        return call_ollama(prompt)
    

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
                "model": "claude-opus-4-20250514",
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
    print("Calling Ollama")
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
    response = call_model(prompt)
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

def sanitize_api_url(path):
    '''
    Sanitize the API path and parameters to convert any random value to correct type like tomorrow becomes actual data
    '''
    # prompt = f"""
    # Sanitize the API path {path} to convert any random value to correct type like tomorrow becomes actual data. Only return the sanitized text.
    # If path is already sanitized, return the same path.
    # """
    # response = call_model(prompt)
    return path

def sanitize_api_params(params):
    '''
    Sanitize the API parameters to convert any random value to correct type like tomorrow becomes actual data
    '''
    print("args", params)
    prompt = f"""
    Sanitize the API parameters {params} to convert any random value to correct type like tomorrow becomes actual data. Return the sanitized parameters in JSON format.
    DO NOT ADD ANYTHING ELSE TO THE RESPONSE.
    If parameters are already sanitized, return the same parameters.
    DO NOT ADD ````json to the response. It should be a valid JSON string.
    """
    response = call_model(prompt)
    print("response", response)
    return response

def generate_curl_command(api, params, path):
    """Generate a curl command from the API and parameters."""
    # use llm to again sanitize the api and params to convert any random value to correct type like tomorrow becomes actual data
    method = api['method'].upper()
    
    # Replace path parameters
    for param_name, param_value in params.items():
        if f"{{{param_name}}}" in path:
            path = path.replace(f"{{{param_name}}}", str(param_value))
    
    curl_parts = [f"curl -X {method}"]
    
    # Add headers
    curl_parts.append('-H "Content-Type: application/json"')
    curl_parts.append('-H "Authorization: Bearer YOUR_API_KEY"')
    
    # Add URL
    base_url = "https://api.example.com"  # Replace with your actual base URL
    url = f"{base_url}{path}"
    curl_parts.append(f'"{url}"')
    
    # Add body for non-GET requests
    if method != "GET" and params:
        # Remove path parameters from body
        body_params = {k: v for k, v in params.items() if f"{{{k}}}" not in path}
        if body_params:
            curl_parts.append(f"-d '{json.dumps(body_params)}'")
    
    return " ".join(curl_parts)

def handle_get_request_params(api, params):
    """
    Handle GET request parameters by parsing URL and extracting query parameters.
    Returns updated params and API path.
    """
    parsed_url = urllib.parse.urlparse(api)
    query_params = urllib.parse.parse_qs(parsed_url.query)
    missing_fields = []
    
    for key, value in query_params.items():
        if value[0] == "REQUIRED_FIELD_MISSING":
            missing_fields.append(key)
            query_params[key] = value[0]
        else:
            query_params[key] = value[0]
    
    if not missing_fields:
        return api, params, None
    
    return parsed_url.path, query_params, missing_fields

def make_api_call(method, api_path, payload=None, headers=None):
    """
    Make an API call to the API server using requests module.
    Returns the response JSON and status.
    """
    try:
        if headers is None:
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer YOUR_API_KEY"
            }
        
        response = requests.request(method, api_path, json=payload, headers=headers)
        response.raise_for_status()
        return response.json(), response.status_code
    except requests.exceptions.RequestException as e:
        print(f"Error making API call: {e}")
        return None, None
    except json.JSONDecodeError as e:
        print(f"Error parsing API response: {e}")
        return None, None 
    
def suggest_next_best_item(current_item, user_query):
    '''
    Suggest the next best item to create based on the current item and user query
    '''
    try:
      print("action", current_item)
      print("user_query", user_query)
      prompt = f"""
      Suggest the next best item to create based on the current item {current_item} and user query {user_query}.
      We need 4 suggestions which user can do once current item is created.
      For eg:
      If initial user query was: Create event at 5pm,
      Then next best items to create are:
      - Create a new broadcast
      - Add speakers to the event
      - Add attendees to the event
      - Add a new staff


      1. Return json array with 4 suggestions. DO NOT ADD ANY TEXT. It should be a valid JSON string.
      {{
          "suggestions": [
              "suggestion1",
              "suggestion2",
              "suggestion3",
              "suggestion4"
          ]
      }}
      """
      response = call_model(prompt)
      # parse the response to json, remove any ```json to the response. It should be a valid JSON string.
      response = response.replace("```json", "").replace("```", "")
      return json.loads(response)
    except Exception as e:
        print("Error in suggest_next_best_item", e)
        return None

def call_openai(prompt):
    '''
    Call the OpenAI API with the given prompt
    '''
    try:
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
            },
            json={
                "model": "gpt-4.1",
                "input": prompt
            }
        )
        response.raise_for_status()
        output = response.json()["output"]
        print("output", output)
        content = output[0]["content"][0]["text"]
        print("content", content)
        return content
    except requests.exceptions.RequestException as e:
        print("Error in call_openai", e)
        return None