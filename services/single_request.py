import json
import time
from services.search import APISearchService
from utils import (
    sanitize_api_url,
    sanitize_api_params,
    make_api_call,
    suggest_next_best_item,
    update_nested_dict,
    extract_action_data,
)


class SingleRequestService:
    """
    Service class to handle single API requests.
    This is the previous version of the agent that can only handle single request per query.
    """
    
    def __init__(self, search_service: APISearchService):
        self.search_service = search_service
    
    def process_single_request(self, user_input: str):
        """
        Process a single API request from natural language input.
        
        Process to find the best matching API:
        1. Separate out actions and data from the given input using AI.
        2. Use the action to find the best matching API in swagger vector embedding. 
        3. Give the best matching API and data from step 1 to AI again to fill the data
        and return the CURL. If some required field is not present the AI tells that.
        4. If all required field are present, return the CURL. If not prompt user to 
        enter those fields.
        
        Args:
            user_input (str): Natural language description of the API request
            
        Returns:
            dict: Result containing either success response, missing fields, or error
        """
        try:
            if not user_input.strip():
                return {'error': 'No query provided'}
            
            print(f"Processing single request: {user_input}")
            
            # Extract action/data/details using AI
            extracted = extract_action_data(user_input)
            print("Improved user query is", extracted)
            
            if extracted:
                search_text = f"{extracted.get('action', '')} {extracted.get('data', '')}".strip()
                if not search_text:
                    search_text = user_input
            else:
                search_text = user_input  # fallback

            # Search for matching API
            matched_api = self.search_service.search_api(user_input)
            print("Matched API: ", matched_api["method"], matched_api["path"])
            
            # Use the search service to process the request
            result = self.search_service.process_api_request(user_input)
            
            if result.get('missing_fields'):
                return {
                    'status': 'missing_fields',
                    'missing_fields': result['missing_fields'],
                    'current_params': result['current_params'],
                    'matched_api': result['matched_api'],
                    'request_method': result['request_method'],
                    'action': result['action'],
                    'user_input': result['user_input']
                }
            
            if result.get('error'):
                return {'error': result['error']}
            
            if result.get('status') == 'success':
                return {
                    'status': 'success',
                    'action': result['action'],
                    'api_path': result['api_path'],
                    'method': result['method'],
                    'payload': result['payload'],
                    'user_input': user_input
                }
            else:
                return {'error': 'Failed to process request'}
                
        except Exception as e:
            print(f"Error processing single request: {e}")
            return {'error': f'An error occurred: {str(e)}'}
    
    def handle_missing_fields(self, provided_fields: dict, current_params: dict, 
                            matched_api: str, request_method: str, action: str, user_input: str):
        """
        Handle missing fields provided by user for a single API request.
        
        Args:
            provided_fields (dict): Fields provided by the user
            current_params (dict): Current parameters for the API call
            matched_api (str): The matched API endpoint
            request_method (str): HTTP method (GET, POST, etc.)
            action (str): Description of the action being performed
            user_input (str): Original user input
            
        Returns:
            dict: Result containing either success response or error
        """
        try:
            print("request method", request_method)
            print("matched api", matched_api)
            print("current params", current_params)
            print("provided fields", provided_fields)
            print("action", action)
            print("user input", user_input)
            
            # Update params with provided fields (handling nested structures)
            updated_params = update_nested_dict(current_params, provided_fields)

            if request_method == "GET":
                api = matched_api + "?" + "&".join([f"{key}={value}" for key, value in updated_params.items()])
            else:
                api = matched_api
            
            # Make the API call
            sanitized_path = sanitize_api_url(api)
            sanitized_params = json.loads(sanitize_api_params(updated_params))
            
            response_data, status_code = make_api_call(
                request_method, 
                sanitized_path, 
                sanitized_params
            )
            
            if status_code > 299:
                return {'error': f'API failed: {status_code}: {response_data}'}
            
            # Return success response
            return {
                'status': 'success',
                'action': action,
                'api_path': sanitized_path,
                'method': request_method,
                'payload': sanitized_params,
                'user_input': user_input
            }
            
        except Exception as e:
            print(f"Error handling missing fields: {e}")
            return {'error': f'An error occurred: {str(e)}'}
    
    def get_final_response(self, action: str, api_path: str, method: str, payload: dict):
        """
        Format the final response for a successful API call.
        
        Args:
            action (str): Description of the action performed
            api_path (str): The API endpoint that was called
            method (str): HTTP method used
            payload (dict): The payload sent to the API
            
        Returns:
            dict: Formatted response
        """
        return {
            'action': action,
            'api_path': api_path,
            'method': method,
            'payload': payload,
            'status': 'success'
        }
    
    def get_next_suggestions(self, action: str, user_input: str):
        """
        Get suggestions for next best actions based on current action.
        
        Args:
            action (str): Current action performed
            user_input (str): Original user input
            
        Returns:
            dict: Suggestions for next actions
        """
        return suggest_next_best_item(action, user_input) 