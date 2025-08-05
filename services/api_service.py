"""
API Search Service - Handles API search and request processing.
"""

import json
from builders.api_response import APIResponseBuilder
from models.api_response import ApiResponse, ApiResponseStatus
from utils.index import (
    build_index, 
    load_cached_index, 
    save_index_to_cache, 
    embed,
    build_scann_index,
    load_cached_scann_index,
    save_scann_index_to_cache,
    search_apis_scann
)
from utils import (
    get_current_time_for_prompt,
    call_model,
    parse_response,
    sanitize_api_url,
    sanitize_api_params,
    extract_action_data,
    handle_get_request_params,
    make_api_call,
    find_missing_fields_nested,
    update_nested_dict
)
from utils.prompts import BUILD_API_PAYLOAD_PROMPT, BUILD_CONTEXT_RICH_API_PAYLOAD_PROMPT, ENHANCE_REQUEST_WITH_CONTEXT
from models.step_result import StepResult
from typing import Dict

class APIService:
    """Service for handling API search and request processing."""
    
    def __init__(self, search_model="faiss"):
        self.search_model = search_model
        self.api_schema = None
        self.index = None
        self.metadata = None
        self.scann_index = None
        self.scann_metadata = None
        self.initialize_api_system()
    
    def initialize_api_system(self):
        """Initialize the API schema and search index."""
        try:
            with open("schema.json", "r") as f:
                self.api_schema = json.load(f)
        except FileNotFoundError:
            print("Error: `schema.json` not found. Please make sure the file is in the same directory.")
            return False

        if self.search_model == "faiss":
            # Initialize the semantic search index (FAISS)
            self.index, self.metadata, _, _, _ = load_cached_index()
            if self.index is None:
                print("Building new index...")
                self.index, self.metadata, _, _, _ = build_index(self.api_schema)
                save_index_to_cache(self.index, self.metadata, None, None)
                print("Index built and cached successfully!")
            else:
                print("Using cached index...")
        elif self.search_model == "scann":
            # Initialize the semantic search index (SCANN)
            self.scann_index, self.scann_metadata, _, _, _ = load_cached_scann_index()
            if self.scann_index is None:
                print("Building new SCANN index...")
                self.scann_index, self.scann_metadata, _, _, _ = build_scann_index(self.api_schema)
                save_scann_index_to_cache(self.scann_index, self.scann_metadata, None, None)
                print("SCANN index built and cached successfully!")
            else:
                print("Using cached SCANN index...")
        else:
            raise ValueError(f"Unknown search model: {self.search_model}")
        return True
    
    def _generate_api_payload(self, user_input, api_path, api_method, api_parameters, api_request_body):
        """
        Builds a prompt to guide the language model in filling API details.
        """
        current_time = get_current_time_for_prompt()

        return BUILD_API_PAYLOAD_PROMPT.format(
            current_time=current_time,
            api_path=api_path,
            api_method=api_method,
            api_parameters=json.dumps(api_parameters, indent=2),
            api_request_body=json.dumps(api_request_body, indent=2),
            user_input=user_input
        )
    
    def build_prompt_with_context(self, user_input, api_path, api_method, api_parameters, api_request_body, previous_results):
        """Build a prompt with context from previous results."""
        current_time = get_current_time_for_prompt()
        
        context_info = ""
        if previous_results:
            context_info = f"""
**CONTEXT FROM PREVIOUS STEPS:**
{json.dumps(previous_results, indent=2)}

Use the information from previous steps to fill in the current request.
For example, if a previous step created an event with ID "123", use that ID in the current request.
"""

        return BUILD_CONTEXT_RICH_API_PAYLOAD_PROMPT.format(
            current_time=current_time,
            context_info=context_info,
            api_path=api_path,
            api_method=api_method,
            api_parameters=json.dumps(api_parameters, indent=2),
            api_request_body=json.dumps(api_request_body, indent=2),
            user_input=user_input
        )
    
    def search_api(self, user_input):
        """Search for the best matching API based on user input."""
        # Extract action/data/details using Claude
        extracted = extract_action_data(user_input)
        print("Improved user query is", extracted)
        
        if extracted:
            search_text = f"{extracted.get('action', '')} {extracted.get('data', '')}".strip()
            if not search_text:
                search_text = user_input
        else:
            search_text = user_input  # fallback

        # Use the selected search model
        if self.search_model == "faiss":
            query_emb = embed([search_text])
            _, top_ids = self.index.search(query_emb, 1)
            matched_api = self.metadata[top_ids[0][0]]
        elif self.search_model == "scann":
            results = search_apis_scann(search_text, self.scann_index, self.scann_metadata, None, None, top_k=1)
            matched_api = results[0]['api']
        else:
            raise ValueError(f"Unknown search model: {self.search_model}")

        return matched_api
    
    def process_api_request(self, user_input) -> ApiResponse:
        """
        Process a single API request using the same logic as handle_api_request.
        Returns the result without emitting WebSocket events.
        """
        try:
            if not user_input:
                return APIResponseBuilder.build_error_response(
                    error='No query provided',
                    status_code=400,
                    status='failed',
                    action=None,
                    api_path=None,
                    method=None,
                    payload=None
                )
            
            print(f"Processing API request: {user_input}")
            
            # Search for matching API
            matched_api = self.search_api(user_input)
            print("Matched API: ", matched_api["method"], matched_api["path"])
            
            prompt = self._generate_api_payload(
                user_input, 
                api_path=matched_api["path"], 
                api_method=matched_api["method"], 
                api_parameters=matched_api["parameters"], 
                api_request_body=matched_api["requestBody"]
            )
            llm_response = call_model(prompt)
            action, params, api = parse_response(llm_response)

            if matched_api["method"] == "GET":
                api, params, missing_fields = handle_get_request_params(api, params)
                if missing_fields:
                    return APIResponseBuilder.build_missing_fields_response(
                        missing_fields=missing_fields,
                        current_params=params,
                        matched_api=api,
                        request_method=matched_api['method'],
                        action=action,
                        user_input=user_input
                    )

            if action and action != 'more_info_needed':
                # Check if there are any required fields missing (including nested ones)
                missing_fields = find_missing_fields_nested(params)
                if missing_fields:
                    return APIResponseBuilder.build_missing_fields_response(
                        missing_fields=missing_fields,
                        current_params=params,
                        matched_api=api,
                        request_method=matched_api['method'],
                        action=action,
                        user_input=user_input
                    )
                
                # Make the API call
                sanitized_path = sanitize_api_url(api)
                sanitized_params = json.loads(sanitize_api_params(params))
                
                response_data, status_code = make_api_call(
                    matched_api['method'], 
                    sanitized_path, 
                    sanitized_params
                )
                
                if status_code > 299:
                    return APIResponseBuilder.build_error_response(
                        error='API call failed',
                        status_code=status_code,
                        status='failed',
                        action=action,
                        api_path=api,
                        method=matched_api['method'],
                        payload=sanitized_params
                    )
                
                
                return APIResponseBuilder.build_success_response(
                    action=action,
                    api_path=api,
                    method=matched_api['method'],
                    payload=sanitized_params,
                    response=response_data,
                    status_code=status_code,
                    status='success'
                )
                
            elif action == 'more_info_needed':
                return APIResponseBuilder.build_error_response(
                    error=f'More information needed: {params.get("text", "No details provided.")}',
                    status_code=400,
                    status='failed',
                    action=action,
                    api_path=api,
                    method=matched_api['method'],
                    payload=sanitized_params
                )
            else:
                return APIResponseBuilder.build_error_response(
                    error="I couldn't determine the correct action. Please try rephrasing your request.",
                    status_code=400,
                    status='failed',
                    action=action,
                    api_path=api,
                    method=matched_api['method'],
                    payload=sanitized_params
                )
                
        except Exception as e:
            print(f"Error processing API request: {e}")
            import traceback
            print(traceback.format_exc())
            return APIResponseBuilder.build_error_response(
                error=f'An error occurred: {str(e)}',
                status_code=500,
                status='failed',
                action=action,
                api_path=api,
                method=matched_api['method'],
                payload=sanitized_params
            )
    
    def enhance_request_with_context(self, current_step_description: str, previous_results: Dict[str, ApiResponse], original_user_input: str) -> str:
        """Enhance the API request with context from previous results."""

        enhanced_request = f'''
        Use any missing information from the original user input to enhance the current step description.
        Do not change the original user input.
        Do not change the current step description.
        Do not change the known context.

        ORIGINAL USER REQUEST: {original_user_input}
        CURRENT STEP DESCRIPTION: {current_step_description}
        '''
        if not previous_results:
            return enhanced_request
        
        # Extract useful information from previous results
        context_info = []
        for key, result in previous_results.items():
            if isinstance(result, ApiResponse) and result.status == ApiResponseStatus.SUCCESS:
                response_data = result.response
                if isinstance(response_data, dict):
                    # Extract common fields that might be useful
                    if 'id' in response_data:
                        context_info.append(f"{key}_id: {response_data['id']}")
                    if 'event' in response_data:
                        context_info.append(f"event id: {response_data['event']}")
                    if 'start_time' in response_data:
                        context_info.append(f"start time: {response_data['start_time']}")
                    if 'end_time' in response_data:
                        context_info.append(f"end time: {response_data['end_time']}")
                    if 'broadcast' in response_data:
                        context_info.append(f"broadcast id: {response_data['broadcast']}")
                    if 'organization' in response_data:
                        context_info.append(f"organization id: {response_data['organization']}")
                    if 'user' in response_data:
                        context_info.append(f"user id: {response_data['user']}")
        
        
        enhanced_request = f'{enhanced_request}\n\nKNOWN CONTEXT: {chr(10).join(context_info)}'
        return enhanced_request