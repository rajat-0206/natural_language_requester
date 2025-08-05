import time
from flask_socketio import emit
from services.single_request import SingleRequestService


class SingleRequestHandlers:
    """
    Handlers for single API request operations.
    These are the previous version handlers that can only handle single request per query.
    """
    
    def __init__(self, single_request_service: SingleRequestService):
        self.single_request_service = single_request_service
    
    def handle_api_request(self, data):
        """
        SocketIO handler for single API requests.
        Finds the best API for the given natural language input and returns it.
        Can only handle single request per query.
        """
        try:
            user_input = data.get('query', '').strip()
            
            if not user_input:
                emit('error', {'message': 'No query provided'})
                return
            
            print(f"Received request: {user_input}")
            emit('status', {'message': 'Processing your request...', 'status': 'processing'})
            
            # Process the single request
            result = self.single_request_service.process_single_request(user_input)
            
            if result.get('error'):
                emit('error', {'message': result['error']})
                return
            
            if result.get('status') == 'missing_fields':
                emit('missing_fields', {
                    'message': 'Some required fields are missing',
                    'missing_fields': result['missing_fields'],
                    'current_params': result['current_params'],
                    'matched_api': result['matched_api'],
                    'request_method': result['request_method'],
                    'action': result['action'],
                    'user_input': result['user_input']
                })
                return
            
            if result.get('status') == 'success':
                final_response = self.single_request_service.get_final_response(
                    result['action'], 
                    result['api_path'], 
                    result['method'], 
                    result['payload']
                )
                emit('api_response', final_response)

                time.sleep(1)
                # suggest what can be done next based on current action
                next_best_items = self.single_request_service.get_next_suggestions(result['action'], user_input)
                if next_best_items:
                    emit('next_best_items', next_best_items)
            else:
                emit('error', {'message': 'Failed to process request'})
                
        except Exception as e:
            print(f"Error processing request: {e}")
            import traceback
            print(traceback.format_exc())
            emit('error', {'message': f'An error occurred: {str(e)}'})
    
    def handle_missing_fields(self, data):
        """
        SocketIO handler for missing fields provided by user.
        """
        try:
            provided_fields = data.get('fields', {})
            current_params = data.get('current_params', {})
            matched_api = data.get('matched_api', {})
            request_method = data.get('request_method', "GET")
            action = data.get('action', "")
            user_input = data.get('user_input', "")

            # Handle missing fields using the service
            result = self.single_request_service.handle_missing_fields(
                provided_fields, current_params, matched_api, 
                request_method, action, user_input
            )
            
            if result.get('error'):
                emit('error', {'message': result['error']})
                return
            
            if result.get('status') == 'success':
                # Send final response
                final_response = self.single_request_service.get_final_response(
                    result['action'], 
                    result['api_path'], 
                    result['method'], 
                    result['payload']
                )
                emit('api_response', final_response)
                
                time.sleep(1)
                next_best_items = self.single_request_service.get_next_suggestions(result['action'], user_input)
                if next_best_items:
                    emit('next_best_items', next_best_items)
            else:
                emit('error', {'message': 'Failed to process missing fields'})
        
        except Exception as e:
            print(f"Error handling missing fields: {e}")
            import traceback
            print(traceback.format_exc())
            emit('error', {'message': f'An error occurred: {str(e)}'}) 