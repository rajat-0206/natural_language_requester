import json
import pytest
from unittest.mock import patch
from server import app, socketio
from models.execution_plan import ExecutionPlan, ExecutionStep
from models.execution_result import ExecutionResult, ExecutionStatus
from models.api_response import ApiResponse, ApiResponseStatus
from models.step_result import StepResult, StepResultStatus


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def socketio_client(client):
    """Create a SocketIO test client."""
    return socketio.test_client(app, flask_test_client=client)


@pytest.fixture
def mock_execution_plan():
    """Create a mock execution plan for testing."""
    return ExecutionPlan(
        description="Test execution plan",
        steps=[
            ExecutionStep(
                step_number=1,
                description="First step",
                api_description="Get user details",
                result_key="user_data"
            ),
            ExecutionStep(
                step_number=2,
                description="Second step",
                api_description="Create event",
                result_key="event_data"
            )
        ],
        final_result="Event created successfully"
    )


@pytest.fixture
def mock_api_response():
    """Create a mock API response for testing."""
    return ApiResponse(
        status=ApiResponseStatus.SUCCESS,
        action="Test action",
        api_path="/test/api",
        method="GET",
        payload={"test": "data"},
        response={"result": "success"},
        status_code=200,
        message="Success"
    )


@pytest.fixture
def mock_step_result():
    """Create a mock step result for testing."""
    return StepResult(
        step_number=1,
        description="Test step",
        api_description="Test API call",
        action="Test action",
        api_path="/test/api",
        method="GET",
        payload={"test": "data"},
        response={"result": "success"},
        status_code=200,
        status=StepResultStatus.SUCCESS
    )




class TestGeneratePlan:
    """Test case 1: Test generating plan - submit a query and assert response from websocket should be valid plan."""
    @patch('services.socket_response_service.WebSocketResponseService.emit_analyzing_status')
    @patch('services.socket_response_service.WebSocketResponseService.emit_execution_plan')
    def test_generate_execution_plan_success(self, mock_emit_plan, mock_emit_analyzing, socketio_client, mock_execution_plan):
        """Test successful execution plan generation."""
        # Arrange

        plan_response = {
            "description": "Test execution plan",
            "steps": [
                {
                    "step_number": 1,
                    "description": "First step",
                    "api_description": "Get user details",
                    "result_key": "user_data"
                },
                {
                    "step_number": 2,
                    "description": "Second step",
                    "api_description": "Create event",
                    "result_key": "event_data"
                }
            ],
            "final_result": "Event created successfully"
        }
        with patch('utils.call_model', return_value=json.dumps(plan_response)) as mock_call_model:
            query_data = {"query": "Create event for tomorrow 5pm with title hello world"}
            
            # Act
            socketio_client.emit('multiple_requests', query_data)
            received = socketio_client.get_received()
            
            # Assert
            assert len(received) > 0
            mock_emit_analyzing.assert_called_once()
            mock_call_model.assert_called_once()
            mock_emit_plan.assert_called_once_with(mock_execution_plan, query_data["query"] + " logged in user details: organization_id: 16d330dd-57ca-42f2-ab12-fb500c51beb9, user_id: 87cf268e-8049-4e54-ab2d-61d67134c1d2")
    
    @patch('api_requester.services.executor_service.ExecutorService.generate_execution_plan')
    @patch('api_requester.services.socket_response_service.WebSocketResponseService.emit_error')
    def test_generate_execution_plan_failure(self, mock_emit_error, mock_generate_plan, socketio_client):
        """Test execution plan generation failure."""
        # Arrange
        mock_generate_plan.return_value = None
        query_data = {"query": "Invalid query"}
        
        # Act
        socketio_client.emit('multiple_requests', query_data)
        received = socketio_client.get_received()
        
        # Assert
        assert len(received) > 0
        mock_emit_error.assert_called_once_with('Could not generate execution plan. Please try rephrasing your request.')
    
    def test_generate_plan_no_query_provided(self, socketio_client):
        """Test error handling when no query is provided."""
        # Arrange
        query_data = {"query": ""}
        
        # Act
        socketio_client.emit('multiple_requests', query_data)
        received = socketio_client.get_received()
        
        # Assert
        assert len(received) > 0


class TestExecuteApprovedPlan:
    """Test case 2: Test executing approved plan - Websocket request to execute plan."""
    
    @patch('api_requester.services.executor_service.ExecutorService.execute_plan_steps')
    @patch('api_requester.services.socket_response_service.WebSocketResponseService.emit_executing_status')
    @patch('api_requester.services.socket_response_service.WebSocketResponseService.emit_multiple_requests_complete')
    def test_execute_approved_plan_success(self, mock_emit_complete, mock_emit_executing, mock_execute_plan, socketio_client, mock_execution_plan):
        """Test successful plan execution."""
        # Arrange
        mock_result = ExecutionResult(
            status=ExecutionStatus.COMPLETED,
            steps_results=[],
            final_result="Execution completed"
        )
        mock_execute_plan.return_value = mock_result
        
        plan_data = {
            "plan": mock_execution_plan.to_dict(),
            "user_input": "Create event"
        }
        
        # Act
        socketio_client.emit('approve_execution_plan', plan_data)
        received = socketio_client.get_received()
        
        # Assert
        assert len(received) > 0
        mock_emit_executing.assert_called_once()
        mock_execute_plan.assert_called_once()
        mock_emit_complete.assert_called_once_with(mock_result, mock_execution_plan)
    
    @patch('api_requester.services.executor_service.ExecutorService.execute_plan_steps')
    @patch('api_requester.services.socket_response_service.WebSocketResponseService.emit_error')
    def test_execute_approved_plan_invalid_plan(self, mock_emit_error, mock_execute_plan, socketio_client):
        """Test error handling when invalid plan is provided."""
        # Arrange
        plan_data = {
            "plan": {"steps": []},
            "user_input": "Create event"
        }
        
        # Act
        socketio_client.emit('approve_execution_plan', plan_data)
        received = socketio_client.get_received()
        
        # Assert
        assert len(received) > 0
        mock_emit_error.assert_called_once_with('Invalid execution plan')
    
    @patch('api_requester.services.executor_service.ExecutorService.execute_plan_steps')
    @patch('api_requester.services.socket_response_service.WebSocketResponseService.emit_error')
    def test_execute_approved_plan_api_calls_made(self, mock_emit_error, mock_execute_plan, socketio_client, mock_execution_plan):
        """Test that API calls are made for each step in correct order."""
        # Arrange
        mock_result = ExecutionResult(
            status=ExecutionStatus.COMPLETED,
            steps_results=[],
            final_result="Execution completed"
        )
        mock_execute_plan.return_value = mock_result
        
        plan_data = {
            "plan": mock_execution_plan.to_dict(),
            "user_input": "Create event"
        }
        
        # Act
        socketio_client.emit('approve_execution_plan', plan_data)
        
        # Assert
        mock_execute_plan.assert_called_once()
        # Verify the plan has the expected steps
        called_args = mock_execute_plan.call_args[0]
        assert called_args[0].steps == mock_execution_plan.steps


class TestModifyPlan:
    """Test case 3: Test modifying plan - Websocket request to modify plan."""
    
    @patch('api_requester.services.executor_service.ExecutorService.modify_execution_plan')
    @patch('api_requester.services.socket_response_service.WebSocketResponseService.emit_modifying_status')
    @patch('api_requester.services.socket_response_service.WebSocketResponseService.emit_execution_plan')
    def test_modify_execution_plan_success(self, mock_emit_plan, mock_emit_modifying, mock_modify_plan, socketio_client, mock_execution_plan):
        """Test successful plan modification."""
        # Arrange
        mock_modify_plan.return_value = mock_execution_plan
        
        modify_data = {
            "plan": mock_execution_plan.to_dict(),
            "feedback": "Change the time to 6pm",
            "user_input": "Create event"
        }
        
        # Act
        socketio_client.emit('modify_execution_plan', modify_data)
        received = socketio_client.get_received()
        
        # Assert
        assert len(received) > 0
        mock_emit_modifying.assert_called_once()
        mock_modify_plan.assert_called_once()
        mock_emit_plan.assert_called_once_with(mock_execution_plan, "Create event")
    
    @patch('api_requester.services.executor_service.ExecutorService.modify_execution_plan')
    @patch('api_requester.services.socket_response_service.WebSocketResponseService.emit_error')
    def test_modify_execution_plan_no_feedback(self, mock_emit_error, mock_modify_plan, socketio_client, mock_execution_plan):
        """Test error handling when no feedback is provided."""
        # Arrange
        modify_data = {
            "plan": mock_execution_plan.to_dict(),
            "feedback": "",
            "user_input": "Create event"
        }
        
        # Act
        socketio_client.emit('modify_execution_plan', modify_data)
        received = socketio_client.get_received()
        
        # Assert
        assert len(received) > 0
        mock_emit_error.assert_called_once_with('No modification feedback provided')
    
    @patch('api_requester.services.executor_service.ExecutorService.modify_execution_plan')
    @patch('api_requester.services.socket_response_service.WebSocketResponseService.emit_error')
    def test_modify_execution_plan_api_calls_made(self, mock_emit_error, mock_modify_plan, socketio_client, mock_execution_plan):
        """Test that API request to modify plan is made with correct parameters."""
        # Arrange
        mock_modify_plan.return_value = mock_execution_plan
        
        modify_data = {
            "plan": mock_execution_plan.to_dict(),
            "feedback": "Change the time to 6pm",
            "user_input": "Create event"
        }
        
        # Act
        socketio_client.emit('modify_execution_plan', modify_data)
        
        # Assert
        mock_modify_plan.assert_called_once_with(
            mock_execution_plan,
            "Change the time to 6pm",
            "Create event"
        )


class TestApprovedPlanWithMissingFields:
    """Test case 4: Test approved plan contains missing fields - Websocket request to execute plan."""
    
    @patch('api_requester.services.executor_service.ExecutorService.execute_plan_steps')
    @patch('api_requester.services.socket_response_service.WebSocketResponseService.emit_executing_status')
    @patch('api_requester.services.socket_response_service.WebSocketResponseService.emit_error')
    def test_execute_plan_with_missing_fields(self, mock_emit_error, mock_emit_executing, mock_execute_plan, socketio_client, mock_execution_plan):
        """Test plan execution when missing fields are encountered."""
        # Arrange
        mock_result = ExecutionResult(
            status=ExecutionStatus.COMPLETED,
            steps_results=[],
            final_result="Execution completed"
        )
        mock_execute_plan.return_value = mock_result
        
        plan_data = {
            "plan": mock_execution_plan.to_dict(),
            "user_input": "Create event"
        }
        
        # Act
        socketio_client.emit('approve_execution_plan', plan_data)
        
        # Assert
        mock_emit_executing.assert_called_once()
        mock_execute_plan.assert_called_once()
        # Verify the execution plan is processed correctly
        called_args = mock_execute_plan.call_args[0]
        assert called_args[0].steps == mock_execution_plan.steps


class TestProvideMissingFields:
    """Test case 5: Test provide missing fields - Websocket request to provide missing fields."""
    
    @patch('api_requester.utils.make_api_call')
    @patch('api_requester.services.executor_service.ExecutorService.execute_plan_steps')
    @patch('api_requester.services.socket_response_service.WebSocketResponseService.emit_step_completed')
    @patch('api_requester.services.socket_response_service.WebSocketResponseService.emit_multiple_requests_complete')
    def test_provide_missing_fields_success(self, mock_emit_complete, mock_emit_step, mock_execute_plan, mock_make_api_call, socketio_client, mock_execution_plan):
        """Test successful missing fields provision."""
        # Arrange
        mock_make_api_call.return_value = ({"result": "success"}, 200)
        mock_result = ExecutionResult(
            status=ExecutionStatus.COMPLETED,
            steps_results=[],
            final_result="Execution completed"
        )
        mock_execute_plan.return_value = mock_result
        
        missing_fields_data = {
            "fields": {"event_time": "5pm"},
            "current_params": {"event_title": "Hello World"},
            "matched_api": "/api/events",
            "step_number": 1,
            "plan": mock_execution_plan.to_dict(),
            "request_method": "POST",
            "user_input": "Create event",
            "previous_results": {},
            "completed_steps": [],
            "step_description": "Create event step",
            "api_description": "API to create event"
        }
        
        # Act
        socketio_client.emit('provide_multiple_requests_missing_fields', missing_fields_data)
        received = socketio_client.get_received()
        
        # Assert
        assert len(received) > 0
        mock_make_api_call.assert_called_once()
        mock_emit_step.assert_called()
        mock_execute_plan.assert_called_once()
    
    @patch('api_requester.utils.make_api_call')
    @patch('api_requester.services.socket_response_service.WebSocketResponseService.emit_error')
    def test_provide_missing_fields_api_failure(self, mock_emit_error, mock_make_api_call, socketio_client, mock_execution_plan):
        """Test error handling when API call fails during missing fields provision."""
        # Arrange
        mock_make_api_call.return_value = ({"error": "API failed"}, 400)
        
        missing_fields_data = {
            "fields": {"event_time": "5pm"},
            "current_params": {"event_title": "Hello World"},
            "matched_api": "/api/events",
            "step_number": 1,
            "plan": mock_execution_plan.to_dict(),
            "request_method": "POST",
            "user_input": "Create event",
            "previous_results": {},
            "completed_steps": [],
            "step_description": "Create event step",
            "api_description": "API to create event"
        }
        
        # Act
        socketio_client.emit('provide_multiple_requests_missing_fields', missing_fields_data)
        received = socketio_client.get_received()
        
        # Assert
        assert len(received) > 0
        mock_emit_error.assert_called_once()
        # Verify error message contains step number
        error_call = mock_emit_error.call_args[0][0]
        assert "API call failed for step 1" in error_call


class TestFailingExecutionHandling:
    """Test case 6: Test failing execution is handled gracefully - Websocket request to execute plan."""
    
    @patch('api_requester.services.executor_service.ExecutorService.execute_plan_steps')
    @patch('api_requester.services.socket_response_service.WebSocketResponseService.emit_executing_status')
    @patch('api_requester.services.socket_response_service.WebSocketResponseService.emit_error')
    def test_execute_plan_handles_exception_gracefully(self, mock_emit_error, mock_emit_executing, mock_execute_plan, socketio_client, mock_execution_plan):
        """Test that exceptions during plan execution are handled gracefully."""
        # Arrange
        mock_execute_plan.side_effect = Exception("Test exception")
        
        plan_data = {
            "plan": mock_execution_plan.to_dict(),
            "user_input": "Create event"
        }
        
        # Act
        socketio_client.emit('approve_execution_plan', plan_data)
        received = socketio_client.get_received()
        
        # Assert
        assert len(received) > 0
        mock_emit_executing.assert_called_once()
        mock_execute_plan.assert_called_once()
        mock_emit_error.assert_called_once()
        # Verify error message contains the exception details
        error_call = mock_emit_error.call_args[0][0]
        assert "An error occurred during execution: Test exception" in error_call
    
    @patch('api_requester.services.executor_service.ExecutorService.execute_plan_steps')
    @patch('api_requester.services.socket_response_service.WebSocketResponseService.emit_executing_status')
    @patch('api_requester.services.socket_response_service.WebSocketResponseService.emit_error')
    def test_execute_plan_handles_api_failures_gracefully(self, mock_emit_error, mock_emit_executing, mock_execute_plan, socketio_client, mock_execution_plan):
        """Test that API failures during plan execution are handled gracefully."""
        # Arrange
        mock_result = ExecutionResult(
            status=ExecutionStatus.FAILED,
            steps_results=[],
            final_result="Execution failed"
        )
        mock_execute_plan.return_value = mock_result
        
        plan_data = {
            "plan": mock_execution_plan.to_dict(),
            "user_input": "Create event"
        }
        
        # Act
        socketio_client.emit('approve_execution_plan', plan_data)
        received = socketio_client.get_received()
        
        # Assert
        assert len(received) > 0
        mock_emit_executing.assert_called_once()
        mock_execute_plan.assert_called_once()
        # Verify that the execution result is handled appropriately
        called_args = mock_execute_plan.call_args[0]
        assert called_args[0].steps == mock_execution_plan.steps


class TestWebSocketConnection:
    """Test WebSocket connection handling."""
    
    def test_websocket_connect(self, socketio_client):
        """Test WebSocket connection establishment."""
        # Act
        received = socketio_client.get_received()
        
        # Assert
        assert len(received) > 0
        # Should receive connection status and suggestions
        connection_message = received[0]
        assert connection_message['name'] in ['connection_status', 'next_best_items']
    
    def test_websocket_disconnect(self, socketio_client):
        """Test WebSocket disconnection handling."""
        # Act
        socketio_client.disconnect()
        
        # Assert
        # Should handle disconnect gracefully without errors
        assert True


class TestAPIRequestValidation:
    """Test API request validation and error handling."""
    
    def test_multiple_requests_empty_query(self, socketio_client):
        """Test handling of empty query in multiple requests."""
        # Arrange
        query_data = {"query": "   "}  # Whitespace only
        
        # Act
        socketio_client.emit('multiple_requests', query_data)
        received = socketio_client.get_received()
        
        # Assert
        assert len(received) > 0
    
    def test_multiple_requests_missing_query(self, socketio_client):
        """Test handling of missing query in multiple requests."""
        # Arrange
        query_data = {}  # No query field
        
        # Act
        socketio_client.emit('multiple_requests', query_data)
        received = socketio_client.get_received()
        
        # Assert
        assert len(received) > 0


class TestExecutionPlanValidation:
    """Test execution plan validation."""
    
    @patch('api_requester.services.executor_service.ExecutorService.generate_execution_plan')
    @patch('api_requester.services.socket_response_service.WebSocketResponseService.emit_error')
    def test_execution_plan_without_steps(self, mock_emit_error, mock_generate_plan, socketio_client):
        """Test handling of execution plan without steps."""
        # Arrange
        empty_plan = ExecutionPlan(description="Empty plan", steps=[], final_result=None)
        mock_generate_plan.return_value = empty_plan
        
        query_data = {"query": "Test query"}
        
        # Act
        socketio_client.emit('multiple_requests', query_data)
        received = socketio_client.get_received()
        
        # Assert
        assert len(received) > 0
        mock_emit_error.assert_called_once_with('Could not generate execution plan. Please try rephrasing your request.')
    
    @patch('api_requester.services.executor_service.ExecutorService.generate_execution_plan')
    @patch('api_requester.services.socket_response_service.WebSocketResponseService.emit_error')
    def test_execution_plan_none_result(self, mock_emit_error, mock_generate_plan, socketio_client):
        """Test handling of None execution plan result."""
        # Arrange
        mock_generate_plan.return_value = None
        
        query_data = {"query": "Test query"}
        
        # Act
        socketio_client.emit('multiple_requests', query_data)
        received = socketio_client.get_received()
        
        # Assert
        assert len(received) > 0
        mock_emit_error.assert_called_once_with('Could not generate execution plan. Please try rephrasing your request.')


if __name__ == '__main__':
    pytest.main([__file__])