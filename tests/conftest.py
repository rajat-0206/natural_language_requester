"""
Pytest configuration and shared fixtures for the API Requester test suite.
This file provides common test utilities and fixtures used across multiple test files.
"""

import pytest
import json
from unittest.mock import Mock, MagicMock
from flask import Flask
from flask_socketio import SocketIO

# Import the app and socketio from the main server
from server import app, socketio
from models.execution_plan import ExecutionPlan, ExecutionStep
from models.execution_result import ExecutionResult, ExecutionStatus
from models.api_response import ApiResponse, ApiResponseStatus
from models.step_result import StepResult, StepResultStatus


@pytest.fixture(scope="session")
def flask_app():
    """Create a Flask app instance for testing."""
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    return app


@pytest.fixture
def client(flask_app):
    """Create a test client for the Flask app."""
    with flask_app.test_client() as client:
        yield client


@pytest.fixture
def socketio_client(client):
    """Create a SocketIO test client."""
    return socketio.test_client(flask_app, flask_test_client=client)


@pytest.fixture
def mock_execution_plan():
    """Create a mock execution plan for testing."""
    return ExecutionPlan(
        description="Test execution plan for creating an event",
        steps=[
            ExecutionStep(
                step_number=1,
                description="Get user details",
                api_description="Retrieve current user information",
                result_key="user_data"
            ),
            ExecutionStep(
                step_number=2,
                description="Create event",
                api_description="Create a new event with provided details",
                result_key="event_data"
            ),
            ExecutionStep(
                step_number=3,
                description="Send confirmation",
                api_description="Send confirmation email to user",
                result_key="confirmation_data"
            )
        ],
        final_result="Event created successfully and confirmation sent"
    )


@pytest.fixture
def mock_simple_execution_plan():
    """Create a simple mock execution plan for testing."""
    return ExecutionPlan(
        description="Simple test execution plan",
        steps=[
            ExecutionStep(
                step_number=1,
                description="Single step",
                api_description="Execute single API call",
                result_key="result"
            )
        ],
        final_result="Single step completed"
    )


@pytest.fixture
def mock_empty_execution_plan():
    """Create an empty execution plan for testing edge cases."""
    return ExecutionPlan(
        description="Empty execution plan",
        steps=[],
        final_result=None
    )


@pytest.fixture
def mock_api_response():
    """Create a mock API response for testing."""
    return ApiResponse(
        status=ApiResponseStatus.SUCCESS,
        action="Test API call",
        api_path="/api/test",
        method="GET",
        payload={"test": "data", "param": "value"},
        response={"result": "success", "data": {"id": 123}},
        status_code=200,
        message="API call successful"
    )


@pytest.fixture
def mock_failed_api_response():
    """Create a mock failed API response for testing."""
    return ApiResponse(
        status=ApiResponseStatus.FAILED,
        action="Test API call",
        api_path="/api/test",
        method="POST",
        payload={"test": "data"},
        response={"error": "Validation failed"},
        status_code=400,
        message="Bad request"
    )


@pytest.fixture
def mock_step_result():
    """Create a mock step result for testing."""
    return StepResult(
        step_number=1,
        description="Test step execution",
        api_description="Test API call description",
        action="Test action",
        api_path="/api/test",
        method="GET",
        payload={"param": "value"},
        response={"result": "success"},
        status_code=200,
        status=StepResultStatus.SUCCESS
    )


@pytest.fixture
def mock_failed_step_result():
    """Create a mock failed step result for testing."""
    return StepResult(
        step_number=1,
        description="Test step execution",
        api_description="Test API call description",
        action="Test action",
        api_path="/api/test",
        method="POST",
        payload={"param": "value"},
        response={"error": "Failed"},
        status_code=500,
        status=StepResultStatus.FAILED
    )


@pytest.fixture
def mock_execution_result():
    """Create a mock execution result for testing."""
    return ExecutionResult(
        status=ExecutionStatus.COMPLETED,
        steps_results=[
            StepResult(
                step_number=1,
                description="Step 1",
                api_description="API 1",
                action="Action 1",
                api_path="/api/1",
                method="GET",
                payload={},
                response={"result": "success"},
                status_code=200,
                status=StepResultStatus.SUCCESS
            ),
            StepResult(
                step_number=2,
                description="Step 2",
                api_description="API 2",
                action="Action 2",
                api_path="/api/2",
                method="POST",
                payload={"data": "value"},
                response={"result": "success"},
                status_code=201,
                status=StepResultStatus.SUCCESS
            )
        ],
        final_result="All steps completed successfully"
    )


@pytest.fixture
def mock_failed_execution_result():
    """Create a mock failed execution result for testing."""
    return ExecutionResult(
        status=ExecutionStatus.FAILED,
        steps_results=[
            StepResult(
                step_number=1,
                description="Step 1",
                api_description="API 1",
                action="Action 1",
                api_path="/api/1",
                method="GET",
                payload={},
                response={"result": "success"},
                status_code=200,
                status=StepResultStatus.SUCCESS
            ),
            StepResult(
                step_number=2,
                description="Step 2",
                api_description="API 2",
                action="Action 2",
                api_path="/api/2",
                method="POST",
                payload={"data": "value"},
                response={"error": "Failed"},
                status_code=500,
                status=StepResultStatus.FAILED
            )
        ],
        final_result="Execution failed at step 2"
    )


@pytest.fixture
def sample_user_query():
    """Sample user query for testing."""
    return "Create event for tomorrow 5pm with title hello world"


@pytest.fixture
def sample_user_feedback():
    """Sample user feedback for testing plan modification."""
    return "Change the time to 6pm and add location"


@pytest.fixture
def sample_missing_fields():
    """Sample missing fields data for testing."""
    return {
        "event_time": "5pm",
        "event_location": "Conference Room A",
        "event_duration": "2 hours"
    }


@pytest.fixture
def sample_api_parameters():
    """Sample API parameters for testing."""
    return {
        "event_title": "Hello World",
        "event_date": "2024-01-15",
        "event_time": "5pm",
        "event_location": "Conference Room A",
        "event_duration": "2 hours"
    }


@pytest.fixture
def mock_websocket_service():
    """Create a mock WebSocket service for testing."""
    service = Mock()
    
    # Mock all the emit methods
    service.emit_connection_status = Mock()
    service.emit_next_best_items = Mock()
    service.emit_analyzing_status = Mock()
    service.emit_executing_status = Mock()
    service.emit_modifying_status = Mock()
    service.emit_execution_plan = Mock()
    service.emit_multiple_requests_complete = Mock()
    service.emit_error = Mock()
    service.emit_executing_step_status = Mock()
    service.emit_step_completed = Mock()
    service.emit_step_completed_status = Mock()
    
    return service


@pytest.fixture
def mock_api_service():
    """Create a mock API service for testing."""
    service = Mock()
    
    # Mock the search_api method
    service.search_api = Mock()
    service.search_api.return_value = Mock()
    
    # Mock the process_api_request method
    service.process_api_request = Mock()
    service.process_api_request.return_value = Mock()
    
    # Mock the enhance_request_with_context method
    service.enhance_request_with_context = Mock()
    service.enhance_request_with_context.return_value = "Enhanced request"
    
    return service


@pytest.fixture
def mock_executor_service():
    """Create a mock executor service for testing."""
    service = Mock()
    
    # Mock the generate_execution_plan method
    service.generate_execution_plan = Mock()
    service.generate_execution_plan.return_value = Mock()
    
    # Mock the modify_execution_plan method
    service.modify_execution_plan = Mock()
    service.modify_execution_plan.return_value = Mock()
    
    # Mock the execute_plan_steps method
    service.execute_plan_steps = Mock()
    service.execute_plan_steps.return_value = Mock()
    
    return service


# Test utilities
class TestUtils:
    """Utility class with helper methods for tests."""
    
    @staticmethod
    def assert_websocket_response_structure(response):
        """Assert that a WebSocket response has the correct structure."""
        assert isinstance(response, dict)
        assert 'name' in response
        assert 'args' in response
    
    @staticmethod
    def assert_api_call_parameters(mock_call, expected_method, expected_url, expected_payload=None):
        """Assert that an API call was made with the correct parameters."""
        mock_call.assert_called()
        
        if expected_payload:
            # Check if the call was made with the expected payload
            call_args = mock_call.call_args
            assert call_args is not None
    
    @staticmethod
    def create_mock_api_response(status_code=200, response_data=None, error_message=None):
        """Create a mock API response with specified parameters."""
        if response_data is None:
            response_data = {"result": "success"}
        
        if error_message:
            return ({"error": error_message}, status_code)
        else:
            return (response_data, status_code)
    
    @staticmethod
    def assert_execution_plan_structure(plan):
        """Assert that an execution plan has the correct structure."""
        assert hasattr(plan, 'description')
        assert hasattr(plan, 'steps')
        assert hasattr(plan, 'final_result')
        assert isinstance(plan.steps, list)
        
        for step in plan.steps:
            assert hasattr(step, 'step_number')
            assert hasattr(step, 'description')
            assert hasattr(step, 'api_description')
            assert hasattr(step, 'result_key')


# Make TestUtils available as a fixture
@pytest.fixture
def test_utils():
    """Provide test utilities to tests."""
    return TestUtils


# Markers for test categorization
pytest_plugins = []


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "websocket: mark test as a WebSocket test"
    )
    config.addinivalue_line(
        "markers", "api: mark test as an API test"
    )


