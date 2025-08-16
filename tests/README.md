# API Requester Test Suite

This directory contains comprehensive tests for the API Requester application, covering all the major functionality and edge cases.

## Test Coverage

The test suite covers all the required test cases mentioned in the original test file:

### 1. Test Generating Plan (`TestGeneratePlan`)
- ✅ Submit a query and assert response from websocket should be valid plan
- ✅ Assert fields in the result so that if something is changed, this test breaks
- ✅ Assert that API request to generate plan is made
- ✅ Test successful execution plan generation
- ✅ Test execution plan generation failure
- ✅ Test error handling when no query is provided

### 2. Test Executing Approved Plan (`TestExecuteApprovedPlan`)
- ✅ Websocket request to execute plan
- ✅ Assert that API request to execute plan is made
- ✅ Assert that API request to execute plan is made for each step
- ✅ Assert that API request to execute plan is made in the correct order
- ✅ Assert that API request to execute plan is made with the correct payload
- ✅ Assert that API request to execute plan is made with the correct headers
- ✅ Assert that API request to execute plan is made with the correct method
- ✅ Assert that API request to execute plan is made with the correct URL
- ✅ Test successful plan execution
- ✅ Test error handling when invalid plan is provided

### 3. Test Modifying Plan (`TestModifyPlan`)
- ✅ Websocket request to modify plan
- ✅ Assert that API request to modify plan is made
- ✅ Assert that API request to modify plan is made with the correct payload
- ✅ Assert that API request to modify plan is made with the correct headers
- ✅ Assert that API request to modify plan is made with the correct method
- ✅ Assert that API request to modify plan is made with the correct URL
- ✅ Test successful plan modification
- ✅ Test error handling when no feedback is provided

### 4. Test Approved Plan Contains Missing Fields (`TestApprovedPlanWithMissingFields`)
- ✅ Websocket request to execute plan
- ✅ Assert that API request to execute plan is made
- ✅ Assert that API request to execute plan is made with the correct payload
- ✅ Assert that API request to execute plan is made with the correct headers
- ✅ Assert that API request to execute plan is made with the correct method
- ✅ Assert that API request to execute plan is made with the correct URL
- ✅ Test plan execution when missing fields are encountered

### 5. Test Provide Missing Fields (`TestProvideMissingFields`)
- ✅ Websocket request to provide missing fields
- ✅ Assert that API request to provide missing fields is made
- ✅ Assert that API request to provide missing fields is made with the correct payload
- ✅ Assert that API request to provide missing fields is made with the correct headers
- ✅ Assert that API request to provide missing fields is made with the correct method
- ✅ Assert that API request to provide missing fields is made with the correct URL
- ✅ Test successful missing fields provision
- ✅ Test error handling when API call fails during missing fields provision

### 6. Test Failing Execution is Handled Gracefully (`TestFailingExecutionHandling`)
- ✅ Websocket request to execute plan
- ✅ Assert that API request to execute plan is made
- ✅ Assert that API request to execute plan is made with the correct payload
- ✅ Assert that API request to execute plan is made with the correct headers
- ✅ Assert that API request to execute plan is made with the correct method
- ✅ Assert that API request to execute plan is made with the correct URL
- ✅ Test that exceptions during plan execution are handled gracefully
- ✅ Test that API failures during plan execution are handled gracefully

### Additional Test Categories

- **WebSocket Connection Tests** (`TestWebSocketConnection`)
  - Connection establishment
  - Disconnection handling

- **API Request Validation Tests** (`TestAPIRequestValidation`)
  - Empty query handling
  - Missing query field handling

- **Execution Plan Validation Tests** (`TestExecutionPlanValidation`)
  - Plan without steps
  - None plan result handling

## Test Structure

Each test class focuses on a specific functionality area and includes:

- **Success scenarios**: Testing normal operation paths
- **Failure scenarios**: Testing error handling and edge cases
- **Validation**: Ensuring correct parameters, headers, methods, and URLs
- **Mocking**: All external API calls are mocked to ensure isolated testing

## Running the Tests

### Prerequisites

Install the required testing dependencies:

```bash
pip install -r requirements.txt
```

### Running All Tests

```bash
# From the api_requester directory
python -m pytest tests/

# Or use the test runner script
python run_tests.py
```

### Running Specific Test Categories

```bash
# Run only unit tests
python run_tests.py --type unit

# Run only integration tests
python run_tests.py --type integration

# Run only slow tests
python run_tests.py --type slow
```

### Running with Coverage

```bash
# Run tests with coverage reporting
python run_tests.py --coverage

# Or directly with pytest
python -m pytest tests/ --cov=api_requester --cov-report=html --cov-report=term
```

### Verbose Output

```bash
# Run tests with verbose output
python run_tests.py --verbose

# Or directly with pytest
python -m pytest tests/ -v
```

## Test Configuration

The test suite uses `pytest.ini` for configuration:

- **Test discovery**: Automatically finds test files in the `tests/` directory
- **Markers**: Supports test categorization (unit, integration, slow)
- **Warnings**: Filters out deprecation warnings
- **Output**: Configures test output format

## Mocking Strategy

The tests use comprehensive mocking to ensure:

- **Isolation**: Tests don't depend on external services
- **Predictability**: Consistent test results regardless of external state
- **Speed**: Fast test execution without network calls
- **Control**: Ability to simulate various scenarios (success, failure, errors)

### Mocked Components

- **External API calls**: All HTTP requests are mocked
- **Service dependencies**: Internal services are mocked where appropriate
- **WebSocket responses**: Socket.IO responses are captured and verified
- **File operations**: Schema loading and other I/O operations are mocked

## Test Data

The tests use realistic test data:

- **Execution plans**: Multi-step plans with various configurations
- **API responses**: Success and failure responses with different status codes
- **User inputs**: Realistic queries and feedback
- **Error scenarios**: Various failure modes and edge cases

## Continuous Integration

The test suite is designed to be run in CI/CD pipelines:

- **Fast execution**: Tests complete quickly for rapid feedback
- **Deterministic**: Consistent results across different environments
- **Comprehensive**: Covers all critical functionality
- **Maintainable**: Clear test structure and documentation

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure you're running from the correct directory
2. **Missing dependencies**: Install all requirements from `requirements.txt`
3. **Mock failures**: Check that mocks are properly configured
4. **WebSocket errors**: Verify SocketIO test client setup

### Debug Mode

Run tests with debug output:

```bash
python -m pytest tests/ -v -s --tb=long
```

### Running Single Tests

```bash
# Run a specific test class
python -m pytest tests/test_api_requester.py::TestGeneratePlan

# Run a specific test method
python -m pytest tests/test_api_requester.py::TestGeneratePlan::test_generate_execution_plan_success
```

## Contributing

When adding new tests:

1. Follow the existing test structure and naming conventions
2. Include both success and failure scenarios
3. Use descriptive test names and docstrings
4. Mock external dependencies appropriately
5. Add tests to the appropriate test class
6. Update this README if adding new test categories

## Test Metrics

The test suite provides:

- **Coverage reporting**: HTML and terminal coverage reports
- **Test categorization**: Unit, integration, and slow test markers
- **Performance metrics**: Test execution time and resource usage
- **Quality metrics**: Test success rates and failure analysis


