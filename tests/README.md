# Test Suite

This directory contains the test suite for the Seoul Bike Sharing MLOps Pipeline.

## Test Structure

### `test_data_processing.py`
- Unit tests for data processing functionality
- Tests data loading, preprocessing, and splitting
- Uses mock implementations for missing dependencies

### `test_api.py`
- Integration tests for API endpoints
- Tests FastAPI application functionality
- Includes error handling and validation tests

### `test_monitoring.py`
- Tests for monitoring system functionality
- Tests data drift detection and quality assessment
- Uses mock implementations for complex dependencies

## Running Tests

### Run All Tests
```bash
make test-all
```

### Run Specific Test Suites
```bash
# Data processing tests
python -m pytest tests/test_data_processing.py -v

# API tests
python -m pytest tests/test_api.py -v

# Monitoring tests
python -m pytest tests/test_monitoring.py -v
```

### Run with Coverage
```bash
python -m pytest tests/ --cov=src --cov-report=html
```

## Test Design

The tests are designed to be:
- **Independent**: Each test can run without dependencies on other tests
- **Mock-based**: Uses mocks for external dependencies
- **Resilient**: Handles missing modules gracefully
- **Comprehensive**: Covers unit, integration, and monitoring functionality

## Adding New Tests

When adding new tests:
1. Follow the existing naming convention: `test_*.py`
2. Use descriptive test method names
3. Include proper docstrings
4. Handle missing dependencies gracefully
5. Use mocks for external services 