#!/bin/bash

# Script to run tests with proper Python path setup

echo "Setting up Python path for testing..."

# Add the project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "Running tests..."

# Run different test suites
echo "=== Running Data Processing Tests ==="
python -m pytest tests/test_data_processing.py -v

echo "=== Running API Tests ==="
python -m pytest tests/test_api.py -v

echo "=== Running Monitoring Tests ==="
python -m pytest tests/test_monitoring.py -v

echo "=== Running All Tests ==="
python -m pytest tests/ -v

echo "Tests completed!" 