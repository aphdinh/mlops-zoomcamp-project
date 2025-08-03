import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'temperature_c': [25, 30, 20, 35, 15],
        'humidity': [60, 70, 50, 65, 55],
        'wind_speed': [2.0, 3.0, 1.5, 2.5, 1.0],
        'visibility_10m': [2000, 1500, 1800, 1600, 2200],
        'dew_point_c': [15, 20, 10, 18, 8],
        'solar_radiation': [0.5, 0.8, 0.3, 0.6, 0.2],
        'rainfall_mm': [0.0, 0.1, 0.0, 0.05, 0.0],
        'snowfall_cm': [0.0, 0.0, 0.0, 0.0, 0.0],
        'season': ['Spring', 'Summer', 'Autumn', 'Winter', 'Spring'],
        'holiday': ['No Holiday', 'Holiday', 'No Holiday', 'No Holiday', 'Holiday'],
        'functioning_day': ['Yes', 'Yes', 'Yes', 'Yes', 'Yes'],
        'rented_bike_count': [100, 150, 80, 120, 90]
    })


@pytest.fixture
def mock_monitor():
    monitor = MagicMock()
    monitor.check_data_drift.return_value = {
        'drift_detected': False,
        'drift_score': 0.1,
        'drifted_columns': 0,
        'total_columns': 10
    }
    monitor.check_data_quality.return_value = {
        'total_rows': 100,
        'missing_values': {},
        'duplicate_rows': 0
    }
    monitor.check_model_performance.return_value = {
        'mae': 10.0,
        'rmse': 15.0,
        'r2_score': 0.85
    }
    return monitor


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict.return_value = np.array([100, 150, 80, 120, 90])
    return model


@pytest.fixture
def sample_predictions():
    """Sample predictions for testing."""
    return [100, 150, 80, 120, 90]


@pytest.fixture
def sample_actuals():
    """Sample actual values for testing."""
    return [95, 160, 85, 125, 88]


@pytest.fixture
def api_client():
    try:
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        
        app = FastAPI()
        client = TestClient(app)
        return client
    except ImportError:
        # Return a mock client if FastAPI is not available
        mock_client = MagicMock()
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"status": "ok"}
        )
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"result": "success"}
        )
        return mock_client


def pytest_configure(config):
    """Configure pytest."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "monitoring: mark test as a monitoring test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    for item in items:
        # Add default markers based on test file names
        if "test_data_processing" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        elif "test_api" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "test_monitoring" in item.nodeid:
            item.add_marker(pytest.mark.monitoring) 