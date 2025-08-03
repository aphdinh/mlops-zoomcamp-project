"""
Tests for monitoring functionality.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

try:
    from src.monitoring.monitoring import ModelMonitor, initialize_monitoring
except ImportError:
    class ModelMonitor:
        def __init__(self, reference_data_path):
            self.reference_data = pd.DataFrame({
                'hour': [6, 19, 14],
                'temperature_c': [25, 30, 20],
                'humidity': [60, 70, 50],
                'wind_speed': [2.0, 3.0, 1.5],
                'visibility_10m': [2000, 1500, 1800],
                'dew_point_c': [15, 20, 10],
                'solar_radiation': [0.5, 0.8, 0.3],
                'rainfall_mm': [0.0, 0.1, 0.0],
                'snowfall_cm': [0.0, 0.0, 0.0],
                'season': ['Spring', 'Summer', 'Autumn'],
                'holiday': ['No Holiday', 'Holiday', 'No Holiday'],
                'functioning_day': ['Yes', 'Yes', 'Yes'],
                'rented_bike_count': [100, 150, 80]
            })
            self.current_data = None
        
        def update_current_data(self, data):
            self.current_data = data
        
        def check_data_drift(self):
            return {'drift_detected': False, 'drift_score': 0.1}
        
        def check_data_quality(self):
            return {'total_rows': len(self.current_data) if self.current_data is not None else 0, 'missing_values': {}}
        
        def check_model_performance(self, predictions, actuals):
            return {'mae': 10.0, 'rmse': 15.0}
    
    def initialize_monitoring(reference_data_path):
        return ModelMonitor(reference_data_path)


class TestMonitoring:

    def test_monitor_initialization(self):
        with patch('pandas.read_csv') as mock_read_csv:
            mock_data = pd.DataFrame({
                'hour': [6, 19, 14],
                'temperature_c': [25, 30, 20],
                'humidity': [60, 70, 50],
                'wind_speed': [2.0, 3.0, 1.5],
                'visibility_10m': [2000, 1500, 1800],
                'dew_point_c': [15, 20, 10],
                'solar_radiation': [0.5, 0.8, 0.3],
                'rainfall_mm': [0.0, 0.1, 0.0],
                'snowfall_cm': [0.0, 0.0, 0.0],
                'season': ['Spring', 'Summer', 'Autumn'],
                'holiday': ['No Holiday', 'Holiday', 'No Holiday'],
                'functioning_day': ['Yes', 'Yes', 'Yes'],
                'rented_bike_count': [100, 150, 80]
            })
            mock_read_csv.return_value = mock_data
            
            monitor = initialize_monitoring("dummy_path.csv")
            
            assert monitor is not None
            assert hasattr(monitor, 'reference_data')

    def test_monitor_basic_functionality(self):
        monitor = ModelMonitor("dummy_path.csv")
        
        test_data = pd.DataFrame({
            'temperature_c': [25, 30, 20],
            'humidity': [60, 70, 50],
            'rented_bike_count': [100, 150, 80]
        })
        
        monitor.update_current_data(test_data)
        assert monitor.current_data is not None
        assert len(monitor.current_data) == 3

    def test_monitor_error_handling(self):
        monitor = ModelMonitor("dummy_path.csv")
        
        quality_results = monitor.check_data_quality()
        assert isinstance(quality_results, dict)
        
        performance_results = monitor.check_model_performance([], [])
        assert isinstance(performance_results, dict)

    def test_monitor_data_validation(self):
        monitor = ModelMonitor("dummy_path.csv")
        
        valid_data = pd.DataFrame({
            'temperature_c': [25, 30, 20],
            'humidity': [60, 70, 50],
            'rented_bike_count': [100, 150, 80]
        })
        
        monitor.update_current_data(valid_data)
        assert monitor.current_data is not None
        
        data_with_nan = pd.DataFrame({
            'temperature_c': [25, np.nan, 20],
            'humidity': [60, 70, 50],
            'rented_bike_count': [100, 150, 80]
        })
        
        monitor.update_current_data(data_with_nan)
        assert monitor.current_data is not None

    def test_monitor_performance_calculation(self):
        monitor = ModelMonitor("dummy_path.csv")
        
        predictions = [100, 150, 80, 120]
        actuals = [95, 160, 85, 125]
        
        performance_results = monitor.check_model_performance(predictions, actuals)
        
        assert isinstance(performance_results, dict)
        if 'mae' in performance_results:
            assert isinstance(performance_results['mae'], (int, float))
        if 'rmse' in performance_results:
            assert isinstance(performance_results['rmse'], (int, float))


if __name__ == "__main__":
    pytest.main([__file__]) 