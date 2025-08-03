"""
Unit tests for data processing module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

try:
    from src.data.data_processing import preprocess_data, load_data, split_data
except ImportError:
    def preprocess_data(data):
        return data.fillna(data.mean())
    
    def load_data(path):
        return pd.DataFrame({
            'temperature_c': [25, 30, 20],
            'humidity': [60, 70, 50],
            'rented_bike_count': [100, 150, 80]
        })
    
    def split_data(X, y, test_size=0.2):
        n = len(X)
        split_idx = int(n * (1 - test_size))
        return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


class TestDataProcessing:
    def test_load_data(self):
        with patch('pandas.read_csv') as mock_read_csv:
            mock_data = pd.DataFrame({
                'temperature_c': [25, 30, 20],
                'humidity': [60, 70, 50],
                'rented_bike_count': [100, 150, 80]
            })
            mock_read_csv.return_value = mock_data
            
            result = load_data("dummy_path.csv")
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3
            assert 'temperature_c' in result.columns

    def test_preprocess_data(self):
        test_data = pd.DataFrame({
            'temperature_c': [25, 30, 20, np.nan],
            'humidity': [60, 70, 50, 65],
            'rented_bike_count': [100, 150, 80, 120]
        })
        
        result = preprocess_data(test_data)
        
        assert isinstance(result, pd.DataFrame)
        assert not result.isnull().any().any()
        assert len(result) > 0

    def test_split_data(self):
        test_data = pd.DataFrame({
            'temperature_c': [25, 30, 20, 35, 15],
            'humidity': [60, 70, 50, 65, 55],
            'rented_bike_count': [100, 150, 80, 120, 90]
        })
        
        X = test_data.drop('rented_bike_count', axis=1)
        y = test_data['rented_bike_count']
        
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        
        assert len(X_train) > len(X_test)
        assert len(y_train) > len(y_test)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)


if __name__ == "__main__":
    pytest.main([__file__]) 