#!/usr/bin/env python3

import pandas as pd
import numpy as np
import logging
from .monitoring import ModelMonitor

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def create_sample_data():
    np.random.seed(42)
    n_samples = 100
    
    reference_data = pd.DataFrame({
        'hour': np.random.randint(0, 24, n_samples),
        'temperature_c': np.random.normal(20, 10, n_samples),
        'humidity': np.random.uniform(30, 90, n_samples),
        'wind_speed': np.random.exponential(5, n_samples),
        'visibility_10m': np.random.uniform(0, 30, n_samples),
        'dew_point_c': np.random.normal(10, 8, n_samples),
        'solar_radiation': np.random.exponential(2, n_samples),
        'rainfall_mm': np.random.exponential(0.5, n_samples),
        'snowfall_cm': np.random.exponential(0.1, n_samples),
        'year': 2023,
        'month': np.random.randint(1, 13, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'hour_sin': np.sin(2 * np.pi * np.random.randint(0, 24, n_samples) / 24),
        'hour_cos': np.cos(2 * np.pi * np.random.randint(0, 24, n_samples) / 24),
        'month_sin': np.sin(2 * np.pi * np.random.randint(1, 13, n_samples) / 12),
        'month_cos': np.cos(2 * np.pi * np.random.randint(1, 13, n_samples) / 12),
        'day_of_week_sin': np.sin(2 * np.pi * np.random.randint(0, 7, n_samples) / 7),
        'day_of_week_cos': np.cos(2 * np.pi * np.random.randint(0, 7, n_samples) / 7),
        'temp_humidity_interaction': np.random.normal(20, 10, n_samples) * np.random.uniform(30, 90, n_samples) / 100,
        'wind_rain_interaction': np.random.exponential(5, n_samples) * np.random.exponential(0.5, n_samples),
        'temp_solar_interaction': np.random.normal(20, 10, n_samples) * np.random.exponential(2, n_samples),
        'season': np.random.choice(['Spring', 'Summer', 'Autumn', 'Winter'], n_samples),
        'holiday': np.random.choice(['Holiday', 'No Holiday'], n_samples),
        'functioning_day': np.random.choice(['Yes', 'No'], n_samples),
        'time_of_day': np.random.choice(['Night', 'Morning', 'Afternoon', 'Evening'], n_samples),
        'temp_feel': np.random.choice(['Very_Cold', 'Cold', 'Mild', 'Warm'], n_samples),
        'is_weekend': np.random.choice([0, 1], n_samples),
        'is_rush_hour': np.random.choice([0, 1], n_samples),
        'has_rain': np.random.choice([0, 1], n_samples),
        'has_snow': np.random.choice([0, 1], n_samples),
        'is_holiday': np.random.choice([0, 1], n_samples),
        'is_functioning': np.random.choice([0, 1], n_samples),
        'is_spring': np.random.choice([0, 1], n_samples),
        'is_summer': np.random.choice([0, 1], n_samples),
        'is_autumn': np.random.choice([0, 1], n_samples),
        'is_winter': np.random.choice([0, 1], n_samples),
        'extreme_weather': np.random.choice([0, 1], n_samples),
        'rented_bike_count': np.random.poisson(50, n_samples)
    })
    
    current_data = reference_data.copy()
    current_data['temperature_c'] += np.random.normal(5, 2, n_samples)
    current_data['humidity'] -= np.random.normal(10, 5, n_samples)
    
    return reference_data, current_data

def test_simple_monitoring():
    try:
        logger.info("Testing simplified monitoring system...")
        
        reference_data, current_data = create_sample_data()
        
        import os
        os.makedirs("data", exist_ok=True)
        reference_data.to_csv("data/reference_data.csv", index=False)
        logger.info("Created sample reference data")
        
        monitor = ModelMonitor("data/reference_data.csv")
        monitor.update_current_data(current_data)
        
        logger.info("Testing data drift check...")
        drift_results = monitor.check_data_drift()
        logger.info(f"Drift results: {drift_results}")
        
        logger.info("Testing data quality check...")
        quality_results = monitor.check_data_quality()
        logger.info(f"Quality results: {quality_results}")
        
        logger.info("Testing model performance check...")
        predictions = [45, 50, 55, 40, 60]
        actuals = [42, 48, 52, 38, 58]
        performance_results = monitor.check_model_performance(predictions, actuals)
        logger.info(f"Performance results: {performance_results}")
        
        logger.info("Testing comprehensive monitoring...")
        comprehensive_results = monitor.run_monitoring()
        logger.info(f"Comprehensive results: {comprehensive_results}")

        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    test_simple_monitoring() 