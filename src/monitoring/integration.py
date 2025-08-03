#!/usr/bin/env python3

import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os
import sys

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .monitoring import initialize_monitoring

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLWorkflow:
    
    def __init__(self, reference_data_path="data/reference_data.csv"):
        self.monitor = initialize_monitoring(reference_data_path)
        self.model = None  # In real scenario, this would be your trained model
        logger.info("ML Workflow initialized with monitoring")
    
    def load_and_preprocess_data(self, data_path):
        logger.info(f"Loading data from {data_path}")
        
        try:
            data = pd.read_csv(data_path)
            logger.info(f"Loaded data shape: {data.shape}")
            
            # Basic preprocessing (in real scenario, you'd have more sophisticated preprocessing)
            # Remove duplicates
            data = data.drop_duplicates()
            
            # Handle missing values
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mean())
            
            # Update current data in monitor
            self.monitor.update_current_data(data)
            
            logger.info(f"Preprocessed data shape: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading/preprocessing data: {e}")
            return None
    
    def make_predictions(self, data):
        logger.info("Making predictions...")
        
        # In real scenario, this would be your actual model prediction
        # For demonstration, we'll create mock predictions
        np.random.seed(42)
        
        # Use actual values as base and add some noise for realistic predictions
        base_predictions = data['rented_bike_count'].values
        
        # Simulate model predictions with some error
        predictions = base_predictions + np.random.normal(0, 3, len(base_predictions))
        
        # Ensure predictions are non-negative
        predictions = np.maximum(predictions, 0)
        
        logger.info(f"Generated {len(predictions)} predictions")
        return predictions
    
    def run_monitoring_checks(self, data, predictions=None):
        logger.info("Running monitoring checks...")
        
        monitoring_results = {}
        
        # 1. Check data drift
        logger.info("Checking data drift...")
        drift_results = self.monitor.check_data_drift()
        monitoring_results['drift'] = drift_results
        
        # 2. Check data quality
        logger.info("Checking data quality...")
        quality_results = self.monitor.check_data_quality()
        monitoring_results['quality'] = quality_results
        
        # 3. Check model performance (if predictions provided)
        if predictions is not None and 'rented_bike_count' in data.columns:
            logger.info("Checking model performance...")
            performance_results = self.monitor.check_model_performance(
                predictions.tolist(), 
                data['rented_bike_count'].tolist()
            )
            monitoring_results['performance'] = performance_results
        
        logger.info("Monitoring checks completed")
        return monitoring_results
    
    def generate_alerts(self, monitoring_results):
        alerts = []
        
        # Check for data drift alerts
        if 'drift' in monitoring_results:
            drift_info = monitoring_results['drift']
            if 'drift_detected' in drift_info and drift_info['drift_detected']:
                alerts.append({
                    'type': 'data_drift',
                    'severity': 'high',
                    'message': f"Data drift detected with score: {drift_info.get('drift_score', 'unknown')}",
                    'timestamp': datetime.now().isoformat()
                })
        
        # Check for data quality alerts
        if 'quality' in monitoring_results:
            quality_info = monitoring_results['quality']
            if 'total_rows' in quality_info and quality_info['total_rows'] < 100:
                alerts.append({
                    'type': 'data_quality',
                    'severity': 'medium',
                    'message': f"Low data volume: {quality_info['total_rows']} rows",
                    'timestamp': datetime.now().isoformat()
                })
        
        # Check for model performance alerts
        if 'performance' in monitoring_results:
            perf_info = monitoring_results['performance']
            if 'r2_score' in perf_info and perf_info['r2_score'] < 0.7:
                alerts.append({
                    'type': 'model_performance',
                    'severity': 'high',
                    'message': f"Low model performance: R² = {perf_info['r2_score']:.4f}",
                    'timestamp': datetime.now().isoformat()
                })
        
        return alerts
    
    def run_workflow(self, data_path):
        logger.info("Starting ML workflow with monitoring...")
        
        # 1. Load and preprocess data
        data = self.load_and_preprocess_data(data_path)
        if data is None:
            logger.error("Failed to load data")
            return None
        
        # 2. Make predictions
        predictions = self.make_predictions(data)
        
        # 3. Run monitoring checks
        monitoring_results = self.run_monitoring_checks(data, predictions)
        
        # 4. Generate alerts
        alerts = self.generate_alerts(monitoring_results)
        
        # 5. Report results
        self.report_results(monitoring_results, alerts)
        
        return {
            'data_shape': data.shape,
            'predictions_count': len(predictions),
            'monitoring_results': monitoring_results,
            'alerts_count': len(alerts)
        }
    
    def report_results(self, monitoring_results, alerts):
        logger.info("=== Monitoring Report ===")
        
        # Data drift summary
        if 'drift' in monitoring_results:
            drift = monitoring_results['drift']
            logger.info(f"Data Drift: {'Detected' if drift.get('drift_detected', False) else 'Not Detected'}")
            if 'drift_score' in drift:
                logger.info(f"Drift Score: {drift['drift_score']:.4f}")
        
        # Data quality summary
        if 'quality' in monitoring_results:
            quality = monitoring_results['quality']
            logger.info(f"Data Quality: {quality.get('total_rows', 'Unknown')} rows")
            if 'missing_values' in quality:
                logger.info(f"Missing Values: {len(quality['missing_values'])} columns")
        
        # Model performance summary
        if 'performance' in monitoring_results:
            perf = monitoring_results['performance']
            logger.info(f"Model Performance: R² = {perf.get('r2_score', 'Unknown'):.4f}")
            logger.info(f"RMSE: {perf.get('rmse', 'Unknown'):.2f}")
            logger.info(f"MAE: {perf.get('mae', 'Unknown'):.2f}")
        
        # Alerts summary
        logger.info(f"Alerts Generated: {len(alerts)}")
        for alert in alerts:
            logger.warning(f"Alert [{alert['severity'].upper()}]: {alert['message']}")
        
        logger.info("=== End Report ===")

def main():
    # Example usage
    workflow = MLWorkflow()
    
    # Run the workflow with sample data
    # In a real scenario, you would use actual data
    result = workflow.run_workflow("data/current_data.csv")
    
    if result:
        print(f"Workflow completed successfully!")
        print(f"Processed {result['data_shape'][0]} data points")
        print(f"Generated {result['predictions_count']} predictions")
        print(f"Found {result['alerts_count']} alerts")
    else:
        print("Workflow failed!")

if __name__ == "__main__":
    main() 