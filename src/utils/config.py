import os
from typing import Optional

class Config:
    
    # AWS Configuration
    AWS_REGION = os.getenv('AWS_REGION', 'eu-north-1')
    S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'seoul-bike-sharing-aphdinh')
    
    # MLflow Configuration
    MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', "sqlite:///mlflow.db")
    EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', "seoul-bike-sharing")
    
    # Data Configuration
    DATA_FILE = os.getenv('DATA_FILE', 'SeoulBikeData.csv')
    DATA_S3_KEY = os.getenv('DATA_S3_KEY', f'data/{DATA_FILE}')
    
    # Model Configuration
    RANDOM_STATE = int(os.getenv('RANDOM_STATE', '42'))
    TEST_SIZE = float(os.getenv('TEST_SIZE', '0.2'))
    VALIDATION_SIZE = float(os.getenv('VALIDATION_SIZE', '0.25'))
    
    # Hyperparameter Tuning
    HYPEROPT_MAX_EVALS = int(os.getenv('HYPEROPT_MAX_EVALS', '20'))  # Reduced from 100
    RANDOM_SEARCH_N_ITER = int(os.getenv('RANDOM_SEARCH_N_ITER', '10'))  # Reduced from 50
    CV_FOLDS = int(os.getenv('CV_FOLDS', '3'))  # Reduced from 5
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    @classmethod
    def validate_aws_credentials(cls) -> bool:
        try:
            import boto3
            from botocore.exceptions import NoCredentialsError, ClientError
            
            # Test S3 access
            s3_client = boto3.client('s3', region_name=cls.AWS_REGION)
            s3_client.head_bucket(Bucket=cls.S3_BUCKET_NAME)
            return True
        except (NoCredentialsError, ClientError) as e:
            print(f"AWS credentials validation failed: {e}")
            return False
    
    @classmethod
    def get_s3_artifacts_path(cls, model_name: str, artifact_type: str, filename: str) -> str:
        return f"models/{model_name.lower().replace(' ', '_')}/{artifact_type}/{filename}"
    
    @classmethod
    def get_results_path(cls, filename: str) -> str:
        return f"results/{filename}"
    
    @classmethod
    def print_config(cls):
        print("=== Configuration ===")
        print(f"AWS Region: {cls.AWS_REGION}")
        print(f"S3 Bucket: {cls.S3_BUCKET_NAME}")
        print(f"MLflow Tracking URI: {cls.MLFLOW_TRACKING_URI}")
        print(f"Experiment Name: {cls.EXPERIMENT_NAME}")
        print(f"Data File: {cls.DATA_FILE}")
        print(f"Random State: {cls.RANDOM_STATE}")
        print(f"Test Size: {cls.TEST_SIZE}")
        print(f"Validation Size: {cls.VALIDATION_SIZE}")
        print(f"Hyperopt Max Evals: {cls.HYPEROPT_MAX_EVALS}")
        print(f"Random Search N Iter: {cls.RANDOM_SEARCH_N_ITER}")
        print(f"CV Folds: {cls.CV_FOLDS}")
        print(f"Log Level: {cls.LOG_LEVEL}")
        print("====================") 