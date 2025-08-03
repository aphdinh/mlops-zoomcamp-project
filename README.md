# Seoul Bike Sharing Prediction - MLOps Zoomcamp Capstone Project

A complete end-to-end machine learning project for predicting bike rental demand in Seoul, implementing industry-standard MLOps practices with cloud deployment and monitoring.

## 📋 About

This project demonstrates a production-ready MLOps pipeline for predicting Seoul bike sharing demand. Built with modern tools like MLflow, Prefect, FastAPI, and Evidently, it showcases best practices in machine learning operations including experiment tracking, workflow orchestration, model deployment, and continuous monitoring.

**Key Features:**
- 🚀 **End-to-end ML pipeline** with automated training and deployment
- 📊 **Real-time monitoring** with data drift detection and performance tracking  
- ☁️ **Cloud-native** with AWS infrastructure and Terraform IaC
- 🔄 **CI/CD pipeline** with GitHub Actions for automated testing and deployment
- 📈 **Model performance** achieving 85-94% R² score with LightGBM/XGBoost

## 🎯 Problem Statement

Predict the number of bikes rented in Seoul based on weather conditions, time features, and seasonal patterns. This project demonstrates a production-ready ML pipeline with automated training, deployment, monitoring, and retraining capabilities.

## 📊 Data Source

This project uses the **Seoul Bike Sharing Demand** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Seoul+Bike+Sharing+Demand).

**Dataset Characteristics:**
- **Size**: 8,760 instances (1 year of hourly data)
- **Features**: 13 variables including weather data and temporal features
- **Target**: Rented Bike Count (integer)
- **Missing Values**: None
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)

**Key Features:**
- **Weather Data**: Temperature (°C), Humidity (%), Wind Speed (m/s), Visibility (10m), Dew Point (°C), Solar Radiation (MJ/m²), Rainfall (mm), Snowfall (cm)
- **Temporal Features**: Date, Hour, Season (Winter/Spring/Summer/Autumn)
- **Operational Features**: Holiday status, Functional Day indicator

**Citation:**
Seoul Bike Sharing Demand [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5F62R.

## ⚡️ Architecture & Tech Stack

### **Cloud Infrastructure (AWS)**
- **S3**: Artifact storage and model versioning
- **EC2**: Training and deployment instances
- **Terraform**: Infrastructure as Code (IaC)

### **MLOps Tools**
- **MLflow**: Experiment tracking and model registry
- **Prefect**: Workflow orchestration and scheduling
- **Evidently**: Model monitoring and data drift detection
- **FastAPI**: Model serving and REST API

### **Development & Deployment**
- **Docker**: Containerization
- **GitHub Actions**: CI/CD pipeline
- **Pytest**: Comprehensive testing framework
- **Makefile**: Build automation and task management

## 🚀 Key Features

### **Experiment Tracking & Model Registry**
- Complete MLflow integration for experiment tracking
- Automated model registration with production staging
- Hyperparameter optimization with Hyperopt
- Model versioning and artifact management

### **Workflow Orchestration**
- Prefect-based pipeline with task dependencies
- Automated retraining schedules
- Error handling and retry mechanisms
- Parallel model training and evaluation

### **Model Deployment**
- FastAPI web service with batch prediction
- Docker containerization for portability
- Health checks and monitoring endpoints

### **Comprehensive Monitoring**
- Data quality assessment and data drift detection with Evidently
- Model performance monitoring
- Automated report generation

## 📊 Model Performance

The pipeline trains and evaluates 12+ ML algorithms:
- **Best Model**: LightGBM/XGBoost
- **R² Score**: 0.85-0.94
- **RMSE**: 200-400 bikes
- **Training Time**: 5-15 minutes

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- AWS CLI configured
- Terraform (for cloud deployment)

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd mlops

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
# Set up environment variables
export AWS_REGION="eu-north-1"
export S3_BUCKET_NAME="seoul-bike-sharing-aphdinh"
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
export MLFLOW_ARTIFACT_URI="s3://seoul-bike-sharing-aphdinh/mlflow-artifacts/"

# Or use the provided script
chmod +x scripts/setup/server-start.sh
./scripts/setup/server-start.sh
```

### 3. Deploy AWS Infrastructure (Optional)

```bash
# Deploy AWS resources
cd terraform
terraform init
terraform plan
terraform apply
cd ..
```

## 🚀 Running the Application

### Quick Start with Makefile

```bash
# Install dependencies
make install

# Setup environment
make setup

# Train models
make train

# Start API server
make api

# Run tests
make test
```

### Manual Commands

```bash
# Option 1: Run with Prefect orchestration
python src/training/train.py prefect

# Option 2: Run core training pipeline
python src/training/train.py core

# Option 3: Create Prefect deployment
python src/training/train.py deploy

# Start FastAPI service
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Access Services

- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health
- **MLflow UI**: http://localhost:5000 (if running)

## 📁 Project Structure

```
mlops/
├── src/                      # Source code
│   ├── api/                  # FastAPI application
│   ├── training/             # Training pipeline
│   ├── monitoring/           # Model monitoring
│   ├── models/               # Model definitions
│   ├── data/                 # Data processing
│   └── utils/                # Utility functions
├── tests/                    # Test suite
├── config/                   # Configuration files
├── data/                     # Training and monitoring data
├── notebook/                 # Jupyter notebooks (EDA & Models)
├── terraform/                # Infrastructure as Code
├── artifacts/                # Generated models and reports
├── requirements.txt          # Dependencies
├── Makefile                  # Build automation
└── README.md                 # This file
```

## 📈 Monitoring & Testing

### Test Monitoring System

```bash
# Test monitoring functionality
python src/monitoring/test_monitoring.py

# Run integration example
python src/monitoring/integration_example.py

# Run all tests
make test
```

### API Testing

```bash
# Test prediction endpoint
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "date": "01/01/2024",
    "hour": 12,
    "temperature_c": 25.0,
    "humidity": 60.0,
    "wind_speed": 2.0,
    "visibility_10m": 2000.0,
    "dew_point_c": 15.0,
    "solar_radiation": 0.5,
    "rainfall_mm": 0.0,
    "snowfall_cm": 0.0,
    "season": "Spring",
    "holiday": "No Holiday",
    "functioning_day": "Yes"
  }'
```

### Monitoring Endpoints

```bash
# Check monitoring status
curl http://localhost:8000/monitoring/status

# Generate data drift report
curl -X POST http://localhost:8000/monitoring/data-drift

# Generate data quality report
curl -X POST http://localhost:8000/monitoring/data-quality
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AWS_REGION` | `eu-north-1` | AWS region |
| `S3_BUCKET_NAME` | `seoul-bike-sharing-aphdinh` | S3 bucket for artifacts |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | MLflow tracking URI |
| `MLFLOW_ARTIFACT_URI` | `s3://...` | S3 artifact storage |

### S3 Bucket Structure

```
s3://seoul-bike-sharing-aphdinh/
├── data/                    # Training data
├── models/                  # Trained models
├── artifacts/              # MLflow artifacts
└── reports/               # Monitoring reports
```

## 📊 Results & Artifacts

### MLflow Experiments
- Complete experiment history
- Model comparison metrics
- Hyperparameter optimization results
- Artifact versioning

### S3 Artifacts
- Trained models with metadata
- Feature importance plots
- Training reports and summaries
- Monitoring dashboards

## 🎯 Course Requirements Fulfillment

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| **Problem Description** | ✅ Complete | Seoul bike sharing prediction with clear objectives |
| **Cloud Integration** | ✅ Complete | AWS S3, EC2, Terraform IaC |
| **Experiment Tracking** | ✅ Complete | MLflow with model registry |
| **Workflow Orchestration** | ✅ Complete | Prefect with scheduling |
| **Model Deployment** | ✅ Complete | FastAPI + Docker |
| **Model Monitoring** | ✅ Complete | Evidently with alerts |
| **Reproducibility** | ✅ Complete | Requirements.txt + setup instructions |
| **Best Practices** | ✅ Complete | Tests, CI/CD, documentation, Makefile |

## 🚀 Deployment Options

### Local Development
```bash
make dev  # Full development setup
```

### Cloud Deployment
```bash
make deploy-infra  # Deploy infrastructure
python src/training/train.py prefect  # Run training on cloud
```

## 📚 Documentation

- [API Documentation](http://localhost:8000/docs)
- [Monitoring Setup](docs/monitoring/)
- [Deployment Guide](docs/deployment/)

---

**Built with ❤️ using modern MLOps practices** 