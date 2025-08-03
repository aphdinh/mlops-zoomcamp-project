# Seoul Bike Sharing Prediction API - Curl Commands

This file contains individual curl commands to test each endpoint of the FastAPI application.

## Prerequisites

1. Start the server:
```bash
cd src
python app.py
```

2. The server will run on `http://localhost:8000`

## Available Endpoints

### 1. Single Prediction Endpoint

**Endpoint:** `POST /predict`

**Command:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "date": "15/12/2017",
    "hour": 14,
    "temperature_c": 15.0,
    "humidity": 65.0,
    "wind_speed": 2.5,
    "visibility_10m": 2000.0,
    "dew_point_c": 8.0,
    "solar_radiation": 0.5,
    "rainfall_mm": 0.0,
    "snowfall_cm": 0.0,
    "season": "Winter",
    "holiday": "No Holiday",
    "functioning_day": "Yes"
  }'
```

### 2. Batch Prediction Endpoint

**Endpoint:** `POST /predict/batch`

**Command:**
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {
        "date": "15/12/2017",
        "hour": 14,
        "temperature_c": 15.0,
        "humidity": 65.0,
        "wind_speed": 2.5,
        "visibility_10m": 2000.0,
        "dew_point_c": 8.0,
        "solar_radiation": 0.5,
        "rainfall_mm": 0.0,
        "snowfall_cm": 0.0,
        "season": "Winter",
        "holiday": "No Holiday",
        "functioning_day": "Yes"
      },
      {
        "date": "16/12/2017",
        "hour": 9,
        "temperature_c": 12.0,
        "humidity": 70.0,
        "wind_speed": 1.8,
        "visibility_10m": 1800.0,
        "dew_point_c": 6.0,
        "solar_radiation": 0.3,
        "rainfall_mm": 0.0,
        "snowfall_cm": 0.0,
        "season": "Winter",
        "holiday": "No Holiday",
        "functioning_day": "Yes"
      }
    ]
  }'
```

### 3. S3 Model Completeness Check

**Endpoint:** `GET /check-s3-model-completeness`

**Command:**
```bash
curl -X GET "http://localhost:8000/check-s3-model-completeness"
```

### 4. API Documentation

**Endpoint:** `GET /docs`

**Command:**
```bash
curl -X GET "http://localhost:8000/docs"
```

### 5. OpenAPI Schema

**Endpoint:** `GET /openapi.json`

**Command:**
```bash
curl -X GET "http://localhost:8000/openapi.json"
```

## Testing with Different Data Scenarios

### Test with Summer Data
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "date": "15/07/2017",
    "hour": 16,
    "temperature_c": 28.0,
    "humidity": 45.0,
    "wind_speed": 1.2,
    "visibility_10m": 3000.0,
    "dew_point_c": 15.0,
    "solar_radiation": 2.1,
    "rainfall_mm": 0.0,
    "snowfall_cm": 0.0,
    "season": "Summer",
    "holiday": "No Holiday",
    "functioning_day": "Yes"
  }'
```

### Test with Holiday Data
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "date": "25/12/2017",
    "hour": 10,
    "temperature_c": 5.0,
    "humidity": 80.0,
    "wind_speed": 3.5,
    "visibility_10m": 1500.0,
    "dew_point_c": 2.0,
    "solar_radiation": 0.2,
    "rainfall_mm": 2.5,
    "snowfall_cm": 0.0,
    "season": "Winter",
    "holiday": "Holiday",
    "functioning_day": "Yes"
  }'
```

### Test with Rainy Weather
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "date": "20/09/2017",
    "hour": 8,
    "temperature_c": 18.0,
    "humidity": 90.0,
    "wind_speed": 4.2,
    "visibility_10m": 800.0,
    "dew_point_c": 16.0,
    "solar_radiation": 0.1,
    "rainfall_mm": 15.0,
    "snowfall_cm": 0.0,
    "season": "Autumn",
    "holiday": "No Holiday",
    "functioning_day": "Yes"
  }'
```

### Test with Invalid Data (Missing Fields)
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "date": "15/12/2017",
    "hour": 14,
    "temperature_c": 15.0
  }'
```

### Test with Boundary Values
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "date": "01/01/2017",
    "hour": 0,
    "temperature_c": -20.0,
    "humidity": 0.0,
    "wind_speed": 0.0,
    "visibility_10m": 0.0,
    "dew_point_c": -25.0,
    "solar_radiation": 0.0,
    "rainfall_mm": 0.0,
    "snowfall_cm": 0.0,
    "season": "Winter",
    "holiday": "No Holiday",
    "functioning_day": "Yes"
  }'
```

## Using the Automated Test Script

Run the complete test suite:

```bash
./test_endpoints.sh
```

This will test all endpoints automatically and provide colored output with results.

## Expected Responses

### Successful Prediction Response
```json
{
  "prediction": 123,
  "confidence": 0.85,
  "model_info": {
    "model_type": "RandomForestRegressor",
    "verification_status": "s3_loaded",
    "s3_bucket": "your-s3-bucket"
  },
  "prediction_timestamp": "2024-01-15T10:30:00.000Z",
  "processing_time_ms": 45.2
}
```

### Batch Prediction Response
```json
{
  "predictions": [
    {
      "index": 0,
      "prediction": 123,
      "confidence": 0.85,
      "status": "success"
    },
    {
      "index": 1,
      "prediction": 89,
      "confidence": 0.85,
      "status": "success"
    }
  ],
  "total_processing_time_ms": 120.5,
  "model_info": {
    "model_type": "RandomForestRegressor",
    "verification_status": "s3_loaded",
    "s3_bucket": "your-s3-bucket",
    "total_requests": 2,
    "successful_predictions": 2
  }
}
```

### S3 Model Completeness Response
```json
{
  "status": "success",
  "completeness_info": {
    "model_exists": true,
    "scaler_exists": true,
    "can_load": true
  },
  "can_load_model": true,
  "model_exists": true,
  "scaler_exists": true
}
``` 