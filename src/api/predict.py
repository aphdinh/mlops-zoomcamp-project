import mlflow
import pandas as pd
import sys

def load_model_and_predict(input_file, output_file, model_name="seoul_bike_best_model"):
    model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")
    
    data = pd.read_csv(input_file)
    predictions = model.predict(data)
    
    results = data.copy()
    results['predicted_bike_count'] = predictions
    results.to_csv(output_file, index=False)
    
if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    load_model_and_predict(input_file, output_file)