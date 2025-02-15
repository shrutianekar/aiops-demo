import mlflow
import mlflow.sklearn
import os

# Configure MLflow Tracking Server
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def log_model_run(model_name, params, metrics, model_path):
    """Logs model training details to MLflow."""
    with mlflow.start_run():
        mlflow.set_tag("model", model_name)

        # Log parameters
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)

        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log the model artifact
        mlflow.sklearn.log_model(model_path, model_name)

        run_id = mlflow.active_run().info.run_id
        print(f"Run logged in MLflow with Run ID: {run_id}")

if __name__ == "__main__":
    # Example usage
    model_name = "text_embedding_model"
    params = {"learning_rate": 0.001, "batch_size": 32, "epochs": 5}
    metrics = {"accuracy": 0.92, "loss": 0.08}
    model_path = "saved_models/model.pkl"  # Replace with actual model file
    
    log_model_run(model_name, params, metrics, model_path)


 # Run the MLflow tracking server locally:
 # mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow_artifacts --host 0.0.0.0 --port 5000  
 # Test Dataset Versioning
 # python mlops/dataset_versioning.py 
 # Test MLflow Tracking
 # python mlops/mlflow_tracking.py
 # Check your MLflow UI: mlflow ui --host 0.0.0.0 --port 5001