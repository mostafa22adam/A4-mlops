import os
import mlflow

# Tracking URI from GitHub secret or local fallback
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment("Assignment5_Mostafa")

# We control accuracy using an environment variable
accuracy = float(os.getenv("FORCE_ACCURACY", "0.90"))

with mlflow.start_run() as run:
    run_id = run.info.run_id

    mlflow.log_param("model_type", "mock_classifier")
    mlflow.log_param("epochs", 1)
    mlflow.log_metric("accuracy", accuracy)

    print(f"Training finished. Accuracy = {accuracy}")
    print(f"Run ID = {run_id}")

    with open("model_info.txt", "w") as f:
        f.write(run_id)