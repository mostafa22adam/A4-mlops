import os
import mlflow

# Use secret in GitHub Actions, and local fallback when testing on your laptop
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Assignment5_Mostafa")

# For screenshots: let us force a failed run or a successful run
accuracy = float(os.getenv("FORCE_ACCURACY", "0.90"))

with mlflow.start_run() as run:
    run_id = run.info.run_id

    mlflow.log_param("model_type", "mock_classifier")
    mlflow.log_param("epochs", 1)
    mlflow.log_metric("accuracy", accuracy)

    print(f"Training finished. Accuracy = {accuracy}")
    print(f"Run ID = {run_id}")

    # Save the REAL current run ID for the next job
    with open("model_info.txt", "w") as f:
        f.write(run_id)