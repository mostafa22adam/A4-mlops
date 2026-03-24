import os
import sys
import mlflow

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

THRESHOLD = 0.85

# Read the run ID saved by validate job
with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

# Fetch the run from MLflow
run = mlflow.get_run(run_id)
accuracy = run.data.metrics.get("accuracy")

if accuracy is None:
    print("Error: accuracy metric not found in MLflow.")
    sys.exit(1)

print(f"Run ID: {run_id}")
print(f"Accuracy: {accuracy}")
print(f"Threshold: {THRESHOLD}")

if accuracy < THRESHOLD:
    print("Model failed threshold.")
    sys.exit(1)

print("Model passed threshold.")