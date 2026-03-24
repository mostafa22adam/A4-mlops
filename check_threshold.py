import os
import sys
import mlflow

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(mlflow_tracking_uri)

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

client = mlflow.tracking.MlflowClient()
run = client.get_run(run_id)

accuracy = run.data.metrics.get("accuracy", None)

print(f"Run ID: {run_id}")
print(f"Accuracy: {accuracy}")

if accuracy is None:
    print("No accuracy metric found. Failing pipeline.")
    sys.exit(1)

if accuracy < 0.85:
    print("Accuracy below threshold. Deployment blocked.")
    sys.exit(1)

print("Accuracy passed threshold. Deployment allowed.")
sys.exit(0)