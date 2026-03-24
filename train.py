import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Assignment5_Mostafa")

# Use mode to create failed and successful runs in a more realistic way
MODEL_MODE = os.getenv("MODEL_MODE", "good").lower()

# Load dataset
X, y = load_iris(return_X_y=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

with mlflow.start_run() as run:
    run_id = run.info.run_id

    if MODEL_MODE == "bad":
        model = DummyClassifier(strategy="most_frequent")
        model_type = "DummyClassifier"
    else:
        model = LogisticRegression(max_iter=200)
        model_type = "LogisticRegression"

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_param("model_mode", MODEL_MODE)
    mlflow.log_param("model_type", model_type)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")

    print(f"Training finished. Accuracy = {accuracy}")
    print(f"Run ID = {run_id}")

    with open("model_info.txt", "w") as f:
        f.write(run_id)