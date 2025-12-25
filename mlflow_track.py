# Model Tracking with mlflow
# Model Tracking with MLflow
import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    classification_report,
)

from config import (
    DATA_PATH,
    EXPERIMENT_NAME,
    TARGET_COL,
    RANDOM_STATE,
    TEST_SIZE,
    REGISTERED_MODEL_NAME,
    MODEL_PARAMS,
    RUN_NAME,
)

# -------------------------
# Ensure artifacts folder exists
# -------------------------
os.makedirs("models", exist_ok=True)

# -------------------------
# Load data
# -------------------------
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# -------------------------
# Train model with config params
# -------------------------
model = RandomForestClassifier(**MODEL_PARAMS)
model.fit(X_train, y_train)

# -------------------------
# Predictions
# -------------------------
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
y_test_prob = model.predict_proba(X_test)[:, 1]

# -------------------------
# Metrics
# -------------------------
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
roc_auc = roc_auc_score(y_test, y_test_prob)
logloss = log_loss(y_test, y_test_prob)

# Cross-validation on TRAIN only
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

# -------------------------
# MLflow logging
# -------------------------
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name=RUN_NAME):

    # Log model params
    mlflow.log_params(MODEL_PARAMS)

    # Log metrics
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("log_loss", logloss)
    mlflow.log_metric("cv_mean_accuracy", cv_scores.mean())
    mlflow.log_metric("cv_std_accuracy", cv_scores.std())
    mlflow.log_metric("generalization_gap", train_acc - test_acc)

    # Save artifacts
    cm = confusion_matrix(y_test, y_test_pred)
    report = classification_report(y_test, y_test_pred)

    with open("models/confusion_matrix.txt", "w") as f:
        f.write(str(cm))

    with open("models/classification_report.txt", "w") as f:
        f.write(report)

    mlflow.log_artifact("models/confusion_matrix.txt")
    mlflow.log_artifact("models/classification_report.txt")

    # Log model
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name=REGISTERED_MODEL_NAME,
    )

print("MLflow tracking completed successfully")
