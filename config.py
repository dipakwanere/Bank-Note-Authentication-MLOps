# =========================
# Project Configuration
# =========================

# Paths
DATA_PATH = "data/BankNote_Authentication.csv"
MODEL_PATH = "models/classifier_model.pkl"

# MLflow
EXPERIMENT_NAME = "banknote_rf_evaluation"
REGISTERED_MODEL_NAME = "BankNoteRandomForest"

# Dataset
TARGET_COL = "class"
RANDOM_STATE = 42
TEST_SIZE = 0.2


# model
MODEL_TYPE = "random_forest"
MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "min_samples_leaf": 10,
    "min_samples_split": 20,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
}
RUN_NAME = "rf_pkl_evaluation"
