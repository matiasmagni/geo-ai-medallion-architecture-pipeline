# src/train_aviation_models.py
import mlflow
from mlflow.models import infer_signature
from mlflow.types.schema import Schema, ColSpec
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
# Ensure MLflow tracking is configured
# mlflow.set_tracking_uri("http://localhost:5000") # Uncomment if using a remote tracking server

# Placeholder for model training logic
# Assume data is loaded from a CSV or Parquet file for demonstration
# IMPORTANT: Replace with your actual data path.
DATA_PATH = "/path/to/your/aviation_data.csv" # Placeholder

# --- Enable MLflow Autologging ---
# This must be called *before* importing or training ML models.
# It automatically logs parameters, metrics, artifacts, and model signatures.
# For scikit-learn, it also logs feature importances and other model-specific details.
mlflow.sklearn.autolog()

# --- Data Loading and Preprocessing ---
def load_and_preprocess_data(file_path: str) -> tuple:
    """Loads and preprocesses data for model training."""
    logger.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error(f"Data file not found at {file_path}. Please check the path.")
        # Create dummy data if file not found, so autologging can be demonstrated
        logger.warning("Creating dummy data for demonstration purposes.")
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            'target': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1] # Binary classification target
        })

    # Basic preprocessing: Assume 'target' is the label, rest are features.
    # In a real scenario, this would involve more complex feature engineering,
    # handling missing values, categorical encoding, etc.
    target_column = 'target'
    if target_column not in df.columns:
        logger.error(f"Target column '{target_column}' not found in data. Please specify correctly.")
        # Fallback to the last column as target if 'target' not found (not recommended)
        target_column = df.columns[-1]
        logger.warning(f"Using '{target_column}' as target column.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split data into training and testing sets
    # Ensure stratify is used for classification if target has multiple classes
    stratify_param = y if len(y.unique()) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_param
    )

    logger.info(f"Data loaded and split. Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    # Infer MLflow signature
    # This is automatically handled by mlflow.sklearn.autolog() but good to know
    # input_schema = Schema([ColSpec("double", col_name) for col_name in X_train.columns])
    # output_schema = Schema([ColSpec("long", target_column)]) # Assuming integer target
    # signature = infer_signature(X_train, y_train)

    return X_train, X_test, y_train, y_test

# --- Model Training Functions ---

def train_random_forest(X_train, X_test, y_train, y_test):
    """Trains a RandomForestClassifier model and logs it with MLflow."""
    logger.info("Training RandomForestClassifier...")
    # Autologging will capture hyperparameters, metrics, artifacts, and model
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train)

    # Log custom metrics if autologging doesn't capture them all or if specific ones are needed
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    logger.info(f"Random Forest Metrics: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

    # Autologging will save the model and potentially feature importance plots
    # If you want to explicitly log artifacts or metrics not covered by autologging:
    # mlflow.log_metric("rf_accuracy", accuracy)
    # mlflow.log_artifact("path/to/feature_importance_plot.png")

    logger.info("RandomForestClassifier training complete.")
    return rf_model

def train_gradient_boosting(X_train, X_test, y_train, y_test):
    """Trains a GradientBoostingClassifier model and logs it with MLflow."""
    logger.info("Training GradientBoostingClassifier...")
    # Autologging will capture hyperparameters, metrics, artifacts, and model
    gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    gb_model.fit(X_train, y_train)

    y_pred = gb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    logger.info(f"Gradient Boosting Metrics: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

    logger.info("GradientBoostingClassifier training complete.")
    return gb_model

def train_all_models(X_train, X_test, y_train, y_test):
    """Trains RandomForest and GradientBoosting models and returns them as a dict."""
    rf = train_random_forest(X_train, X_test, y_train, y_test)
    gb = train_gradient_boosting(X_train, X_test, y_train, y_test)
    return {"random_forest": rf, "gradient_boosting": gb}


# --- Main Execution ---
if __name__ == "__main__":
    logger.info("Starting MLflow autologging for traditional model training.")

    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(DATA_PATH)

    # Start an MLflow run to group related experiments
    # MLflow autologging will automatically create child runs if not in an active run
    with mlflow.start_run(run_name="Traditional Aviation Models Training"):
        logger.info("MLflow run started for Traditional Aviation Models.")

        # Train models
        rf_model = train_random_forest(X_train, X_test, y_train, y_test)
        gb_model = train_gradient_boosting(X_train, X_test, y_train, y_test)

        logger.info("All models trained. MLflow autologging captured details.")
        logger.info("Check the MLflow UI for logged parameters, metrics, artifacts (models, plots), and signatures.")

    logger.info("Script execution finished.")
