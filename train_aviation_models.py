#!/usr/bin/env python3
"""
Aviation Model Training with MLflow Autologging.
This script trains Random Forest and Gradient Boosting models for flight delay prediction
with full MLflow tracking of features, hyperparameters, and metrics.
"""

import os
import mlflow
import mlflow.pyfunc
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

# Enable MLflow Autologging for traditional ML models
mlflow.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True,
    disable=False
)

# MLflow configuration
mlflow.set_experiment("Aviation_Flight_Delay_Prediction")

def load_and_prepare_data(data_path):
    """
    Load aviation data and prepare features/targets for training.
    """
    # Load data (adjust based on your actual data format)
    df = pd.read_parquet(data_path)
    
    # Feature engineering
    feature_columns = [
        'departure_delay', 'arrival_delay', 'distance',
        'air_time', 'weather_delay', 'airport_congestion',
        'day_of_week', 'month', 'hour_of_day'
    ]
    
    target_column = 'total_delay_minutes'
    
    # Handle missing values
    df = df.dropna(subset=feature_columns + [target_column])
    
    X = df[feature_columns]
    y = df[target_column]
    
    return X, y

def train_random_forest(X_train, y_train, X_test, y_test, run_name):
    """
    Train Random Forest model with MLflow tracking.
    """
    with mlflow.start_run(run_name=f"Random_Forest_{run_name}"):
        # Model parameters
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Log metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        
        # Feature importance (MLflow autolog should capture this)
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Random Forest Feature Importance:")
        print(feature_importance)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        return model, rmse

def train_gradient_boosting(X_train, y_train, X_test, y_test, run_name):
    """
    Train Gradient Boosting model with MLflow tracking.
    """
    with mlflow.start_run(run_name=f"Gradient_Boosting_{run_name}"):
        # Model parameters
        params = {
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.1,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Log metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Gradient Boosting Feature Importance:")
        print(feature_importance)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        return model, rmse

def train_genai_models_with_mlflow(genai_prompts_data, silver_data_path):
    """
    Train models incorporating GenAI features with MLflow tracking.
    This demonstrates how traditional ML can be combined with GenAI capabilities.
    """
    with mlflow.start_run(run_name="GenAI_Enhanced_Models"):
        # Load and prepare data
        X, y = load_and_prepare_data(silver_data_path)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Log dataset info
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])
        
        # Train traditional models with autologging
        rf_model, rf_rmse = train_random_forest(X_train, y_train, X_test, y_test, "RF")
        gb_model, gb_rmse = train_gradient_boosting(X_train, y_train, X_test, y_test, "GB")
        
        # Compare models
        if rf_rmse < gb_rmse:
            best_model = rf_model
            best_rmse = rf_rmse
            model_type = "Random Forest"
        else:
            best_model = gb_model
            best_rmse = gb_rmse
            model_type = "Gradient Boosting"
        
        # Log best model selection
        mlflow.log_param("best_model_type", model_type)
        mlflow.log_metric("best_rmse", best_rmse)
        
        print(f"Best model: {model_type} with RMSE: {best_rmse}")
        
        # Log GenAI prompts used for feature engineering
        if genai_prompts_data:
            mlflow.log_param("genai_prompts_used", json.dumps(list(genai_prompts_data.keys())))
        
        # Register the best model
        mlflow.sklearn.log_model(best_model, "best_model")
        
        return best_model

def main():
    """
    Main execution function with comprehensive error handling.
    """
    try:
        # Configuration
        SILVER_DATA_PATH = "/mnt/geoai/silver/aviation_incidents_enriched"
        GENAI_PROMPTS_PATH = "/path/to/genai_prompts.json"  # Adjust path as needed
        
        # Load GenAI prompts if available
        genai_prompts = {}
        if os.path.exists(GENAI_PROMPTS_PATH):
            import json
            with open(GENAI_PROMPTS_PATH, 'r') as f:
                genai_prompts = json.load(f)
            print(f"Loaded {len(genai_prompts)} GenAI prompts for model training")
        
        # Train models
        print("Starting model training with MLflow autologging...")
        best_model = train_genai_models_with_mlflow(genai_prompts, SILVER_DATA_PATH)
        
        print("Model training completed successfully!")
        print("View training metrics and model details in MLflow UI under 'Experiments' tab")
        
    except Exception as e:
        print(f"Error during model training: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()