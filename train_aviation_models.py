#!/usr/bin/env python3
"""
Aviation Model Training with MLflow Autologging.
Trains Random Forest and Gradient Boosting models for flight delay prediction
with full MLflow tracking of features, hyperparameters, and metrics.
"""

import os
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Initialize MLflow
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment("Aviation_Flight_Delay_Prediction")

# Enable MLflow Autologging for sklearn models
mlflow.sklearn.autolog()

def create_synthetic_training_data(n_samples=1000):
    """Create synthetic aviation data for model training."""
    np.random.seed(42)
    
    data = {
        'departure_delay': np.random.exponential(15, n_samples),
        'arrival_delay': np.random.exponential(20, n_samples),
        'distance': np.random.uniform(100, 3000, n_samples),
        'air_time': np.random.uniform(30, 400, n_samples),
        'weather_delay': np.random.exponential(10, n_samples),
        'airport_congestion': np.random.uniform(0, 1, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'month': np.random.randint(1, 13, n_samples),
        'hour_of_day': np.random.randint(0, 24, n_samples),
        'carrier_delay': np.random.exponential(5, n_samples),
        'nas_delay': np.random.exponential(8, n_samples),
        'security_delay': np.random.exponential(2, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable: total_delay_minutes
    df['total_delay_minutes'] = (
        df['departure_delay'] * 0.3 +
        df['arrival_delay'] * 0.4 +
        df['weather_delay'] * 0.2 +
        df['carrier_delay'] * 0.1 +
        np.random.normal(0, 5, n_samples)
    )
    
    # Add some noise and ensure positive values
    df['total_delay_minutes'] = np.maximum(0, df['total_delay_minutes'] + np.random.normal(0, 3, n_samples))
    
    return df

def train_random_forest(X_train, y_train, X_test, y_test, run_name):
    """Train Random Forest model with MLflow autologging."""
    with mlflow.start_run(run_name=f"RF_{run_name}"):
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Log custom parameters
        mlflow.log_params(params)
        mlflow.log_param("model_type", "RandomForestRegressor")
        
        # Train
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Log metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n📊 Random Forest Feature Importance:")
        for _, row in feature_importance.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Log feature importance as artifact
        feature_importance.to_csv('/tmp/rf_feature_importance.csv', index=False)
        mlflow.log_artifact('/tmp/rf_feature_importance.csv')
        
        return model, rmse

def train_gradient_boosting(X_train, y_train, X_test, y_test, run_name):
    """Train Gradient Boosting model with MLflow autologging."""
    with mlflow.start_run(run_name=f"GB_{run_name}"):
        params = {
            'n_estimators': 150,
            'max_depth': 5,
            'learning_rate': 0.1,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        
        mlflow.log_params(params)
        mlflow.log_param("model_type", "GradientBoostingRegressor")
        
        # Train
        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        
        print(f"\n📊 Gradient Boosting Feature Importance:")
        for name, imp in sorted(zip(X_train.columns, model.feature_importances_), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {name}: {imp:.4f}")
        
        return model, rmse

def main():
    print("=" * 60)
    print("Aviation Model Training with MLflow Autologging")
    print("=" * 60)
    
    print("\n📊 Creating synthetic training data...")
    df = create_synthetic_training_data(n_samples=1000)
    print(f"Generated {len(df)} samples with {len(df.columns)} features")
    
    # Prepare features and target
    feature_cols = ['departure_delay', 'arrival_delay', 'distance', 'air_time',
                    'weather_delay', 'airport_congestion', 'day_of_week', 'month',
                    'hour_of_day', 'carrier_delay', 'nas_delay', 'security_delay']
    
    X = df[feature_cols]
    y = df['total_delay_minutes']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\n📊 Data split:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Train models
    print("\n🔄 Training Random Forest...")
    rf_model, rf_rmse = train_random_forest(X_train, y_train, X_test, y_test, "aviation_delays")
    
    print("\n🔄 Training Gradient Boosting...")
    gb_model, gb_rmse = train_gradient_boosting(X_train, y_train, X_test, y_test, "aviation_delays")
    
    # Compare models
    print("\n" + "=" * 60)
    print("📊 Model Comparison:")
    print(f"  Random Forest RMSE:      {rf_rmse:.2f}")
    print(f"  Gradient Boosting RMSE:  {gb_rmse:.2f}")
    
    best_model = "Random Forest" if rf_rmse < gb_rmse else "Gradient Boosting"
    best_rmse = min(rf_rmse, gb_rmse)
    
    with mlflow.start_run(run_name="model_comparison_summary"):
        mlflow.log_param("best_model", best_model)
        mlflow.log_metric("best_rmse", best_rmse)
        mlflow.log_metric("rf_rmse", rf_rmse)
        mlflow.log_metric("gb_rmse", gb_rmse)
    
    print(f"\n🏆 Best Model: {best_model} (RMSE: {best_rmse:.2f})")
    print("\n✅ Model training complete!")
    print("   View in MLflow UI under 'Experiments' tab")
    print("   - Hyperparameters captured")
    print("   - Feature importance logged")
    print("   - Metrics tracked (MSE, RMSE, R²)")

if __name__ == "__main__":
    main()