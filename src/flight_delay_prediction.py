#!/usr/bin/env python3
"""
================================================================================
FLIGHT DELAY PREDICTION MODEL WITH MLFLOW
================================================================================
File: src/flight_delay_prediction.py

Purpose:
    Predict flight delays based on weather conditions.
    MLflow for experiment tracking, model registry, and deployment.

Author: GeoAI Principal Data Engineer
================================================================================
"""

import logging
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = "http://localhost:5001"
EXPERIMENT_NAME = "flight_delay_prediction"

NYC_BOUNDS = {
    'lat_min': 40.5, 'lat_max': 41.0,
    'lon_min': -74.5, 'lon_max': -73.5
}

AIRLINE_MAP = {
    'AAL': 1, 'UAL': 2, 'DAL': 3, 'JBU': 4, 'SWA': 5,
    'ASA': 6, 'FFT': 7, 'RPA': 8, 'JB': 9, 'NKS': 10
}


def setup_mlflow():
    """Initialize MLflow tracking server."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        mlflow.set_experiment(EXPERIMENT_NAME)
        logger.info(f"MLflow tracking: {MLFLOW_TRACKING_URI}")
        logger.info(f"Experiment: {EXPERIMENT_NAME}")
    except Exception as e:
        logger.warning(f"Could not connect to MLflow server: {e}")
        logger.info("Will save model locally and attempt to log when server is available.")


def generate_sample_data(n_flights: int = 1000) -> tuple:
    """Generate synthetic flight and weather data."""
    import random
    
    flights, weather = [], []
    boroughs = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
    conditions = ['Clear', 'Cloudy', 'Rain', 'Fog', 'Storm']
    
    for _ in range(n_flights):
        ts = datetime.now().isoformat()
        hour = random.randint(0, 23)
        
        flights.append({
            'flight_id': f'FL{random.randint(1000, 9999)}',
            'callsign': f"{random.choice(list(AIRLINE_MAP.keys()))}{random.randint(100, 999)}",
            'latitude': random.uniform(40.5, 41.0),
            'longitude': random.uniform(-74.5, -73.5),
            'altitude_ft': random.randint(5000, 35000),
            'velocity_mps': random.uniform(100, 300),
            'heading': random.uniform(0, 360),
            'timestamp': ts,
            'distance_nm': random.randint(100, 2000),
            'hour': hour,
            'is_night': 1 if hour >= 20 or hour < 6 else 0,
            'is_weekend': 1 if random.randint(0, 6) >= 5 else 0,
        })
    
    for borough in boroughs:
        weather.append({
            'borough': borough,
            'temperature_f': random.uniform(25, 95),
            'humidity_pct': random.uniform(30, 95),
            'wind_speed_mph': random.uniform(0, 35),
            'wind_direction_deg': random.uniform(0, 360),
            'visibility_miles': random.uniform(0.5, 10),
            'ceiling_ft': random.uniform(100, 15000),
            'precipitation_in': random.uniform(0, 0.5),
            'conditions': random.choice(conditions),
        })
    
    return pd.DataFrame(flights), pd.DataFrame(weather)


def prepare_features(flights_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Join flights with weather and compute delay labels."""
    import random
    flights = flights_df.copy()
    weather = weather_df.copy()
    
    n = len(flights)
    
    base_temp = random.uniform(20, 80)
    base_wind = random.uniform(5, 25)
    base_vis = random.uniform(2, 10)
    base_precip = random.uniform(0, 0.3)
    
    features = pd.DataFrame({
        'temperature_f': [base_temp + random.uniform(-10, 10) for _ in range(n)],
        'humidity_pct': [random.uniform(40, 80) for _ in range(n)],
        'wind_speed_mph': [base_wind + random.uniform(-5, 15) for _ in range(n)],
        'wind_direction_deg': [random.uniform(0, 360) for _ in range(n)],
        'visibility_miles': [base_vis + random.uniform(-3, 5) for _ in range(n)],
        'ceiling_ft': [random.uniform(500, 12000) for _ in range(n)],
        'precipitation_in': [max(0, base_precip + random.uniform(-0.2, 0.3)) for _ in range(n)],
        'is_night': flights['is_night'].values,
        'is_weekend': flights['is_weekend'].values,
        'airline_code': flights['callsign'].str[:3].map(lambda x: AIRLINE_MAP.get(x, 0)).fillna(0).values,
        'route_distance_nm': flights['distance_nm'].values,
        'latitude': flights['latitude'].values,
        'longitude': flights['longitude'].values,
        'callsign': flights['callsign'].values,
        'flight_id': flights['flight_id'].values,
    })
    
    features['delay_score'] = (
        (features['wind_speed_mph'] > 20).astype(int) * 0.3 +
        (features['visibility_miles'] < 3).astype(int) * 0.3 +
        (features['precipitation_in'] > 0.1).astype(int) * 0.25 +
        (features['temperature_f'] < 32).astype(int) * 0.15
    )
    features['delay'] = (features['delay_score'] > 0.3).astype(int)
    
    features['delay_probability'] = features['delay_score']
    features['estimated_delay_mins'] = (features['delay_probability'] * 120).astype(int)
    features['delay_severity'] = features['delay_probability'].apply(_get_severity)
    
    return features


def train_model(X_train, y_train, X_test, y_test, params: Dict = None) -> tuple:
    """Train gradient boosting model."""
    if params is None:
        params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
        }
    
    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_test, y_proba),
    }
    
    return model, metrics


def log_run(model, metrics, params, feature_names: list):
    """Log model and metrics to MLflow."""
    try:
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            
            for name, importance in zip(feature_names, model.feature_importances_):
                mlflow.log_metric(f'feature_importance_{name}', importance)
            
            mlflow.sklearn.log_model(model, "model", registered_model_name="FlightDelayModel")
            
            logger.info(f"Logged run: accuracy={metrics['accuracy']:.3f}, auc_roc={metrics['auc_roc']:.3f}")
        
        return mlflow.active_run().info.run_id
    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}")
        logger.info("Saving model locally for later registration...")
        
        model_dir = Path("geo-ai-mlflow/models")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / "flight_delay_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'params': params,
                'metrics': metrics,
                'feature_names': feature_names,
            }, f)
        
        logger.info(f"Model saved to {model_path}")
        return None


def predict_delays(flights_df: pd.DataFrame, weather_df: pd.DataFrame, model_version: int = 1) -> pd.DataFrame:
    """Predict delays using registered MLflow model."""
    features = prepare_features(flights_df, weather_df)
    
    model_uri = f"models:/FlightDelayModel/{model_version}"
    model = mlflow.sklearn.load_model(model_uri)
    
    X = features[features.columns.drop(['delay', 'delay_score', 'latitude', 'longitude', 'callsign', 'flight_id'])]
    
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    
    features['delay_probability'] = probabilities
    features['delay_predicted'] = predictions
    features['estimated_delay_mins'] = (probabilities * 120).astype(int)
    features['delay_severity'] = features['delay_probability'].apply(_get_severity)
    
    output_df = features[['flight_id', 'callsign', 'latitude', 'longitude', 
                          'delay_probability', 'estimated_delay_mins', 'delay_severity']].copy()
    
    return output_df


def _get_severity(prob: float) -> str:
    if prob < 0.15: return 'none'
    elif prob < 0.30: return 'low'
    elif prob < 0.50: return 'medium'
    elif prob < 0.70: return 'high'
    else: return 'severe'


def train_and_register():
    """Full pipeline: train, log, and register model."""
    setup_mlflow()
    
    logger.info("Generating sample data...")
    flights_df, weather_df = generate_sample_data(1000)
    
    logger.info("Preparing features...")
    features_df = prepare_features(flights_df, weather_df)
    
    feature_cols = [
        'temperature_f', 'humidity_pct', 'wind_speed_mph', 'wind_direction_deg',
        'visibility_miles', 'ceiling_ft', 'precipitation_in',
        'is_night', 'is_weekend', 'airline_code', 'route_distance_nm'
    ]
    
    X = features_df[feature_cols]
    y = features_df['delay']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
    }
    
    logger.info("Training model...")
    model, metrics = train_model(X_train, y_train, X_test, y_test, params)
    
    logger.info("Logging to MLflow...")
    run_id = log_run(model, metrics, params, feature_cols)
    
    output_dir = Path("geo-ai-heatmap/public/data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    geojson = {
        "type": "FeatureCollection",
        "features": features_df.apply(lambda row: {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [row['longitude'], row['latitude']]},
            "properties": {
                "flight_id": row['flight_id'],
                "callsign": row['callsign'],
                "delay_probability": round(row['delay_probability'], 3),
                "estimated_delay_mins": int(row['estimated_delay_mins']),
                "delay_severity": row['delay_severity'],
            }
        }, axis=1).tolist()
    }
    
    with open(output_dir / "flight_delays.geojson", "w") as f:
        json.dump(geojson, f, indent=2)
    
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Metrics: {metrics}")
    logger.info(f"Predictions saved to {output_dir / 'flight_delays.geojson'}")
    
    return model, metrics, run_id


def serve_predictions():
    """Load model and generate predictions for API serving."""
    model_uri = "models:/FlightDelayModel/latest"
    model = mlflow.sklearn.load_model(model_uri)
    
    return model


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        logger.info("Starting prediction server mode...")
        model = serve_predictions()
        logger.info("Model loaded. Ready for serving.")
    else:
        train_and_register()