#!/usr/bin/env python3
"""
===============================================================================
NYC AVIATION ML PIPELINE - 5 MODEL TRAINING
===============================================================================
File: src/train_aviation_models.py

Purpose:
    Train and register 5 ML models for NYC aviation analytics:
    1. NYC_FlightDelayPredictor - Predicts flight delay in minutes
    2. NYC_WeatherImpactModel - Predicts weather severity (0-100)
    3. NYC_AirTrafficVolume - Predicts active flights per hour
    4. NYC_RunwayCongestionModel - Predicts runway congestion
    5. NYC_FuelConsumptionEstimator - Estimates extra fuel consumption

Output:
    flight_predictions_gold.parquet - 5 predictions per flight route

Author: GeoAI Principal MLOps Engineer
===============================================================================
"""

import logging
import os
import random
import json
from datetime import datetime, timedelta
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    GradientBoostingClassifier
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score
)

# =============================================================================
# CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# MLflow Configuration
MLFLOW_TRACKING_URI = "http://localhost:5001"
EXPERIMENT_NAME = "nyc_aviation_ml"

# NYC Airports with coordinates
NYC_AIRPORTS = {
    "JFK": {"lat": 40.6413, "lon": -73.7781, "name": "John F. Kennedy International"},
    "LGA": {"lat": 40.7769, "lon": -73.8740, "name": "LaGuardia Airport"},
    "EWR": {"lat": 40.6895, "lon": -74.1745, "name": "Newark Liberty International"}
}

# Bounding box for NYC flights
NYC_BOUNDS = {
    "lat_min": 40.4, "lat_max": 41.0,
    "lon_min": -74.5, "lon_max": -73.5
}


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_weather_conditions(n_samples: int) -> pd.DataFrame:
    """Generate synthetic weather data for NYC area."""
    weather_data = []
    
    conditions = ["Clear", "Partly Cloudy", "Cloudy", "Light Rain", "Heavy Rain", "Fog", "Thunderstorm"]
    wind_directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    
    for i in range(n_samples):
        hour = random.randint(0, 23)
        is_night = hour >= 20 or hour < 6
        
        base_temp = random.uniform(20, 85)
        humidity = random.uniform(30, 95)
        wind_speed = random.uniform(0, 35)
        visibility = random.uniform(0.5, 10)
        ceiling = random.uniform(200, 15000)
        precipitation = random.uniform(0, 0.8)
        condition = random.choice(conditions)
        
        weather_data.append({
            "weather_id": f"WEATHER_{i:05d}",
            "timestamp": (datetime.now() - timedelta(hours=random.randint(0, 168))).isoformat(),
            "temperature_f": base_temp,
            "humidity_pct": humidity,
            "wind_speed_mph": wind_speed,
            "wind_direction": random.choice(wind_directions),
            "visibility_miles": visibility,
            "ceiling_ft": ceiling,
            "precipitation_in": precipitation,
            "conditions": condition,
            "is_night": 1 if is_night else 0,
            "hour": hour,
            "is_weekend": 1 if random.randint(0, 6) >= 5 else 0
        })
    
    return pd.DataFrame(weather_data)


def generate_flight_routes(n_flights: int, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Generate synthetic flight routes between NYC airports."""
    flights = []
    airport_codes = list(NYC_AIRPORTS.keys())
    
    for i in range(n_flights):
        origin_code = random.choice(airport_codes)
        dest_code = random.choice([a for a in airport_codes if a != origin_code])
        
        origin = NYC_AIRPORTS[origin_code]
        dest = NYC_AIRPORTS[dest_code]
        
        lat1 = origin["lat"] + random.uniform(-0.3, 0.3)
        lon1 = origin["lon"] + random.uniform(-0.3, 0.3)
        lat2 = dest["lat"] + random.uniform(-0.3, 0.3)
        lon2 = dest["lon"] + random.uniform(-0.3, 0.3)
        
        altitude_ft = random.randint(15000, 40000)
        velocity_mps = random.uniform(150, 300)
        distance_nm = random.uniform(10, 200)
        
        weather_idx = random.randint(0, len(weather_df) - 1)
        weather = weather_df.iloc[weather_idx]
        
        flights.append({
            "flight_id": f"FL{random.randint(10000, 99999)}",
            "callsign": f"{random.choice(['AAL', 'UAL', 'DAL', 'JBU', 'SWA', 'ASA', 'FFT'])}{random.randint(100, 999)}",
            "origin_airport": origin_code,
            "destination_airport": dest_code,
            "start_lat": lat1,
            "start_lon": lon1,
            "end_lat": lat2,
            "end_lon": lon2,
            "altitude_ft": altitude_ft,
            "velocity_mps": velocity_mps,
            "distance_nm": distance_nm,
            "hour": weather["hour"],
            "is_night": weather["is_night"],
            "is_weekend": weather["is_weekend"],
            "weather_id": weather["weather_id"],
            "temperature_f": weather["temperature_f"],
            "humidity_pct": weather["humidity_pct"],
            "wind_speed_mph": weather["wind_speed_mph"],
            "visibility_miles": weather["visibility_miles"],
            "ceiling_ft": weather["ceiling_ft"],
            "precipitation_in": weather["precipitation_in"]
        })
    
    return pd.DataFrame(flights)


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def prepare_features(df: pd.DataFrame) -> tuple:
    """Prepare feature matrix for all models."""
    feature_cols = [
        "temperature_f", "humidity_pct", "wind_speed_mph", "visibility_miles",
        "ceiling_ft", "precipitation_in", "is_night", "is_weekend",
        "distance_nm", "altitude_ft"
    ]
    
    X = df[feature_cols].copy()
    
    # Add engineered features
    X["temp_deviation"] = abs(X["temperature_f"] - 70) / 70
    X["visibility_factor"] = 10 - X["visibility_miles"]
    X["wind_factor"] = X["wind_speed_mph"] / 35
    X["precip_factor"] = X["precipitation_in"] * 10
    X["ceiling_factor"] = 15000 - X["ceiling_ft"]
    X["night_penalty"] = X["is_night"] * 0.1
    X["weekend_factor"] = X["is_weekend"] * 0.05
    
    return X, feature_cols


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_flight_delay_predictor(X_train, y_train, X_test, y_test) -> tuple:
    """
    MODEL 1: NYC_FlightDelayPredictor
    Predicts flight delay in minutes using RandomForestRegressor.
    """
    logger.info("Training Model 1: NYC_FlightDelayPredictor")
    
    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42
    }
    
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred)
    }
    
    logger.info(f"  MAE: {metrics['mae']:.2f} min, RMSE: {metrics['rmse']:.2f}, R2: {metrics['r2']:.3f}")
    
    return model, metrics, params


def train_weather_impact_model(X_train, y_train, X_test, y_test) -> tuple:
    """
    MODEL 2: NYC_WeatherImpactModel
    Predicts weather severity impact (0-100) using GradientBoostingRegressor.
    """
    logger.info("Training Model 2: NYC_WeatherImpactModel")
    
    params = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 6,
        "min_samples_split": 5,
        "random_state": 42
    }
    
    model = GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred)
    }
    
    logger.info(f"  MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}, R2: {metrics['r2']:.3f}")
    
    return model, metrics, params


def train_air_traffic_volume(X_train, y_train, X_test, y_test) -> tuple:
    """
    MODEL 3: NYC_AirTrafficVolume
    Predicts active flights per hour using RandomForestRegressor.
    """
    logger.info("Training Model 3: NYC_AirTrafficVolume")
    
    params = {
        "n_estimators": 80,
        "max_depth": 8,
        "min_samples_split": 4,
        "random_state": 42
    }
    
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred)
    }
    
    logger.info(f"  MAE: {metrics['mae']:.2f} flights, RMSE: {metrics['rmse']:.2f}, R2: {metrics['r2']:.3f}")
    
    return model, metrics, params


def train_runway_congestion_model(X_train, y_train, X_test, y_test) -> tuple:
    """
    MODEL 4: NYC_RunwayCongestionModel
    Predicts congestion probability (0-1) using GradientBoostingClassifier.
    """
    logger.info("Training Model 4: NYC_RunwayCongestionModel")
    
    params = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 5,
        "min_samples_split": 5,
        "random_state": 42
    }
    
    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0)
    }
    
    logger.info(f"  Accuracy: {metrics['accuracy']:.3f}, Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}")
    
    return model, metrics, params


def train_fuel_consumption_estimator(X_train, y_train, X_test, y_test) -> tuple:
    """
    MODEL 5: NYC_FuelConsumptionEstimator
    Estimates extra fuel consumption (gallons) using RandomForestRegressor.
    """
    logger.info("Training Model 5: NYC_FuelConsumptionEstimator")
    
    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 4,
        "random_state": 42
    }
    
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred)
    }
    
    logger.info(f"  MAE: {metrics['mae']:.2f} gal, RMSE: {metrics['rmse']:.2f}, R2: {metrics['r2']:.3f}")
    
    return model, metrics, params


# =============================================================================
# LABEL GENERATION (Synthetic Ground Truth)
# =============================================================================

def generate_delay_labels(X: pd.DataFrame) -> pd.Series:
    """Generate synthetic delay labels based on weather conditions."""
    delay_score = (
        (X["wind_speed_mph"] / 35) * 30 +
        ((10 - X["visibility_miles"]) / 10) * 25 +
        (X["precipitation_in"] * 10) * 20 +
        ((15000 - X["ceiling_ft"]) / 15000) * 15 +
        X["night_penalty"] * 5 +
        X["weekend_factor"] * 5
    )
    return delay_score.clip(0, 120).astype(int)


def generate_weather_impact_labels(X: pd.DataFrame) -> pd.Series:
    """Generate synthetic weather impact severity labels (0-100)."""
    impact = (
        (abs(X["temperature_f"] - 70) / 70) * 20 +
        (X["humidity_pct"] / 100) * 20 +
        (X["wind_speed_mph"] / 35) * 25 +
        ((10 - X["visibility_miles"]) / 10) * 20 +
        (X["precipitation_in"] * 10) * 15
    )
    return impact.clip(0, 100).astype(int)


def generate_traffic_volume_labels(X: pd.DataFrame) -> pd.Series:
    """Generate synthetic traffic volume labels (flights per hour)."""
    base_volume = random.randint(30, 80)
    volume = base_volume + (
        X["is_weekend"] * random.randint(-15, 5) +
        X["is_night"] * random.randint(-20, -5) +
        X["temp_deviation"] * 10
    )
    return volume.clip(10, 120).astype(int)


def generate_congestion_labels(X: pd.DataFrame) -> pd.Series:
    """Generate synthetic congestion labels (0 or 1)."""
    risk_score = (
        (X["wind_speed_mph"] > 20).astype(int) * 0.25 +
        (X["visibility_miles"] < 3).astype(int) * 0.25 +
        (X["precipitation_in"] > 0.2).astype(int) * 0.20 +
        (X["ceiling_ft"] < 1000).astype(int) * 0.15 +
        X["is_night"] * 0.15
    )
    return (risk_score > 0.3).astype(int)


def generate_fuel_labels(X: pd.DataFrame) -> pd.Series:
    """Generate synthetic fuel consumption labels (extra gallons)."""
    base_fuel = X["distance_nm"] * 0.02
    extra = (
        base_fuel * 1.5 * (X["wind_speed_mph"] / 35) +
        base_fuel * 1.2 * ((10 - X["visibility_miles"]) / 10) +
        base_fuel * 1.3 * (X["precipitation_in"] * 10)
    )
    return extra.clip(0, 500).astype(int)


# =============================================================================
# MLFLOW LOGGING
# =============================================================================

def setup_mlflow():
    """Initialize MLflow tracking."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f"MLflow tracking: {MLFLOW_TRACKING_URI}")
    logger.info(f"Experiment: {EXPERIMENT_NAME}")


def log_model_to_mlflow(model, model_name: str, metrics: dict, params: dict, feature_importance: np.ndarray = None) -> str:
    """Log model and metrics to MLflow registry."""
    run = mlflow.start_run(run_name=model_name)
    run_id = run.info.run_id
    
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    
    if feature_importance is not None:
        for i, importance in enumerate(feature_importance[:10]):
            mlflow.log_metric(f"feature_importance_rank_{i+1}", importance)
    
    mlflow.sklearn.log_model(model, model_name, registered_model_name=model_name)
    
    mlflow.end_run()
    
    logger.info(f"  Registered: {model_name} v1")
    
    return run_id


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def train_all_models(n_flights: int = 1000) -> tuple:
    """
    Execute full ML training pipeline for all 5 models.
    
    Returns:
        tuple: (models_dict, predictions_df)
    """
    logger.info("=" * 70)
    logger.info("NYC AVIATION ML PIPELINE - 5 MODEL TRAINING")
    logger.info("=" * 70)
    
    # Setup MLflow
    setup_mlflow()
    
    # Generate data
    logger.info("\n[1/5] Generating synthetic NYC flight and weather data...")
    weather_df = generate_weather_conditions(n_flights)
    flights_df = generate_flight_routes(n_flights, weather_df)
    logger.info(f"  Generated {len(flights_df)} flight routes")
    
    # Prepare features
    logger.info("\n[2/5] Preparing features...")
    X, feature_cols = prepare_features(flights_df)
    logger.info(f"  Features: {len(feature_cols) + 6} columns")
    
    # Generate target labels for each model
    logger.info("\n[3/5] Generating synthetic labels for all 5 models...")
    
    y_delay = generate_delay_labels(X)
    y_weather = generate_weather_impact_labels(X)
    y_traffic = generate_traffic_volume_labels(X)
    y_congestion = generate_congestion_labels(X)
    y_fuel = generate_fuel_labels(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_delay, test_size=0.2, random_state=42
    )
    
    # Train models
    logger.info("\n[4/5] Training ML models...")
    
    models = {}
    all_metrics = {}
    
    # Model 1: Flight Delay Predictor
    model1, metrics1, params1 = train_flight_delay_predictor(X_train, y_delay.iloc[X_train.index], X_test, y_delay.iloc[X_test.index])
    models["NYC_FlightDelayPredictor"] = model1
    all_metrics["NYC_FlightDelayPredictor"] = metrics1
    log_model_to_mlflow(model1, "NYC_FlightDelayPredictor", metrics1, params1, model1.feature_importances_)
    
    # Model 2: Weather Impact Model
    model2, metrics2, params2 = train_weather_impact_model(X_train, y_weather.iloc[X_train.index], X_test, y_weather.iloc[X_test.index])
    models["NYC_WeatherImpactModel"] = model2
    all_metrics["NYC_WeatherImpactModel"] = metrics2
    log_model_to_mlflow(model2, "NYC_WeatherImpactModel", metrics2, params2, model2.feature_importances_)
    
    # Model 3: Air Traffic Volume
    model3, metrics3, params3 = train_air_traffic_volume(X_train, y_traffic.iloc[X_train.index], X_test, y_traffic.iloc[X_test.index])
    models["NYC_AirTrafficVolume"] = model3
    all_metrics["NYC_AirTrafficVolume"] = metrics3
    log_model_to_mlflow(model3, "NYC_AirTrafficVolume", metrics3, params3, model3.feature_importances_)
    
    # Model 4: Runway Congestion Model
    model4, metrics4, params4 = train_runway_congestion_model(X_train, y_congestion.iloc[X_train.index], X_test, y_congestion.iloc[X_test.index])
    models["NYC_RunwayCongestionModel"] = model4
    all_metrics["NYC_RunwayCongestionModel"] = metrics4
    log_model_to_mlflow(model4, "NYC_RunwayCongestionModel", metrics4, params4, model4.feature_importances_)
    
    # Model 5: Fuel Consumption Estimator
    model5, metrics5, params5 = train_fuel_consumption_estimator(X_train, y_fuel.iloc[X_train.index], X_test, y_fuel.iloc[X_test.index])
    models["NYC_FuelConsumptionEstimator"] = model5
    all_metrics["NYC_FuelConsumptionEstimator"] = metrics5
    log_model_to_mlflow(model5, "NYC_FuelConsumptionEstimator", metrics5, params5, model5.feature_importances_)
    
    # Generate predictions
    logger.info("\n[5/5] Generating predictions for all flight routes...")
    
    predictions_df = flights_df.copy()
    predictions_df["model1_flight_delay_min"] = models["NYC_FlightDelayPredictor"].predict(X)
    predictions_df["model2_weather_impact_score"] = models["NYC_WeatherImpactModel"].predict(X)
    predictions_df["model3_air_traffic_volume"] = models["NYC_AirTrafficVolume"].predict(X)
    predictions_df["model4_runway_congestion_prob"] = models["NYC_RunwayCongestionModel"].predict_proba(X)[:, 1]
    predictions_df["model5_fuel_consumption_gal"] = models["NYC_FuelConsumptionEstimator"].predict(X)
    
    # Round predictions
    predictions_df["model1_flight_delay_min"] = predictions_df["model1_flight_delay_min"].round(2)
    predictions_df["model2_weather_impact_score"] = predictions_df["model2_weather_impact_score"].round(2)
    predictions_df["model3_air_traffic_volume"] = predictions_df["model3_air_traffic_volume"].round(2)
    predictions_df["model4_runway_congestion_prob"] = predictions_df["model4_runway_congestion_prob"].round(4)
    predictions_df["model5_fuel_consumption_gal"] = predictions_df["model5_fuel_consumption_gal"].round(2)
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info("\nModel Summary:")
    logger.info("  1. NYC_FlightDelayPredictor      - MAE: {:.2f} min".format(all_metrics["NYC_FlightDelayPredictor"]["mae"]))
    logger.info("  2. NYC_WeatherImpactModel        - MAE: {:.2f}".format(all_metrics["NYC_WeatherImpactModel"]["mae"]))
    logger.info("  3. NYC_AirTrafficVolume          - MAE: {:.2f} flights/hr".format(all_metrics["NYC_AirTrafficVolume"]["mae"]))
    logger.info("  4. NYC_RunwayCongestionModel     - Accuracy: {:.3f}".format(all_metrics["NYC_RunwayCongestionModel"]["accuracy"]))
    logger.info("  5. NYC_FuelConsumptionEstimator   - MAE: {:.2f} gal".format(all_metrics["NYC_FuelConsumptionEstimator"]["mae"]))
    
    return models, predictions_df, all_metrics


def save_predictions(predictions_df: pd.DataFrame, output_path: str = "geo-ai-mlflow/flight_predictions_gold.parquet"):
    """Save predictions to Parquet format for Blender visualization."""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    columns = [
        "flight_id", "callsign", "origin_airport", "destination_airport",
        "start_lat", "start_lon", "end_lat", "end_lon",
        "model1_flight_delay_min", "model2_weather_impact_score",
        "model3_air_traffic_volume", "model4_runway_congestion_prob",
        "model5_fuel_consumption_gal"
    ]
    
    predictions_df[columns].to_parquet(output_path, index=False)
    logger.info(f"\nSaved predictions to: {output_path}")
    logger.info(f"  Rows: {len(predictions_df)}")
    logger.info(f"  Columns: {len(columns)}")
    
    return output_path


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys
    
    n_flights = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    
    logger.info(f"Training with {n_flights} synthetic flight routes...")
    
    models, predictions_df, metrics = train_all_models(n_flights)
    
    output_path = save_predictions(predictions_df)
    
    logger.info("\n" + "=" * 70)
    logger.info("NEXT STEPS")
    logger.info("=" * 70)
    logger.info("1. View MLflow UI: http://localhost:5001")
    logger.info("2. Run Blender visualization:")
    logger.info(f"   blender --background --python scripts/render_flight_simulation.py")
    logger.info("=" * 70)