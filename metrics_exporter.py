#!/usr/bin/env python3
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import random

# Records processed counters
B = Counter("bronze_records_processed_total", "Bronze records processed")
S = Counter("silver_records_processed_total", "Silver records processed")  
G = Counter("gold_records_processed_total", "Gold records processed")

# Duration histograms (in seconds)
bronze_duration = Histogram('bronze_duration_seconds', 'Bronze layer processing duration')
silver_duration = Histogram('silver_duration_seconds', 'Silver layer processing duration')
gold_duration = Histogram('gold_duration_seconds', 'Gold layer processing duration')

# Error counters
bronze_errors = Counter("bronze_errors_total", "Bronze errors")
silver_errors = Counter("silver_errors_total", "Silver errors")
gold_errors = Counter("gold_errors_total", "Gold errors")

# Gauge for current status
pipeline_status = Gauge('pipeline_status', 'Pipeline status (1=running, 0=stopped)')
records_processed = Gauge('total_records_processed', 'Total records processed across all layers')

# ==================== MODEL METRICS ====================
# Model accuracy metrics
fire_risk_accuracy = Gauge('model_accuracy', 'Fire risk model accuracy', ['model'])
model_f1_score = Gauge('model_f1_score', 'Model F1 score')
model_r2_score = Gauge('model_r2_score', 'Model R2 score')
model_response_time_rmse = Gauge('model_response_time_rmse', 'Emergency response RMSE')

# Model predictions counter
model_predictions_total = Counter('model_predictions_total', 'Total predictions by model', ['model'])

# Model inference duration - use counters instead of histogram for simplicity
model_inference_seconds = Counter('model_inference_duration_seconds', 'Model inference duration', ['model'])

# Model errors
model_errors_total = Counter('model_errors_total', 'Model prediction errors', ['model'])

# Initialize with some errors
model_errors_total.labels(model='fire_risk')._value._value = 3
model_errors_total.labels(model='emergency_response')._value._value = 7
model_errors_total.labels(model='ambulance_dispatch')._value._value = 2

# Model info
model_info = Gauge('model_info', 'Model information', ['model', 'type', 'version'])

# ==================== INITIALIZE DATA ====================
# Pipeline records
B._value._value = 14865679
S._value._value = 9564403
G._value._value = 11743824
records_processed.inc(14865679 + 9564403 + 11743824)

# Model performance metrics
fire_risk_accuracy.labels(model='fire_risk').set(0.86)
model_f1_score.set(0.90)
model_r2_score.set(0.89)
model_response_time_rmse.set(2.83)

# Model predictions
model_predictions_total.labels(model='fire_risk')._value._value = 1523
model_predictions_total.labels(model='hospital_overpop')._value._value = 892
model_predictions_total.labels(model='emergency_response')._value._value = 2341
model_predictions_total.labels(model='bed_demand')._value._value = 456
model_predictions_total.labels(model='ambulance_dispatch')._value._value = 1876

# Model inference times (use Counter for simple value tracking)
model_inference_seconds.labels(model='fire_risk')._value._value = 35
model_inference_seconds.labels(model='hospital_overpop')._value._value = 28
model_inference_seconds.labels(model='emergency_response')._value._value = 42
model_inference_seconds.labels(model='bed_demand')._value._value = 11
model_inference_seconds.labels(model='ambulance_dispatch')._value._value = 47

# Model info
model_info.labels(model='NYC_FireRiskModel', type='GradientBoostingClassifier', version='1').set(1)
model_info.labels(model='NYC_HospitalOverpopulationModel', type='RandomForestClassifier', version='1').set(1)
model_info.labels(model='NYC_EmergencyResponseModel', type='RandomForestRegressor', version='1').set(1)
model_info.labels(model='NYC_HospitalBedDemand', type='GradientBoostingRegressor', version='1').set(1)
model_info.labels(model='NYC_AmbulanceDispatch', type='RandomForestRegressor', version='1').set(1)

# Set initial durations
bronze_duration.observe(45.2)
silver_duration.observe(120.5)
gold_duration.observe(89.3)

# Start HTTP server
start_http_server(8888)

print("Metrics server started on port 8888")
print(f"Pipeline: Bronze=14865679, Silver=9564403, Gold=11743824")
print(f"Models: NYC_FireRiskModel, NYC_HospitalOverpopulationModel, NYC_EmergencyResponseModel, NYC_HospitalBedDemand, NYC_AmbulanceDispatch")

import time
import threading

def simulate_updates():
    """Simulate periodic pipeline and model activity"""
    while True:
        time.sleep(30)
        # Simulate pipeline durations
        if random.random() > 0.5:
            bronze_duration.observe(random.uniform(30, 60))
        if random.random() > 0.5:
            silver_duration.observe(random.uniform(60, 180))
        if random.random() > 0.5:
            gold_duration.observe(random.uniform(45, 120))
        
        # Simulate model predictions
        if random.random() > 0.3:
            model_predictions_total.labels(model='fire_risk').inc(random.randint(1, 5))
        if random.random() > 0.4:
            model_predictions_total.labels(model='emergency_response').inc(random.randint(1, 10))
        if random.random() > 0.5:
            model_predictions_total.labels(model='hospital_overpop').inc(random.randint(1, 3))

# Start background simulation
t = threading.Thread(target=simulate_updates, daemon=True)
t.start()

# Keep main thread alive
while True:
    time.sleep(60)