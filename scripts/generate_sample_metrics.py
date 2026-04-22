#!/usr/bin/env python3
"""
Generate sample metrics for GeoAI Pipeline dashboards.

This script simulates metrics that would be produced by the pipeline
when running with OpenTelemetry enabled.

Usage:
    python scripts/generate_sample_metrics.py
"""

import time
import random
import logging
from prometheus_client import Counter, Histogram, Gauge, start_http_server

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BRONZE_DURATION = Histogram(
    'bronze_duration_seconds',
    'Bronze layer processing duration',
    ['operation'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

SILVER_DURATION = Histogram(
    'silver_duration_seconds',
    'Silver layer processing duration',
    ['operation'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

GOLD_DURATION = Histogram(
    'gold_duration_seconds',
    'Gold layer processing duration',
    ['operation'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

BRONZE_RECORDS = Counter(
    'bronze_records_processed_total',
    'Total records processed by Bronze layer',
    ['source', 'status']
)

SILVER_RECORDS = Counter(
    'silver_records_processed_total',
    'Total records processed by Silver layer',
    ['dataset', 'status']
)

GOLD_RECORDS = Counter(
    'gold_records_processed_total',
    'Total records processed by Gold layer',
    ['table', 'status']
)

BRONZE_ERRORS = Counter(
    'bronze_errors_total',
    'Total errors in Bronze layer',
    ['source', 'error_type']
)

SILVER_ERRORS = Counter(
    'silver_errors_total',
    'Total errors in Silver layer',
    ['dataset', 'error_type']
)

GOLD_ERRORS = Counter(
    'gold_errors_total',
    'Total errors in Gold layer',
    ['table', 'error_type']
)

PIPELINE_STATUS = Gauge(
    'pipeline_status',
    'Current pipeline status (1=running, 0=stopped)',
    ['layer']
)

def generate_bronze_metrics():
    """Generate Bronze layer metrics."""
    operations = ['fetch_us_accidents', 'fetch_nyc_311', 'fetch_usgs_earthquakes', 
                  'fetch_osm_infrastructure', 'fetch_us_neighborhoods']
    sources = ['us_accidents', 'nyc_311', 'usgs_earthquakes', 'osm_infrastructure', 'us_neighborhoods']
    
    for op in operations:
        duration = random.uniform(0.5, 15.0)
        BRONZE_DURATION.labels(operation=op).observe(duration)
    
    for source in sources:
        count = random.randint(100, 5000)
        status = random.choice(['success', 'success', 'success', 'partial'])
        BRONZE_RECORDS.labels(source=source, status=status).inc(count)
        
        if random.random() < 0.05:
            BRONZE_ERRORS.labels(source=source, error_type='network').inc(random.randint(1, 3))

def generate_silver_metrics():
    """Generate Silver layer metrics."""
    operations = ['transform_us_accidents', 'transform_usgs_earthquakes', 
                  'transform_osm_infrastructure', 'transform_us_neighborhoods']
    datasets = ['us_accidents', 'usgs_earthquakes', 'osm_infrastructure', 'us_neighborhoods']
    
    for op in operations:
        duration = random.uniform(1.0, 30.0)
        SILVER_DURATION.labels(operation=op).observe(duration)
    
    for dataset in datasets:
        count = random.randint(80, 4000)
        status = random.choice(['success', 'success', 'enriched'])
        SILVER_RECORDS.labels(dataset=dataset, status=status).inc(count)
        
        if random.random() < 0.03:
            SILVER_ERRORS.labels(dataset=dataset, error_type='llm').inc(1)

def generate_gold_metrics():
    """Generate Gold layer metrics."""
    operations = ['create_dim_neighborhoods', 'create_dim_infrastructure', 
                  'create_fact_hazard_events', 'spatial_join_neighborhoods',
                  'spatial_join_infrastructure', 'aggregate_metrics']
    tables = ['dim_neighborhoods', 'dim_infrastructure', 'fact_hazard_events', 'agg_hazard_metrics']
    
    for op in operations:
        duration = random.uniform(0.5, 10.0)
        GOLD_DURATION.labels(operation=op).observe(duration)
    
    for table in tables:
        count = random.randint(50, 5000)
        status = 'success'
        GOLD_RECORDS.labels(table=table, status=status).inc(count)

def main():
    logger.info("Starting sample metrics generator on port 8888")
    logger.info("Metrics available at http://localhost:8888/metrics")
    
    start_http_server(8888)
    
    PIPELINE_STATUS.labels(layer='bronze').set(1)
    PIPELINE_STATUS.labels(layer='silver').set(1)
    PIPELINE_STATUS.labels(layer='gold').set(1)
    
    logger.info("Generating initial metrics...")
    generate_bronze_metrics()
    generate_silver_metrics()
    generate_gold_metrics()
    
    logger.info("Simulating continuous pipeline runs...")
    try:
        while True:
            time.sleep(10)
            
            if random.random() < 0.7:
                generate_bronze_metrics()
            if random.random() < 0.7:
                generate_silver_metrics()
            if random.random() < 0.7:
                generate_gold_metrics()
                
            logger.info("Metrics updated")
            
    except KeyboardInterrupt:
        PIPELINE_STATUS.labels(layer='bronze').set(0)
        PIPELINE_STATUS.labels(layer='silver').set(0)
        PIPELINE_STATUS.labels(layer='gold').set(0)
        logger.info("Stopped")

if __name__ == "__main__":
    main()