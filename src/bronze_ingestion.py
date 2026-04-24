#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
BRONZE LAYER - Raw Data Ingestion
================================================================================
File: src/bronze_ingestion.py

Purpose:
    Ingest raw data from 5 distinct sources into MinIO Bronze bucket:
    s3a://geo-lakehouse/bronze/

Data Sources:
    1. US Accidents               - Kaggle/CSV
    2. NYC 311 Service Requests  - Socrata API / CSV
    3. USGS Earthquake Hazards  - GeoJSON API
    4. OSM Infrastructure      - Overpass API (Hospitals/Fire Stations)
    5. US Neighborhoods        - Census Bureau GeoJSON

Architecture Decision:
    This layer does MINIMAL transformation - just lands data in raw format.
    All cleaning/parsing happens in Silver layer.

Author: GeoAI Principal Data Engineer
Version: 2.0.0 (Shift-Left AI Pattern)
================================================================================
"""

import os
import sys
import json
import time
import logging
import requests
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

# Prometheus metrics for error tracking
try:
    from prometheus_client import Counter
    _bronze_err = Counter('bronze_errors', 'Total errors in Bronze layer')
except:
    _bronze_err = None

def inc_bronze_errors():
    if _bronze_err:
        _bronze_err.inc()
        print(f"[METRICS] Bronze error incremented! Count: {_bronze_err._value._value}")

# OpenTelemetry imports
try:
    from telemetry import setup_telemetry, traced_context, flush_telemetry
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False
    def traced_context(layer, operation):
        class DummyContext:
            def __enter__(self): return self
            def __exit__(self, *a): pass
        return DummyContext()
    def flush_telemetry(): pass
    def setup_telemetry(**kwargs): pass

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


class Config:
    """
    Configuration for Bronze layer ingestion.

    Uses MinIO s3a:// protocol for object storage.
    """

    # MinIO/S3 Configuration (s3a:// for better performance)
    MINIO_ENDPOINT: str = os.getenv("S3_ENDPOINT", "http://localhost:9000")
    MINIO_ACCESS_KEY: str = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    MINIO_SECRET_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123")

    # Bronze bucket path
    BRONZE_BUCKET: str = os.getenv("BRONZE_BUCKET", "geo-lakehouse/bronze")

    # API Endpoints
    USGS_API_URL: str = os.getenv(
        "USGS_API_URL", "https://earthquake.usgs.gov/fdsnws/event/1/query"
    )
    OSM_API_URL: str = os.getenv(
        "OSM_API_URL", "https://overpass-api.de/api/interpreter"
    )
    NYC311_API_URL: str = os.getenv(
        "NYC311_API_URL", "https://data.cityofnewyork.us/resource/erm2-nfe9.json"
    )

    # Local data directory (fallback if MinIO unavailable)
    LOCAL_DATA_DIR: Path = Path(os.getenv("LOCAL_DATA_DIR", "/tmp/geoai/bronze"))

    # Ensure local directory exists
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# MINIO CLIENT (s3a://)
# =============================================================================


def get_minio_client():
    """
    Get MinIO client using s3a:// protocol.

    Returns
    -------
    boto3.client
        Configured S3 client
    """
    try:
        import boto3
        from botocore.config import Config as BotoConfig

        client = boto3.client(
            "s3",
            endpoint_url=Config.MINIO_ENDPOINT,
            aws_access_key_id=Config.MINIO_ACCESS_KEY,
            aws_secret_access_key=Config.MINIO_SECRET_KEY,
            config=BotoConfig(signature_version="s3v4"),
            region_name="us-east-1",
        )
        logger.info(f"MinIO client connected: {Config.MINIO_ENDPOINT}")
        return client
    except ImportError:
        logger.warning("boto3 not installed, using local fallback")
        return None


def upload_to_bronze(local_path: Path, s3_key: str, format: str = "parquet") -> bool:
    """
    Upload local file to Bronze bucket using s3a:// protocol.

    Parameters
    ----------
    local_path : Path
        Local file path
    s3_key : str
        S3 object key (e.g., 'us_accidents/2024/parquet')
    format : str
        File format for content-type

    Returns
    -------
    bool
        Success status
    """
    client = get_minio_client()
    if client is None:
        # Fallback: just verify local file exists
        return local_path.exists()

    try:
        bucket_name = Config.BRONZE_BUCKET.split("/")[0]
        # Add the path prefix (e.g., 'bronze/') to s3_key
        bucket_path = "/".join(Config.BRONZE_BUCKET.split("/")[1:])
        full_s3_key = f"{bucket_path}/{s3_key}" if bucket_path else s3_key

        with open(local_path, "rb") as f:
            content = f.read()

        content_type = {
            "parquet": "application/x-parquet",
            "csv": "text/csv",
            "json": "application/json",
            "geojson": "application/geo+json",
        }.get(format, "application/octet-stream")

        client.put_object(
            Bucket=bucket_name,
            Key=full_s3_key,
            Body=content,
            ContentType=content_type,
        )

        logger.info(f"Uploaded to bronze: {full_s3_key}")
        return True

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return False


# =============================================================================
# SOURCE 1: US ACCIDENTS (CSV from Kaggle)
# =============================================================================


def fetch_us_accidents(output_dir: Optional[Path] = None) -> bool:
    """
    Fetch US Accidents data.

    This is a static dataset - in production, download from Kaggle.
    For now, creates sample structure matching expected schema.

    Expected Schema (Bronze - Raw):
        - incident_id: string
        - incident_description: string (for LLM processing in Silver)
        - latitude: float
        - longitude: float
        - severity: integer (source severity, NOT AI generated)
        - start_time: timestamp
        - end_time: timestamp
        - address: string
        - city: string
        - state: string
        - zipcode: string
        - source: string

    Parameters
    ----------
    output_dir : Path, optional
        Output directory, defaults to Config.LOCAL_DATA_DIR

    Returns
    -------
    bool
        Success status
    """
    output_dir = output_dir or Config.LOCAL_DATA_DIR
    source_dir = output_dir / "us_accidents"
    source_dir.mkdir(parents=True, exist_ok=True)

    # In production, this would download from Kaggle
    # For now, create sample data matching expected schema
    sample_data = {
        "incident_id": ["US-2024-001", "US-2024-002", "US-2024-003"],
        "incident_description": [
            "Multi-vehicle collision on highway. One vehicle collided with another, causing significant damage. Road blocked for 2 hours.",
            "Single vehicle accident with pole. Driver reported speeding before impact. Medical team on scene.",
            "Rear-end collision at traffic light. Minor damage to both vehicles. No injuries reported.",
        ],
        "latitude": [40.7128, 40.7589, 40.6895],
        "longitude": [-74.0060, -73.9851, -74.0445],
        "severity": [3, 4, 2],
        "start_time": [
            "2024-01-15T08:30:00",
            "2024-01-15T14:22:00",
            "2024-01-15T17:45:00",
        ],
        "end_time": [
            "2024-01-15T10:30:00",
            "2024-01-15T15:30:00",
            "2024-01-15T18:00:00",
        ],
        "address": ["123 Main St", "456 Park Ave", "789 Broadway"],
        "city": ["New York", "New York", "New York"],
        "state": ["NY", "NY", "NY"],
        "zipcode": ["10001", "10022", "10003"],
        "source": ["us_accidents", "us_accidents", "us_accidents"],
    }

    df = pd.DataFrame(sample_data)
    output_path = source_dir / "part-00000.parquet"

    # Save as Parquet (better than CSV for schema evolution)
    pq.write_table(pa.Table.from_pandas(df), output_path, compression="snappy")

    logger.info(f"Saved US Accidents to {output_path}")

    # Upload to Bronze
    return upload_to_bronze(output_path, "us_accidents/2024-01.parquet", "parquet")


# =============================================================================
# SOURCE 2: NYC 311 SERVICE REQUESTS (CSV/JSON API)
# =============================================================================


def fetch_nyc_311_requests(output_dir: Optional[Path] = None) -> bool:
    """
    Fetch NYC 311 Service Requests.

    API: https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/d4nm-qzi4

    Expected Schema (Bronze - Raw):
        - request_id: string
        - description: string (for LLM processing)
        - latitude: float
        - longitude: float
        - status: string
        - created_date: timestamp
        - closed_date: timestamp
        - agency: string
        - complaint_type: string
        - location: string
        - city: string
        - borough: string

    Parameters
    ----------
    output_dir : Path, optional

    Returns
    -------
    bool
    """
    output_dir = output_dir or Config.LOCAL_DATA_DIR
    source_dir = output_dir / "nyc_311"
    source_dir.mkdir(parents=True, exist_ok=True)

    # Try API first, fallback to sample data
    try:
        params = {"$limit": 1000, "$order": "created_date DESC", "status": "Closed"}
        response = requests.get(Config.NYC311_API_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Transform to our schema
        records = []
        for row in data:
            records.append(
                {
                    "request_id": row.get("unique_key"),
                    "description": row.get("complaint_type", "")
                    + ": "
                    + row.get("descriptor", ""),
                    "latitude": float(row.get("latitude", 0))
                    if row.get("latitude")
                    else None,
                    "longitude": float(row.get("longitude", 0))
                    if row.get("longitude")
                    else None,
                    "status": row.get("status"),
                    "created_date": row.get("created_date"),
                    "closed_date": row.get("closed_date"),
                    "agency": row.get("agency"),
                    "complaint_type": row.get("complaint_type"),
                    "location": row.get("location"),
                    "city": row.get("city"),
                    "borough": row.get("borough"),
                }
            )

        df = pd.DataFrame(records)

    except Exception as e:
        logger.warning(f"NYC 311 API failed ({e}), using sample data")
        sample_data = {
            "request_id": ["NYC-2024-001", "NYC-2024-002", "NYC-2024-003"],
            "description": [
                "Noise - Residential: Loud music from apartment. Disturbing residents at night.",
                "Street Condition: Pothole on main road causing traffic issues. Needs immediate repair.",
                "Illegal Dumping: Furniture and debris on sidewalk. Block pedestrian path.",
            ],
            "latitude": [40.7580, 40.7489, 40.7280],
            "longitude": [-73.9855, -73.9680, -74.0100],
            "status": ["Closed", "Open", "Closed"],
            "created_date": [
                "2024-01-15T10:00:00",
                "2024-01-15T12:00:00",
                "2024-01-15T14:00:00",
            ],
            "closed_date": ["2024-01-15T11:00:00", None, "2024-01-15T16:00:00"],
            "agency": ["DEP", "DOT", "DOHMH"],
            "complaint_type": [
                "Noise - Residential",
                "Street Condition",
                "Illegal Dumping",
            ],
            "location": ["123 Main St", "456 Park Ave", "789 Broadway"],
            "city": ["New York", "New York", "New York"],
            "borough": ["Manhattan", "Manhattan", "Manhattan"],
        }
        df = pd.DataFrame(sample_data)

    output_path = source_dir / "part-00000.parquet"
    pq.write_table(pa.Table.from_pandas(df), output_path, compression="snappy")

    logger.info(f"Saved NYC 311 to {output_path}")
    return upload_to_bronze(output_path, "nyc_311/2024-01.parquet", "parquet")


# =============================================================================
# SOURCE 3: USGS EARTHQUAKE HAZARDS (GeoJSON API)
# =============================================================================


def fetch_usgs_earthquakes(output_dir: Optional[Path] = None) -> bool:
    """
    Fetch USGS Earthquake data.

    API: https://earthquake.usgs.gov/fdsnws/event/1/query

    Expected Schema (Bronze - Raw):
        - event_id: string
        - description: string (for LLM processing)
        - latitude: float
        - longitude: float
        - magnitude: float
        - place: string
        - time: timestamp
        - depth: float
        - felt: integer
        - cdi: float (modified Mercalli intensity)
        - source: string

    Parameters
    ----------
    output_dir : Path, optional

    Returns
    -------
    bool
    """
    output_dir = output_dir or Config.LOCAL_DATA_DIR
    source_dir = output_dir / "usgs_earthquakes"
    source_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Query past 30 days, M1.0+ earthquakes
        params = {
            "format": "geojson",
            "starttime": (datetime.now() - timedelta(days=30)).isoformat(),
            "minmagnitude": 1.0,
        }
        response = requests.get(Config.USGS_API_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        records = []
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            geom = feature.get("geometry", {})
            coords = geom.get("coordinates", [0, 0, 0])

            records.append(
                {
                    "event_id": feature.get("id"),
                    "description": f"Earthquake magnitude {props.get('mag', 'N/A')} at {props.get('place', 'Unknown')}",
                    "latitude": coords[1],
                    "longitude": coords[0],
                    "magnitude": props.get("mag"),
                    "place": props.get("place"),
                    "time": datetime.fromtimestamp(
                        props.get("time", 0) / 1000
                    ).isoformat()
                    if props.get("time")
                    else None,
                    "depth": coords[2],
                    "felt": props.get("felt", 0),
                    "cdi": props.get("cdi"),
                    "source": ["usgs_earthquakes", "usgs_earthquakes"],
                }
            )

        df = pd.DataFrame(records)

    except Exception as e:
        logger.warning(f"USGS API failed ({e}), using sample data")
        sample_data = {
            "event_id": ["usgs-2024-001", "usgs-2024-002"],
            "description": [
                "Earthquake magnitude 2.5 near San Jose, California. Minor shaking felt.",
                "Earthquake magnitude 4.1 near San Francisco, California. Moderate shaking reported.",
            ],
            "latitude": [37.3382, 37.7749],
            "longitude": [-121.8863, -122.4194],
            "magnitude": [2.5, 4.1],
            "place": ["3km E of San Jose, CA", "8km NW of San Francisco, CA"],
            "time": ["2024-01-15T08:30:00", "2024-01-15T14:22:00"],
            "depth": [10.5, 8.2],
            "felt": [10, 500],
            "cdi": [3.0, 5.5],
            "source": ["usgs_earthquakes", "usgs_earthquakes"],
        }
        df = pd.DataFrame(sample_data)

    output_path = source_dir / "part-00000.parquet"
    pq.write_table(pa.Table.from_pandas(df), output_path, compression="snappy")

    logger.info(f"Saved USGS Earthquakes to {output_path}")
    return upload_to_bronze(output_path, "usgs_earthquakes/2024-01.parquet", "parquet")


# =============================================================================
# SOURCE 4: OSM INFRASTRUCTURE (GeoJSON Overpass API)
# =============================================================================


def fetch_osm_infrastructure(output_dir: Optional[Path] = None) -> bool:
    """
    Fetch OSM Infrastructure (Hospitals, Fire Stations).

    Overpass API query for essential services.

    Expected Schema (Bronze - Raw):
        - facility_id: string
        - name: string
        - facility_type: string (hospital, fire_station)
        - latitude: float
        - longitude: float
        - address: string
        - source: string

    Parameters
    ----------
    output_dir : Path, optional

    Returns
    -------
    bool
    """
    output_dir = output_dir or Config.LOCAL_DATA_DIR
    source_dir = output_dir / "osm_infrastructure"
    source_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Overpass QL query
        query = """
        [out:json][timeout:60];
        (
          node["amenity"="hospital"](40.5, -74.3, 40.9, -73.7);
          node["amenity"="fire_station"](40.5, -74.3, 40.9, -73.7);
          way["amenity"="hospital"](40.5, -74.3, 40.9, -73.7);
          way["amenity"="fire_station"](40.5, -74.3, 40.9, -73.7);
        );
        out center;
        """

        response = requests.get(Config.OSM_API_URL, data={"data": query}, timeout=60)
        response.raise_for_status()
        data = response.json()

        records = []
        for element in data.get("elements", []):
            if element.get("type") == "node":
                lat = element.get("lat")
                lon = element.get("lon")
            else:  # way
                center = element.get("center", {})
                lat = center.get("lat")
                lon = center.get("lon")

            amenity = element.get("tags", {}).get("amenity", "")

            records.append(
                {
                    "facility_id": f"OSM-{element.get('id')}",
                    "name": element.get("tags", {}).get("name", amenity),
                    "facility_type": "hospital"
                    if amenity == "hospital"
                    else "fire_station",
                    "latitude": lat,
                    "longitude": lon,
                    "address": element.get("tags", {}).get("addr:street", "")
                    + " "
                    + element.get("tags", {}).get("addr:city", ""),
                    "source": [
                        "osm_infrastructure",
                        "osm_infrastructure",
                        "osm_infrastructure",
                    ],
                }
            )

        df = pd.DataFrame(records)

    except Exception as e:
        logger.warning(f"OSM API failed ({e}), using sample data")
        sample_data = {
            "facility_id": ["OSM-001", "OSM-002", "OSM-003"],
            "name": ["City Hospital", "Central Fire Station", "Bay Area Hospital"],
            "facility_type": ["hospital", "fire_station", "hospital"],
            "latitude": [40.7128, 40.7589, 40.6895],
            "longitude": [-74.0060, -73.9851, -74.0445],
            "address": ["123 Medical Ave", "456 Fire Blvd", "789 Health St"],
            "source": [
                "osm_infrastructure",
                "osm_infrastructure",
                "osm_infrastructure",
            ],
        }
        df = pd.DataFrame(sample_data)

    output_path = source_dir / "part-00000.parquet"
    pq.write_table(pa.Table.from_pandas(df), output_path, compression="snappy")

    logger.info(f"Saved OSM Infrastructure to {output_path}")
    return upload_to_bronze(
        output_path, "osm_infrastructure/2024-01.parquet", "parquet"
    )


# =============================================================================
# SOURCE 5: US NEIGHBORHOODS (Census/GeoJSON)
# =============================================================================


def fetch_us_neighborhoods(output_dir: Optional[Path] = None) -> bool:
    """
    Fetch US Neighborhood/County boundaries.

    Source: Census Bureau TIGER/Lines or local GeoJSON.
    Used for spatial joins in Gold layer.

    Expected Schema (Bronze - Raw - GeoJSON):
        - neighborhood_id: string
        - name: string
        - county_fips: string
        - state_fips: string
        - geometry: binary (GeoJSON)
        - source: string

    Parameters
    ----------
    output_dir : Path, optional

    Returns
    -------
    bool
    """
    output_dir = output_dir or Config.LOCAL_DATA_DIR
    source_dir = output_dir / "us_neighborhoods"
    source_dir.mkdir(parents=True, exist_ok=True)

    # In production, download from Census Bureau
    # For NYC area, create sample neighborhoods
    sample_data = {
        "neighborhood_id": ["NYC-001", "NYC-002", "NYC-003"],
        "name": ["Lower Manhattan", "Midtown", "Upper East Side"],
        "county_fips": ["36061", "36061", "36047"],
        "state_fips": ["36", "36", "36"],
        "geometry": [
            '{"type":"Polygon","coordinates":[[[-74.01,40.70],[-73.99,40.70],[-73.99,40.72],[-74.01,40.72],[-74.01,40.70]]}',
            '{"type":"Polygon","coordinates":[[[-73.99,40.74],[-73.97,40.74],[-73.97,40.76],[-73.99,40.76],[-73.99,40.74]]}',
            '{"type":"Polygon","coordinates":[[[-73.97,40.77],[-73.95,40.77],[-73.95,40.79],[-73.97,40.79],[-73.97,40.77]]}',
        ],
        "source": ["us_neighborhoods", "us_neighborhoods", "us_neighborhoods"],
    }

    df = pd.DataFrame(sample_data)
    output_path = source_dir / "part-00000.parquet"
    pq.write_table(pa.Table.from_pandas(df), output_path, compression="snappy")

    logger.info(f"Saved US Neighborhoods to {output_path}")
    return upload_to_bronze(output_path, "us_neighborhoods/2024-01.parquet", "parquet")


# =============================================================================
# BRONZE INGESTION RUNNER
# =============================================================================


def run_bronze_ingestion() -> bool:
    """
    Run complete Bronze layer ingestion.

    Orchestrates all 5 data sources:
        1. US Accidents
        2. NYC 311
        3. USGS Earthquakes
        4. OSM Infrastructure
        5. US Neighborhoods

    Returns
    -------
    bool
        True if all sources succeeded
    """
    # Setup telemetry
    if TELEMETRY_AVAILABLE:
        setup_telemetry(service_name="bronze-ingestion", environment="development")

    logger.info("Starting Bronze layer ingestion...")

    results = []

    # Run all fetchers with error tracking
    try:
        with traced_context("bronze", "fetch_us_accidents"):
            results.append(("us_accidents", fetch_us_accidents()))
    except Exception as e:
        logger.error(f"Error fetching us_accidents: {e}")
        inc_bronze_errors()
        results.append(("us_accidents", False))
    
    try:
        with traced_context("bronze", "fetch_nyc_311"):
            results.append(("nyc_311", fetch_nyc_311_requests()))
    except Exception as e:
        logger.error(f"Error fetching nyc_311: {e}")
        inc_bronze_errors()
        results.append(("nyc_311", False))
    
    try:
        with traced_context("bronze", "fetch_usgs_earthquakes"):
            results.append(("usgs_earthquakes", fetch_usgs_earthquakes()))
    except Exception as e:
        logger.error(f"Error fetching usgs_earthquakes: {e}")
        inc_bronze_errors()
        results.append(("usgs_earthquakes", False))
    
    try:
        with traced_context("bronze", "fetch_osm_infrastructure"):
            results.append(("osm_infrastructure", fetch_osm_infrastructure()))
    except Exception as e:
        logger.error(f"Error fetching osm_infrastructure: {e}")
        inc_bronze_errors()
        results.append(("osm_infrastructure", False))
    
    try:
        with traced_context("bronze", "fetch_us_neighborhoods"):
            results.append(("us_neighborhoods", fetch_us_neighborhoods()))
    except Exception as e:
        logger.error(f"Error fetching us_neighborhoods: {e}")
        inc_bronze_errors()
        results.append(("us_neighborhoods", False))

    # Summary
    success_count = sum(1 for _, success in results if success)
    total_count = len(results)
    
    # Count failures and increment error metrics
    failed_count = total_count - success_count
    if failed_count > 0:
        for _ in range(failed_count):
            inc_bronze_errors()
    
    logger.info(
        f"Bronze ingestion complete: {success_count}/{total_count} sources successful"
    )

    # Count failures and increment error metrics
    failed_count = total_count - success_count
    if failed_count > 0:
        for _ in range(failed_count):
            inc_bronze_errors()

    for source, success in results:
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"  {source}: {status}")

    # Flush telemetry before returning
    if TELEMETRY_AVAILABLE:
        flush_telemetry()

    return success_count == total_count


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Start Prometheus metrics server (exposes /metrics endpoint)
    try:
        from prometheus_client import start_http_server
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 8888))
        sock.close()
        if result == 0:
            logger.info("Port 8888 already in use, reusing existing server")
        else:
            start_http_server(8888)
            logger.info("Prometheus metrics server started on port 8888")
    except Exception as e:
        logger.debug(f"Could not start metrics server: {e}")
    
    success = run_bronze_ingestion()
    sys.exit(0 if success else 1)
