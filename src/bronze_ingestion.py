#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
BRONZE LAYER INGESTION - Data Extraction from Multiple Sources
================================================================================
File: src/bronze_ingestion.py

Purpose:
    Extract raw data from 5 distinct sources and save to MinIO Bronze bucket:
    1. US Accidents (Kaggle - Static CSV)
    2. US City Neighborhood Boundaries (Kaggle - Static GeoJSON)
    3. USGS Earthquake Hazards (Live API - GeoJSON)
    4. OSM Hospitals/Fire Stations (Overpass API - GeoJSON)
    5. NYC 311 Service Requests (Socrata API/CSV)

Author: GeoAI Principal Data Engineer
Version: 1.0.0
================================================================================
"""

import os
import sys
import json
import time
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# UTILITY HELPERS
# =============================================================================


def safe_parse_json(json_str: str) -> Optional[Dict[str, Any]]:
    """
    Safely parse a JSON string, returning None on failure.

    Parameters
    ----------
    json_str : str
        Raw JSON string (potentially from API response).

    Returns
    -------
    Optional[Dict[str, Any]]
        Parsed dict or None if parsing fails.
    """
    if not json_str or not isinstance(json_str, str):
        return None
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def generate_source_id(source: str, timestamp_str: str) -> str:
    """
    Generate a deterministic source ID for a record.

    Parameters
    ----------
    source : str
        Source name (e.g. 'usgs_earthquakes', 'osm_hospitals').
    timestamp_str : str
        Timestamp string (e.g. '2024-01-01' or '20240101').

    Returns
    -------
    str
        Deterministic source ID in format: {source}_{timestamp_str}.
    """
    clean_ts = timestamp_str.replace("-", "").replace(":", "").replace(" ", "")
    return f"{source}_{clean_ts}"


# =============================================================================
# CONFIGURATION
# =============================================================================


class Config:
    """Configuration for Bronze layer ingestion."""

    # MinIO/S3 Configuration (use host.docker.internal for cross-network access)
    MINIO_ENDPOINT = os.getenv("S3_ENDPOINT", "http://host.docker.internal:9900")
    MINIO_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    MINIO_SECRET_KEY = os.getenv(
        "AWS_SECRET_ACCESS_KEY", "minioadmin"
    )  # Correct password
    BRONZE_BUCKET = "geo-lakehouse/bronze"

    # Kaggle Configuration
    KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME", "")
    KAGGLE_API_KEY = os.getenv("KAGGLE_API_KEY", "")

    # API Endpoints
    USGS_API_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    OSM_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
    NYC_311_API_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"

    # Dataset settings
    OUTPUT_FORMAT = "json"  # or "csv"
    BATCH_SIZE = 50000

    # Local paths for fallback (Docker container path)
    # Try to use /workspace, fall back to temp dir if not writable
    try:
        _local_path = Path("/workspace/data/bronze")
        _local_path.mkdir(parents=True, exist_ok=True)
        LOCAL_DATA_DIR = _local_path
    except OSError:
        import tempfile

        LOCAL_DATA_DIR = Path(tempfile.gettempdir()) / "geoai" / "bronze"
        LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# MINIO CLIENT
# =============================================================================


class MinIOClient:
    """MinIO client for uploading raw data to Bronze bucket."""

    def __init__(self, config: Config):
        self.config = config
        self.client = None
        self._init_client()

    def _init_client(self):
        """Initialize MinIO client with boto3."""
        try:
            import boto3
            from botocore.config import Config as BotoConfig

            self.client = boto3.client(
                "s3",
                endpoint_url=self.config.MINIO_ENDPOINT,
                aws_access_key_id=self.config.MINIO_ACCESS_KEY,
                aws_secret_access_key=self.config.MINIO_SECRET_KEY,
                config=BotoConfig(signature_version="s3v4"),
                region_name="us-east-1",
            )

            # Create bucket if not exists
            try:
                bucket_name = self.config.BRONZE_BUCKET.split("/")[0]
                self.client.create_bucket(Bucket=bucket_name)
            except Exception:
                pass  # Bucket may already exist

            logger.info(f"MinIO client initialized: {self.config.MINIO_ENDPOINT}")
        except ImportError:
            logger.warning("boto3 not installed, using local fallback")

    def upload_file(self, local_path: Path, s3_key: str) -> bool:
        """Upload file to MinIO."""
        try:
            if self.client:
                bucket_name = self.config.BRONZE_BUCKET.split("/")[0]
                self.client.upload_file(str(local_path), bucket_name, s3_key)
                logger.info(f"Uploaded to: {s3_key}")
                return True
            return False
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return False

    def upload_dataframe(
        self, df: pd.DataFrame, s3_key: str, format: str = "json"
    ) -> bool:
        """Upload DataFrame to MinIO."""
        try:
            import io

            bucket_name = self.config.BRONZE_BUCKET.split("/")[0]

            if format == "csv":
                content = df.to_csv(index=False)
                content_type = "text/csv"
            else:
                content = df.to_json(orient="records", lines=True)
                content_type = "application/json"

            self.client.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=content.encode("utf-8"),
                ContentType=content_type,
            )
            logger.info(f"Uploaded DataFrame to: {s3_key}")
            return True
        except Exception as e:
            logger.error(f"DataFrame upload failed: {e}")
            return False


# =============================================================================
# KAGGLE INGESTION
# =============================================================================


def download_kaggle_dataset(
    dataset_name: str, output_path: Path, config: Config
) -> bool:
    """
    Download dataset from Kaggle using kaggle python package.

    Parameters:
        dataset_name: Kaggle dataset identifier (e.g., 'sobhe/us-accidents')
        output_path: Local path to save downloaded files
        config: Configuration

    Returns:
        True if successful
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        # Initialize Kaggle API
        api = KaggleApi()
        api.authenticate()

        # Download dataset
        logger.info(f"Downloading Kaggle dataset: {dataset_name}")
        api.dataset_download_files(dataset_name, path=str(output_path), unzip=True)

        logger.info(f"Kaggle dataset downloaded to: {output_path}")
        return True

    except ImportError:
        logger.error("kaggle package not installed: pip install kaggle")
        return False
    except Exception as e:
        logger.error(f"Kaggle download failed: {e}")
        return False


def upload_kaggle_to_bronze(
    dataset_path: Path, s3_prefix: str, minio_client: MinIOClient, config: Config
) -> bool:
    """Upload downloaded Kaggle files to Bronze bucket."""

    try:
        # Find CSV/JSON files
        for file_path in dataset_path.rglob("*"):
            if file_path.suffix.lower() in [".csv", ".json", ".geojson"]:
                s3_key = f"{s3_prefix}/{file_path.name}"

                # Read and re-upload as JSON/CSV
                if file_path.suffix == ".csv":
                    df = pd.read_csv(file_path, low_memory=False)
                    return minio_client.upload_dataframe(df, s3_prefix + ".csv", "csv")
                elif file_path.suffix == ".json":
                    df = pd.read_json(file_path)
                    return minio_client.upload_dataframe(
                        df, s3_prefix + ".json", "json"
                    )

        return True
    except Exception as e:
        logger.error(f"Kaggle upload failed: {e}")
        return False


# =============================================================================
# USGS EARTHQUAKE API INGESTION
# =============================================================================


def fetch_usgs_earthquakes(
    config: Config, days_back: int = 30, min_magnitude: float = 2.0
) -> Optional[pd.DataFrame]:
    """
    Fetch earthquake data from USGS API.

    Uses USGS FDSN Web Services to query earthquake events.

    Parameters:
        config: Configuration object
        days_back: Number of days to look back
        min_magnitude: Minimum magnitude to filter

    Returns:
        DataFrame with earthquake data
    """
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        # Build GeoJSON query
        params = {
            "format": "geojson",
            "starttime": start_date.strftime("%Y-%m-%d"),
            "endtime": end_date.strftime("%Y-%m-%d"),
            "minmagnitude": min_magnitude,
            "orderby": "time",
        }

        logger.info(f"Fetching USGS earthquakes (last {days_back} days)")
        response = requests.get(config.USGS_API_URL, params=params, timeout=60)
        response.raise_for_status()

        # Parse GeoJSON
        data = response.json()

        # Extract features
        features = data.get("features", [])
        records = []

        for feature in features:
            props = feature.get("properties", {})
            coords = feature.get("geometry", {}).get("coordinates", [])

            record = {
                "event_id": feature.get("id"),
                "event_type": props.get("type"),
                "place": props.get("place"),
                "magnitude": props.get("mag"),
                "felt": props.get("felt"),
                "cdi": props.get("cdi"),
                "mmi": props.get("mmi"),
                "alert": props.get("alert"),
                "status": props.get("status"),
                "tsunami": props.get("tsunami"),
                "significance": props.get("significance"),
                "datetime": props.get("time"),
                "updated": props.get("updated"),
                "url": props.get("url"),
                "detail_url": props.get("detail"),
                "longitude": coords[0] if len(coords) > 0 else None,
                "latitude": coords[1] if len(coords) > 1 else None,
                "depth": coords[2] if len(coords) > 2 else None,
            }
            records.append(record)

        df = pd.DataFrame(records)
        logger.info(f"Fetched {len(df)} earthquake records")

        return df

    except Exception as e:
        logger.error(f"USGS fetch failed: {e}")
        return None


# =============================================================================
# OSM OVERPASS API INGESTION
# =============================================================================


def fetch_osm_infrastructure(
    config: Config,
    bbox: tuple = (-125, 24, -66, 50),  # US bounding box
    facility_types: List[str] = ["hospital", "fire_station"],
) -> Optional[pd.DataFrame]:
    """
    Fetch hospitals and fire stations from OSM using Overpass API.

    Parameters:
        config: Configuration
        bbox: Bounding box (west, south, east, north)
        facility_types: Types of facilities to query

    Returns:
        DataFrame with OSM infrastructure
    """
    try:
        all_records = []

        for facility_type in facility_types:
            # OverpassQL query
            query = f"""
            [out:json][timeout:300];
            (
              node["amenity"={facility_type}]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
              way["amenity"={facility_type}]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
            );
            out center tags;
            """

            logger.info(f"Fetching OSM {facility_type}s...")

            response = requests.post(
                config.OSM_OVERPASS_URL, data={"data": query}, timeout=300
            )
            response.raise_for_status()

            data = response.json()
            elements = data.get("elements", [])

            for element in elements:
                # Get center point for ways, node for nodes
                lat = element.get("lat")
                lon = element.get("lon")

                if element.get("type") == "way" and "center" in element:
                    lat = element["center"].get("lat")
                    lon = element["center"].get("lon")

                tags = element.get("tags", {})

                record = {
                    "osm_id": element.get("id"),
                    "osm_type": element.get("type"),
                    "facility_type": facility_type,
                    "name": tags.get("name"),
                    "operator": tags.get("operator"),
                    "emergency": tags.get("emergency"),
                    "opening_hours": tags.get("opening_hours"),
                    "phone": tags.get("phone"),
                    "website": tags.get("website"),
                    "addr_street": tags.get("addr:street"),
                    "addr_city": tags.get("addr:city"),
                    "addr_state": tags.get("addr:state"),
                    "addr_postcode": tags.get("addr:postcode"),
                    "longitude": lon,
                    "latitude": lat,
                }
                all_records.append(record)

            # Rate limiting
            time.sleep(2)

        df = pd.DataFrame(all_records)
        logger.info(f"Fetched {len(df)} OSM infrastructure records")

        return df

    except Exception as e:
        logger.error(f"OSM fetch failed: {e}")
        return None


# =============================================================================
# NYC 311 API INGESTION
# =============================================================================


def fetch_nyc_311_requests(
    config: Config, limit: int = 100000, date_range: int = 365
) -> Optional[pd.DataFrame]:
    """
    Fetch NYC 311 service requests from Socrata API.

    Parameters:
        config: Configuration
        limit: Maximum records to fetch
        date_range: Days to look back

    Returns:
        DataFrame with 311 requests
    """
    try:
        # Calculate date filter
        end_date = datetime.now()
        start_date = end_date - timedelta(days=date_range)
        date_filter = start_date.strftime("%Y-%m-%dT%H:%M:%S")

        # Socrata SODA API query
        params = {
            "$where": f"created_date >= '{date_filter}'",
            "$limit": limit,
            "$order": "created_date DESC",
            "$select": "complaint_type,descriptor,created_date,closed_date,agency,agency_name,latitude,longitude,status,location_type,open_data_channel,due_date,resolution_description,community_board,borough",
        }

        logger.info(f"Fetching NYC 311 requests (last {date_range} days)")
        response = requests.get(config.NYC_311_API_URL, params=params, timeout=120)
        response.raise_for_status()

        # Parse JSON response
        data = response.json()
        df = pd.DataFrame(data)

        logger.info(f"Fetched {len(df)} NYC 311 records")

        return df

    except Exception as e:
        logger.error(f"NYC 311 fetch failed: {e}")
        return None


def fetch_nyc_311_csv_fallback(config: Config, output_path: Path) -> bool:
    """
    Download NYC 311 from NYC Open Data CSV export (fallback).

    Note: Full dataset is ~1GB, requires bulk download.
    """
    csv_url = (
        "https://data.cityofnewyork.us/api/views/erm2-nwe9/rows.csv?accessType=DOWNLOAD"
    )

    try:
        logger.info("Downloading NYC 311 CSV (fallback method)...")
        response = requests.get(csv_url, stream=True, timeout=600)

        if response.status_code == 200:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"NYC 311 CSV saved to: {output_path}")
            return True

        return False
    except Exception as e:
        logger.error(f"NYC 311 CSV download failed: {e}")
        return False


# =============================================================================
# MAIN INGESTION PIPELINE
# =============================================================================


def run_bronze_ingestion(minio_client: Optional[MinIOClient] = None) -> bool:
    """
    Execute complete Bronze layer ingestion.

    Parameters:
        minio_client: Optional MinIO client (creates if not provided)

    Returns:
        True if successful
    """
    config = Config()

    if minio_client is None:
        minio_client = MinIOClient(config)

    logger.info("=" * 60)
    logger.info("STARTING BRONZE LAYER INGESTION")
    logger.info("=" * 60)

    # =========================================================================
    # 1. INGEST US ACCIDENTS (Kaggle)
    # =========================================================================
    logger.info("[1/5] Ingesting US Accidents (Kaggle)...")

    try:
        # Kaggle dataset name for US Accidents
        ACCIDENTS_DATASET = "sobhe/us-accidents"

        # Attempt Kaggle download (requires API key)
        if config.KAGGLE_USERNAME and config.KAGGLE_API_KEY:
            kaggle_path = config.LOCAL_DATA_DIR / "us_accidents"
            download_kaggle_dataset(ACCIDENTS_DATASET, kaggle_path, config)
            upload_kaggle_to_bronze(kaggle_path, "us_accidents", minio_client, config)
        else:
            logger.warning("Kaggle credentials not configured, skipping")

    except Exception as e:
        logger.error(f"US Accidents ingestion failed: {e}")

    # =========================================================================
    # 2. INGEST US CITY NEIGHBORHOOD BOUNDARIES (Kaggle)
    # =========================================================================
    logger.info("[2/5] Ingesting US City Neighborhood Boundaries...")

    try:
        NEIGHBORHOODS_DATASET = "cityofnewyork/nyc-neighborhoods"

        if config.KAGGLE_USERNAME and config.KAGGLE_API_KEY:
            kaggle_path = config.LOCAL_DATA_DIR / "neighborhoods"
            download_kaggle_dataset(NEIGHBORHOODS_DATASET, kaggle_path, config)
            upload_kaggle_to_bronze(kaggle_path, "neighborhoods", minio_client, config)
        else:
            logger.warning("Kaggle credentials not configured, skipping")

    except Exception as e:
        logger.error(f"Neighborhoods ingestion failed: {e}")

    # =========================================================================
    # 3. INGEST USGS EARTHQUAKES (Live API)
    # =========================================================================
    logger.info("[3/5] Ingesting USGS Earthquake Hazards...")

    try:
        df_usgs = fetch_usgs_earthquakes(config, days_back=30, min_magnitude=2.0)

        if df_usgs is not None and len(df_usgs) > 0:
            local_path = config.LOCAL_DATA_DIR / "usgs_earthquakes.json"
            df_usgs.to_json(local_path, orient="records", lines=True)

            # Upload to MinIO
            minio_client.upload_dataframe(df_usgs, "usgs_earthquakes.json", "json")

        logger.info(
            f"USGS earthquakes: {len(df_usgs) if df_usgs is not None else 0} records"
        )

    except Exception as e:
        logger.error(f"USGS ingestion failed: {e}")

    # =========================================================================
    # 4. INGEST OSM INFRASTRUCTURE (Overpass API)
    # =========================================================================
    logger.info("[4/5] Ingesting OSM Hospitals/Fire Stations...")

    try:
        df_osm = fetch_osm_infrastructure(
            config,
            bbox=(-125, 24, -66, 50),  # US bounding box
            facility_types=["hospital", "fire_station"],
        )

        if df_osm is not None and len(df_osm) > 0:
            local_path = config.LOCAL_DATA_DIR / "osm_infrastructure.json"
            df_osm.to_json(local_path, orient="records", lines=True)

            # Upload to MinIO
            minio_client.upload_dataframe(df_osm, "osm_infrastructure.json", "json")

        logger.info(
            f"OSM infrastructure: {len(df_osm) if df_osm is not None else 0} records"
        )

    except Exception as e:
        logger.error(f"OSM ingestion failed: {e}")

    # =========================================================================
    # 5. INGEST NYC 311 SERVICE REQUESTS (Socrata API)
    # =========================================================================
    logger.info("[5/5] Ingesting NYC 311 Service Requests...")

    try:
        df_311 = fetch_nyc_311_requests(config, limit=100000, date_range=365)

        if df_311 is not None and len(df_311) > 0:
            local_path = config.LOCAL_DATA_DIR / "nyc_311_requests.json"
            df_311.to_json(local_path, orient="records", lines=True)

            # Upload to MinIO
            minio_client.upload_dataframe(df_311, "nyc_311_requests.json", "json")

        logger.info(
            f"NYC 311 requests: {len(df_311) if df_311 is not None else 0} records"
        )

    except Exception as e:
        logger.error(f"NYC 311 ingestion failed: {e}")

    # =========================================================================
    # COMPLETE
    # =========================================================================
    logger.info("=" * 60)
    logger.info("BRONZE LAYER INGESTION COMPLETE")
    logger.info("=" * 60)

    return True


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bronze Layer Ingestion")
    parser.add_argument(
        "--local-only", action="store_true", help="Skip MinIO, save locally only"
    )
    args = parser.parse_args()

    if args.local_only:
        # Run without MinIO
        run_bronze_ingestion(minio_client=None)
    else:
        # Run with MinIO
        run_bronze_ingestion()
