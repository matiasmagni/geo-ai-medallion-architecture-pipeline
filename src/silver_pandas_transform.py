#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SILVER LAYER - Spatial Transformation using Pandas/GeoPandas
Fallback for environments without Spark/Sedona.
"""

import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, shape
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BRONZE_PATH = Path("/tmp/geoai/bronze")
SILVER_PATH = Path("/tmp/geoai/silver")
SILVER_PATH.mkdir(parents=True, exist_ok=True)

def load_neighborhoods_mask():
    """Load neighborhood polygons to use as a land mask."""
    path = BRONZE_PATH / "us_neighborhoods" / "part-00000.parquet"
    if not path.exists():
        logger.error(f"Neighborhoods not found at {path}")
        return None
    
    df = pd.read_parquet(path)
    # Convert GeoJSON strings to shapely geometries
    df['geometry'] = df['geometry'].apply(lambda x: shape(json.loads(x)))
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    return gdf

def filter_points_on_land(points_df, land_mask_gdf, lat_col='latitude', lon_col='longitude'):
    """Filter points to keep only those within the land mask polygons."""
    if land_mask_gdf is None:
        logger.warning("No land mask provided, skipping filter.")
        return points_df
    
    initial_count = len(points_df)
    # Create GeoDataFrame from points
    geometry = [Point(xy) for xy in zip(points_df[lon_col], points_df[lat_col])]
    points_gdf = gpd.GeoDataFrame(points_df, geometry=geometry, crs="EPSG:4326")
    
    # Spatial join: keep points that are within any land polygon
    # 'inner' join with land mask
    filtered_gdf = gpd.sjoin(points_gdf, land_mask_gdf, how="inner", predicate='intersects')
    
    # Remove join columns and geometry
    result_df = pd.DataFrame(filtered_gdf.drop(columns=['index_right', 'geometry']))
    
    final_count = len(result_df)
    logger.info(f"Filtered {initial_count} -> {final_count} points (removed {initial_count - final_count} water dots)")
    return result_df

def transform_source(name, lat_col='latitude', lon_col='longitude', land_mask=None):
    """Generic transformation for a point-based source."""
    path = BRONZE_PATH / name / "part-00000.parquet"
    if not path.exists():
        logger.warning(f"Source {name} not found at {path}")
        return
    
    logger.info(f"Transforming {name}...")
    df = pd.read_parquet(path)
    
    # Apply land mask filter
    if land_mask is not None:
        df = filter_points_on_land(df, land_mask, lat_col, lon_col)
    
    # Save to Silver
    output_dir = SILVER_PATH / f"{name}_silver"
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_dir / "part-00000.parquet", index=False)
    logger.info(f"Saved {name} to {output_dir}")

def run_pipeline():
    logger.info("Starting Silver transformation (Pandas/GeoPandas fallback)...")
    
    # 1. Load Land Mask
    land_mask = load_neighborhoods_mask()
    
    # 2. Transform sources
    transform_source("us_accidents", land_mask=land_mask)
    transform_source("nyc_311", land_mask=land_mask)
    transform_source("osm_infrastructure", land_mask=land_mask)
    
    # USGS Earthquakes (Global, don't mask by NYC land)
    transform_source("usgs_earthquakes", land_mask=None)
    
    logger.info("Silver transformation complete!")

if __name__ == "__main__":
    run_pipeline()
