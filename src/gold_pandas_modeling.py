#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GOLD LAYER - Dimensional Modeling using Pandas
Combines all silver point data into a final FACT table for the heatmap.
"""

import os
import pandas as pd
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SILVER_PATH = Path("/tmp/geoai/silver")
GOLD_PATH = Path("/tmp/geoai/gold")
GOLD_PATH.mkdir(parents=True, exist_ok=True)

def transform_gold():
    logger.info("Starting Gold transformation (Pandas fallback)...")
    
    all_events = []
    
    # 1. Process US Accidents
    path = SILVER_PATH / "us_accidents_silver" / "part-00000.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        # Standardize for Fact table
        df['ai_hazard_type'] = 'traffic'
        df['final_severity'] = df['severity'].fillna(5).astype(int)
        all_events.append(df[['latitude', 'longitude', 'final_severity', 'ai_hazard_type']])
        logger.info(f"Added {len(df)} US Accidents to Gold")

    # 2. Process NYC 311
    path = SILVER_PATH / "nyc_311_silver" / "part-00000.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        # Basic mapping for hazard type
        def map_311(desc):
            desc = str(desc).lower()
            if 'noise' in desc: return 'noise'
            if 'street' in desc: return 'infrastructure'
            if 'dumping' in desc: return 'environmental'
            return 'other'
        
        df['ai_hazard_type'] = df['description'].apply(map_311)
        # If severity doesn't exist, use 3 as Series
        severity_series = df['severity'] if 'severity' in df.columns else pd.Series(3, index=df.index)
        df['final_severity'] = severity_series.fillna(3).astype(int)
        all_events.append(df[['latitude', 'longitude', 'final_severity', 'ai_hazard_type']])
        logger.info(f"Added {len(df)} NYC 311 requests to Gold")

    # 3. Process OSM Infrastructure
    path = SILVER_PATH / "osm_infrastructure_silver" / "part-00000.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        # Infrastructure points are "hazard mitigators" but we show them on map
        def map_osm(ftype):
            if ftype == 'fire_station': return 'fire'
            return 'medical'
            
        df['ai_hazard_type'] = df['facility_type'].apply(map_osm)
        # Static infrastructure has low priority severity (1)
        df['final_severity'] = 1
        all_events.append(df[['latitude', 'longitude', 'final_severity', 'ai_hazard_type']])
        logger.info(f"Added {len(df)} OSM facilities to Gold")

    if not all_events:
        logger.error("No silver data found to transform to Gold!")
        return
    
    # Combine everything
    fact_hazard_events = pd.concat(all_events, ignore_index=True)
    
    # Save to local project root for the heatmap script to find it easily
    output_path = Path("fact_hazard_events.parquet")
    fact_hazard_events.to_parquet(output_path, index=False)
    
    # Also save to Gold bucket area
    fact_hazard_events.to_parquet(GOLD_PATH / "fact_hazard_events.parquet", index=False)
    
    logger.info(f"Gold transformation complete! Saved {len(fact_hazard_events)} events to {output_path}")

if __name__ == "__main__":
    transform_gold()
