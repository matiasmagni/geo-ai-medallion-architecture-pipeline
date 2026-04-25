#!/usr/bin/env python3
"""
NYC Incident Heatmap Generator - Uses real ML predictions and hazard events.
Reads from the medallion pipeline's gold/silver layer parquet files.
"""

import json
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, shape


def get_land_mask():
    """Load neighborhood polygons to use as a precise land mask."""
    data_dir = Path(__file__).parent.parent
    
    # Try nyc_landmask.json first for highest precision
    mask_path = data_dir / "nyc_landmask.json"
    if mask_path.exists():
        print(f"DEBUG: Loading precise land mask from {mask_path}")
        gdf = gpd.read_file(mask_path)
        print(f"DEBUG: Loaded {len(gdf)} features for land mask from JSON")
        return gdf

    # Fallback to Bronze neighborhoods
    path = data_dir / "tmp" / "geoai" / "bronze" / "us_neighborhoods" / "part-00000.parquet"
    print(f"DEBUG: Checking for land mask at {path}")
    if not path.exists():
        # Try absolute path if relative fails
        path = Path("/tmp/geoai/bronze/us_neighborhoods/part-00000.parquet")
        print(f"DEBUG: Checking absolute path {path}")
        if not path.exists():
            print("DEBUG: Land mask file NOT FOUND")
            return None

    print(f"DEBUG: Loading land mask from {path}")
    df = pd.read_parquet(path)
    df['geometry'] = df['geometry'].apply(lambda x: shape(json.loads(x)))
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    print(f"DEBUG: Loaded {len(gdf)} polygons for land mask")
    return gdf


def generate_heatmap(output_path: str = "geo-ai-heatmap/public/data/heatmap.geojson") -> dict:
    """Generate heatmap GeoJSON from real ML prediction data."""

    data_dir = Path(__file__).parent.parent
    land_mask = get_land_mask()

    import pyarrow.parquet as pq

    predictions_df = pd.read_parquet(data_dir / "ml_predictions_gold.parquet")
    hazards_df = pd.read_parquet(data_dir / "fact_hazard_events.parquet")

    # ── LAND MASK FILTERING ──────────────────────────────────────────────────
    if land_mask is not None:
        def apply_mask(df):
            geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
            filtered = gpd.sjoin(gdf, land_mask, how="inner", predicate='intersects')
            return pd.DataFrame(filtered.drop(columns=['index_right', 'geometry']))

        predictions_df = apply_mask(predictions_df)
        print(f"  ML Predictions: Filtered to {len(predictions_df)} on land")

        # Hazards were already filtered in Silver, but let's be safe
        hazards_df = apply_mask(hazards_df)

    predictions = predictions_df.to_dict('list')
    hazards = hazards_df.to_dict('list')

    features = []

    # ── ML Predictions (fire_risk_prob + response metrics) ──────────────────────
    for i in range(len(predictions["latitude"])):
        lat = predictions["latitude"][i]
        lon = predictions["longitude"][i]

        fire_risk = predictions["fire_risk_prob"][i]
        facility = predictions["facility_type"][i]
        response = predictions["response_time_mins"][i]
        bed_demand = predictions["bed_demand"][i]
        overpop = predictions["overpopulation_risk"][i]

        # Classify by highest risk signal
        max_risk = max(fire_risk, bed_demand, overpop)

        if fire_risk > 0.5:
            category = "fire"
        elif overpop > 0.5:
            category = "high"
        elif bed_demand > 0.5:
            category = "medical"
        elif fire_risk > 0.2:
            category = "hazmat"
        else:
            category = "minimal"

        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "category": category,
                "weight": max_risk,
                "source": "ml_prediction",
                "facility_type": facility,
                "fire_risk_prob": round(fire_risk, 4),
                "response_time_mins": round(response, 2),
                "bed_demand": round(bed_demand, 4),
                "overpopulation_risk": round(overpop, 4),
            }
        })

    # ── Hazard Events (real incidents from silver layer) ───────────────────────
    severity_weights = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5, 6: 0.6, 7: 0.7, 8: 0.8, 9: 0.9, 10: 1.0}
    for i in range(len(hazards["latitude"])):
        lat = hazards["latitude"][i]
        lon = hazards["longitude"][i]

        severity = hazards["final_severity"][i]
        hazard_type = hazards["ai_hazard_type"][i]

        # Map hazard type to category
        if hazard_type == "fire":
            category = "fire"
        elif hazard_type in ("infrastructure", "structural"):
            category = "hazmat"
        elif hazard_type == "traffic":
            category = "minimal"
        else:
            category = "medical"

        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "category": category,
                "weight": severity_weights.get(severity, severity / 10),
                "source": "hazard_event",
                "hazard_type": hazard_type,
                "severity": severity,
                "final_severity": severity,
            }
        })

    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "total_features": len(features),
            "ml_predictions": len(predictions["latitude"]),
            "hazard_events": len(hazards["latitude"]),
        }
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(geojson, f, indent=2)

    print(f"Saved: {output_path}")
    print(f"  Total features: {len(features)}")
    return geojson


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "geo-ai-heatmap/public/data/heatmap.geojson"
    generate_heatmap(out)