#!/usr/bin/env python3
"""
Silver Layer AI Enrichment with MLflow Tracing.
Processes Bronze layer data using Llama 3 via Ollama for hazard extraction,
with full MLflow tracing capabilities.
"""

import json
import time
import os
import re
import requests
import pandas as pd
import mlflow
from mlflow.tracing.fluent import trace

# Initialize MLflow
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment("GeoAI_Medallion_Architecture")

# Ollama configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLAMA3_MODEL = "llama3.2:1b"  # Adjust based on your model tag

@trace
def llama3_hazard_extraction(incident_text):
    """Extract hazards from incident text using Llama 3 via Ollama."""
    start_time = time.time()
    
    # Prepare extraction prompt
    prompt = f"""You are an expert in extracting structured hazard information from aviation incident reports.
Given the following incident narrative, extract all mentioned hazards and return them as a JSON array.
Each hazard object should have the following fields:
- hazard_type: Category of hazard (e.g., weather, mechanical, human_factor, procedural)
- description: Concise description of the hazard
- severity: Estimated severity level (low, medium, high)

Incident Narrative:
{incident_text}

Extracted Hazards (JSON array):"""

    # Call Ollama
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": LLAMA3_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "top_p": 0.9}
    }
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        result = {"response": "[]", "error": str(e), "prompt_eval_count": 0, "eval_count": 0}
    
    end_time = time.time()
    latency = end_time - start_time
    
    # Robust JSON extraction
    extracted_text = result.get("response", "[]")
    hazards = []
    
    try:
        # Try direct parse first
        extracted_text = extracted_text.strip()
        hazards = json.loads(extracted_text)
    except json.JSONDecodeError:
        pass
    
    if not isinstance(hazards, list):
        # Try to find JSON array in text
        json_match = re.search(r'\[\s*\{[\s\S]*\}\s*\]', extracted_text)
        if json_match:
            try:
                hazards = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                hazards = []
    
    if not isinstance(hazards, list):
        hazards = []
    
    # Extract token usage
    prompt_tokens = result.get("prompt_eval_count", 0)
    completion_tokens = result.get("eval_count", 0)
    total_tokens = prompt_tokens + completion_tokens
    
    return {
        "hazards": hazards,
        "latency_seconds": latency,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "model": LLAMA3_MODEL
    }

def process_incident_batch(incidents_df, text_column='incident_narrative'):
    """Process a batch of incidents with MLflow tracing."""
    results = []
    total_latency = 0
    total_tokens = 0
    
    for idx, row in incidents_df.iterrows():
        incident_text = row.get(text_column, "")
        
        if pd.isna(incident_text) or incident_text == "":
            results.append({
                "incident_id": row.get("incident_id", idx),
                "extracted_hazards": "[]",
                "status": "skipped"
            })
            continue
        
        # Extract hazards using traced function
        result = llama3_hazard_extraction(incident_text)
        
        total_latency += result["latency_seconds"]
        total_tokens += result["total_tokens"]
        
        results.append({
            "incident_id": row.get("incident_id", idx),
            "extracted_hazards": json.dumps(result["hazards"]),
            "status": "success",
            "latency": result["latency_seconds"],
            "tokens": result["total_tokens"],
            "num_hazards": len(result["hazards"])
        })
    
    return results, total_latency, total_tokens

def create_sample_data():
    """Create sample incident data for testing."""
    sample_incidents = [
        {"incident_id": "INC001", "incident_narrative": "Aircraft experienced severe turbulence during approach to runway 27L due to wind shear. The crew initiated a go-around procedure. Weather conditions included thunderstorms and heavy rain."},
        {"incident_id": "INC002", "incident_narrative": "Engine failure detected during cruise flight at FL350. Pilots diverted to nearest suitable airport. Emergency landing completed safely. Maintenance inspection revealed fuel contamination as cause."},
        {"incident_id": "INC003", "incident_narrative": "Runway incursion reported when aircraft crossed active runway without clearance. Air traffic control intervened immediately. Investigation shows communication misunderstanding between tower and flight crew."},
        {"incident_id": "INC004", "incident_narrative": "Bird strike occurred during departure from runway 18. Multiple birds ingested into both engines. One engine lost power. Aircraft returned to airport successfully."},
        {"incident_id": "INC005", "incident_narrative": "Medical emergency onboard required diversion. Passenger experienced cardiac symptoms. Flight crew administered first aid and landed at nearest airport. Passenger transported to hospital."}
    ]
    return pd.DataFrame(sample_incidents)

def main():
    print("=" * 60)
    print("Silver Layer AI Enrichment with MLflow Tracing")
    print("=" * 60)
    
    print("\n📊 Loading sample incident data...")
    df = create_sample_data()
    print(f"Loaded {len(df)} incidents")
    
    print("\n🔄 Processing with Llama 3 extraction (tracing enabled)...")
    with mlflow.start_run(run_name="silver_layer_batch_extraction"):
        results, total_latency, total_tokens = process_incident_batch(df)
        
        # Log metrics
        mlflow.log_metric("total_incidents", len(df))
        mlflow.log_metric("successful_extractions", len([r for r in results if r["status"] == "success"]))
        mlflow.log_metric("total_latency_seconds", total_latency)
        mlflow.log_metric("total_tokens", total_tokens)
        mlflow.log_metric("avg_latency_per_incident", total_latency / len(results) if results else 0)
        mlflow.log_metric("avg_tokens_per_incident", total_tokens / len(results) if results else 0)
        
        print("\n📋 Results:")
        print("-" * 60)
        for r in results:
            print(f"  {r['incident_id']}: {r['status']}")
            if r['status'] == 'success':
                print(f"    Latency: {r.get('latency', 0):.2f}s, Tokens: {r.get('tokens', 0)}")
                print(f"    Hazards extracted: {r.get('num_hazards', 0)}")
    
    print("\n✅ Traces captured! View in MLflow UI under 'Traces' tab")
    return results

if __name__ == "__main__":
    main()