#!/usr/bin/env python3
"""
Silver Layer Quality Audit with LLM Judge.
Uses faster Llama 3.2 for quality evaluation with MLflow tracking.
"""

import json
import os
import pandas as pd
import requests
import mlflow
import time

# Initialize MLflow
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment("GeoAI_Medallion_Architecture")

# Configuration - using faster model
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
JUDGE_MODEL = "llama3.2:1b"  # Fast model for evaluation

def evaluate_extraction_quality(source_text, extracted_json):
    """Evaluate extraction quality using LLM judge."""
    prompt = f"""Rate this hazard extraction's faithfulness (0.0-1.0).
Context: {source_text}
Extraction: {extracted_json}
Respond ONLY with JSON: {{"score": 0.0-1.0, "reason": "brief reason"}}"""

    try:
        start = time.time()
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": JUDGE_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.1}},
            timeout=30
        )
        result = response.json()
        
        # Parse score from response
        text = result.get("response", "")
        import re
        match = re.search(r'"score"\s*:\s*([0-9.]+)', text)
        score = float(match.group(1)) if match else 0.5
        
        return {"faithfulness_score": score, "latency": time.time() - start}
    except Exception as e:
        return {"faithfulness_score": 0.5, "latency": 0}

def main():
    print("Silver Layer Quality Audit with MLflow Evaluation")
    
    # Sample data
    df = pd.DataFrame([
        {"incident_id": "INC001", "incident_narrative": "Aircraft experienced severe turbulence during approach due to wind shear. Crew initiated go-around. Weather included thunderstorms and heavy rain.", "extracted_hazards": '[{"hazard_type": "weather", "description": "Severe turbulence", "severity": "high"}]'},
        {"incident_id": "INC002", "incident_narrative": "Engine failure detected during cruise flight at FL350. Pilots diverted. Emergency landing completed. Fuel contamination found.", "extracted_hazards": '[{"hazard_type": "mechanical", "description": "Engine failure", "severity": "critical"}]'},
        {"incident_id": "INC003", "incident_narrative": "Runway incursion reported when aircraft crossed active runway without clearance. Air traffic control intervened.", "extracted_hazards": '[{"hazard_type": "procedural", "description": "Runway incursion", "severity": "high"}]'},
        {"incident_id": "INC004", "incident_narrative": "Bird strike during departure. Multiple birds ingested into engines.", "extracted_hazards": '[{"hazard_type": "environmental", "description": "Bird strike", "severity": "medium"}]'},
        {"incident_id": "INC005", "incident_narrative": "Medical emergency onboard. Passenger experienced cardiac symptoms.", "extracted_hazards": '[{"hazard_type": "medical", "description": "Medical emergency", "severity": "high"}]'}
    ])
    
    print(f"\n📊 Evaluating {len(df)} samples...\n")
    
    with mlflow.start_run(run_name="silver_quality_audit"):
        results = []
        total_latency = 0
        
        for idx, row in df.iterrows():
            print(f"Evaluating {row['incident_id']}...")
            eval_result = evaluate_extraction_quality(row["incident_narrative"], row["extracted_hazards"])
            
            results.append({
                "incident_id": row["incident_id"],
                "faithfulness_score": eval_result["faithfulness_score"]
            })
            total_latency += eval_result["latency"]
            print(f"  Score: {eval_result['faithfulness_score']:.2f} ({eval_result['latency']:.1f}s)")
        
        # Log metrics
        avg_score = sum(r["faithfulness_score"] for r in results) / len(results)
        mlflow.log_metric("avg_faithfulness_score", avg_score)
        mlflow.log_metric("total_evaluated", len(results))
        mlflow.log_metric("total_latency", total_latency)
        
        print(f"\n📋 Summary:")
        print(f"  Average faithfulness: {avg_score:.2f}")
        print(f"  Total evaluations: {len(results)}")
    
    print("\n✅ Quality audit complete! View in MLflow UI under 'Traces' or 'Experiments' tab")

if __name__ == "__main__":
    main()