import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri('http://localhost:5001')

print("Registering Ollama models in MLflow Model Registry...\n")

client = MlflowClient()

models = [
    {
        "name": "geoai-llama3.2-1b",
        "description": "Llama 3.2 1B parameter model for structured data extraction from geospatial hazard reports. Used in Silver layer for extracting severity, hazard_type, and location data.",
        "tags": {
            "model_family": "llama",
            "model_name": "llama3.2:1b",
            "framework": "ollama",
            "pipeline": "medallion",
            "layer": "silver_enrichment",
            "purpose": "structured_extraction",
            "parameters": "1B",
        },
    },
    {
        "name": "geoai-mistral-7b",
        "description": "Mistral 7B parameter model for LLM-as-a-Judge quality evaluation. Used in Silver layer quality audit to score extraction accuracy (1-5 scale) and detect hallucinations.",
        "tags": {
            "model_family": "mistral",
            "model_name": "mistral",
            "framework": "ollama",
            "pipeline": "medallion",
            "layer": "silver_quality_audit",
            "purpose": "llm_judge_evaluation",
            "parameters": "7B",
        },
    },
    {
        "name": "geoai-qwen3-4b",
        "description": "Qwen 3 4B parameter model for multilingual geospatial data processing. Used for handling international hazard data and multi-language queries.",
        "tags": {
            "model_family": "qwen",
            "model_name": "qwen3:4b",
            "framework": "ollama",
            "pipeline": "medallion",
            "layer": "silver_enrichment",
            "purpose": "multilingual_processing",
            "parameters": "4B",
        },
    },
    {
        "name": "geoai-phi4-mini",
        "description": "Phi-4 Mini 3.8B parameter model for lightweight inference. Used for fast prototyping and smaller-scale extraction tasks.",
        "tags": {
            "model_family": "microsoft",
            "model_name": "phi4-mini",
            "framework": "ollama",
            "pipeline": "medallion",
            "layer": "silver_enrichment",
            "purpose": "fast_inference",
            "parameters": "3.8B",
        },
    },
]

for model_spec in models:
    name = model_spec["name"]
    description = model_spec["description"]
    tags = model_spec["tags"]
    
    print(f"Registering model: {name}")
    try:
        # Create or update model
        try:
            client.create_registered_model(
                name=name, 
                description=description, 
                tags=tags
            )
            print(f"  ✓ Created model: {name}")
        except Exception as e:
            if "already exists" in str(e).lower():
                # Update tags for existing model
                for tag_key, tag_value in tags.items():
                    client.set_registered_model_tag(name, tag_key, tag_value)
                print(f"  ✓ Updated tags for: {name}")
            else:
                raise
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print()

print("\nDone! View models at: http://localhost:5001/#/models")