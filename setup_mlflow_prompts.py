#!/usr/bin/env python3
"""
Setup MLflow Prompts for Hazard Extraction and Hallucination Judgment.
This script creates and versions the prompts used in the GeoAI Medallion Architecture.
"""

import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
import os

def setup_mlflow_prompts():
    """
    Set up MLflow prompts for:
    1. Llama 3 Hazard Extraction
    2. DeepSeek-R1 Hallucination Judge
    """
    # Set MLflow tracking URI if not already set via environment
    # mlflow.set_tracking_uri("http://localhost:5000")  # Example for local MLflow server
    
    # Ensure we are in the right MLflow experiment
    mlflow.set_experiment("GeoAI_Medallion_Architecture")
    
    # Prompt 1: Llama 3 Hazard Extraction
    hazard_extraction_prompt = """
    You are an expert in extracting structured hazard information from aviation incident reports.
    Given the following incident narrative, extract all mentioned hazards and return them as a JSON array.
    Each hazard object should have the following fields:
    - hazard_type: Category of hazard (e.g., weather, mechanical, human_factor, procedural)
    - description: Concise description of the hazard
    - severity: Estimated severity level (low, medium, high) based on context
    - location: Location where hazard occurred (if mentioned)
    - timestamp: Time related to hazard (if mentioned)
    
    If no hazards are found, return an empty array.
    
    Incident Narrative:
    {incident_text}
    
    Extracted Hazards (JSON array):
    """
    
    # Prompt 2: DeepSeek-R1 Hallucination Judge
    hallucination_judge_prompt = """
    You are an expert evaluator tasked with detecting hallucinations in AI-generated hazard extractions.
    Compare the extracted hazards JSON against the original incident narrative.
    Evaluate whether the extracted hazards are:
    1. FAITHFUL: All extracted hazards are directly supported by the text.
    2. PARTIALLY FAITHFUL: Most hazards are supported, but some may be inferred or slightly exaggerated.
    3. HALLUCINATED: Significant portions of the extraction are not supported by the text or contradict it.
    
    Provide your evaluation as a JSON object with:
    - faithfulness_score: Float between 0.0 (completely hallucinated) and 1.0 (completely faithful)
    - explanation: Brief explanation of your scoring
    - unsupported_claims: List of specific claims in the extraction that are not supported by the text
    
    Original Incident Narrative:
    {source_text}
    
    Extracted Hazards JSON:
    {extracted_json}
    
    Evaluation (JSON object):
    """
    
    # Log the prompts to MLflow
    with mlflow.start_run(run_name="Setup_Prompts") as run:
        # Log Hazard Extraction Prompt
        mlflow.prompts.log_prompt(
            name="llama3_hazard_extraction",
            prompt=hazard_extraction_prompt,
            version="1.0",
            description="Prompt for Llama 3 to extract structured hazard information from aviation incident narratives",
            tags={
                "agent_type": "extraction",
                "model": "llama3",
                "task": "hazard_extraction",
                "layer": "silver"
            }
        )
        
        # Log Hallucination Judge Prompt
        mlflow.prompts.log_prompt(
            name="deepseek_hallucination_judge",
            prompt=hallucination_judge_prompt,
            version="1.0",
            description="Prompt for DeepSeek-R1 to judge faithfulness of hazard extractions",
            tags={
                "agent_type": "judge",
                "model": "deepseek-r1",
                "task": "hallucination_judgment",
                "layer": "silver"
            }
        )
        
        print(f"Prompts logged successfully in run {run.info.run_id}")
        print("View prompts in MLflow UI under the 'Prompts' tab")

if __name__ == "__main__":
    setup_mlflow_prompts()