#!/usr/bin/env python3
"""
Runner script for the GeoAI Medallion Architecture Pipeline.
Executes: Bronze → Silver → Gold
"""

import os
import sys
import subprocess


def run_command(cmd, description):
    """Run a shell command and print result."""
    print(f"\n{'=' * 60}")
    print(f"  {description}")
    print("=" * 60)
    result = subprocess.run(cmd, shell=True, capture_output=False)
    return result.returncode == 0


def main():
    # Get project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)

    print("""
╔══════════════════════════════════════════════════════════════╗
║     GEOAI MEDALLION ARCHITECTURE PIPELINE                  ║
║     Shift-Left AI Pattern                                   ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # Environment setup
    os.environ["OLLAMA_BASE_URL"] = "http://geoai-ollama:11434"
    os.environ["OLLAMA_MODEL"] = "llama3.2:1b"

    # Step 1: Bronze Layer (raw data)
    print("\n" + "=" * 60)
    print("  BRONZE LAYER - Raw Data Ingestion")
    print("=" * 60)
    os.system("python src/bronze_ingestion.py")

    # Step 2: Silver Layer (spatial + LLM enrichment)
    print("\n" + "=" * 60)
    print("  SILVER LAYER - Spatial + LLM Enrichment")
    print("=" * 60)
    os.system("python src/silver_enrichment.py")

    # Step 3: Gold Layer (dimensional modeling)
    print("\n" + "=" * 60)
    print("  GOLD LAYER - Star Schema Modeling")
    print("=" * 60)
    os.system("python src/gold_dimensional_modeling.py")

    print("""
╔══════════════════════════════════════════════════════════════╗
║     PIPELINE COMPLETE                                      ║
╚══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
