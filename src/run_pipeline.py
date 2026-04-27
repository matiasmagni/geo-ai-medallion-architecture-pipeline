#!/usr/bin/env python3
"""
Runner script for the GeoAI Medallion Architecture Pipeline.
Executes: Bronze → Silver → Gold
"""

import os
import sys
import subprocess


# Inject JVM options for Java 17+ compatibility
os.environ["JDK_JAVA_OPTIONS"] = (
    "--add-opens=java.base/java.lang=ALL-UNNAMED "
    "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED "
    "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED "
    "--add-opens=java.base/java.io=ALL-UNNAMED "
    "--add-opens=java.base/java.net=ALL-UNNAMED "
    "--add-opens=java.base/java.nio=ALL-UNNAMED "
    "--add-opens=java.base/java.util=ALL-UNNAMED "
    "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED "
    "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED "
    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
    "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED "
    "--add-opens=java.base/sun.security.action=ALL-UNNAMED "
    "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED "
    "--add-opens=java.security.jgss/sun.security.krb5=ALL-UNNAMED "
)


def run_command(cmd, description):
    """Run a shell command and print result."""
    print(f"
{'=' * 60}")
    print(f"  {description}")
    print("=" * 60)
    result = subprocess.run(cmd, shell=True, capture_output=False)
    return result.returncode == 0


def main():
    # Get project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)

    header_lines = [
        "╔══════════════════════════════════════════════════════════════╗",
        "║     GEOAI MEDALLION ARCHITECTURE PIPELINE                  ║",
        "║     Shift-Left AI Pattern                                   ║",
        "╚══════════════════════════════════════════════════════════════╝"
    ]
    print("
" + "
".join(header_lines) + "
")

    # Environment setup
    os.environ["OLLAMA_BASE_URL"] = "http://geoai-ollama:11434"
    os.environ["OLLAMA_MODEL"] = "llama3.2:1b"

    # Step 1: Bronze Layer (raw data)
    print("
" + "=" * 60)
    print("  BRONZE LAYER - Raw Data Ingestion")
    print("=" * 60)
    os.system("python src/bronze_ingestion.py")

    # Step 2: Silver Layer (spatial + LLM enrichment)
    print("
" + "=" * 60)
    print("  SILVER LAYER - Spatial + LLM Enrichment")
    print("=" * 60)
    os.system("python src/silver_enrichment.py")

    # Step 3: Gold Layer (dimensional modeling)
    print("
" + "=" * 60)
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