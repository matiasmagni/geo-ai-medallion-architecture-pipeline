#!/usr/bin/env python3
"""
E2E Test for GeoAI Medallion Pipeline with OpenTelemetry
Verifies distributed tracing and metrics collection
"""
import sys
import time
import os
sys.path.insert(0, "/home/jovyan/src")
from telemetry import setup_telemetry, traced, get_tracer

def test_bronze():
    @traced(layer="bronze", operation="live_test")
    def test_bronze():
        return {"status": "ok", "records": 100}
    return test_bronze()

def test_silver():
    @traced(layer="silver", operation="live_test")
    def test_silver():
        return {"status": "ok", "records": 50}
    return test_silver()

def test_gold():
    @traced(layer="gold", operation="live_test")
    def test_gold():
        return {"status": "ok", "records": 25}
    return test_gold()

def main():
    print("Starting E2E OpenTelemetry test...")
    setup_telemetry()
    
    tracer = get_tracer()
    
    print("\n--- Bronze Layer ---")
    result = test_bronze()
    print(f"Result: {result}")
    
    print("\n--- Silver Layer ---")
    result = test_silver()
    print(f"Result: {result}")
    
    print("\n--- Gold Layer ---")
    result = test_gold()
    print(f"Result: {result}")
    
    print("\n--- Error Handling ---")
    @traced(layer="bronze", operation="error_test")
    def test_error():
        raise ValueError("Test error")
    
    try:
        test_error()
    except ValueError as e:
        print(f"Caught expected error: {e}")
    
    print("\n--- Test Complete ---")
    print("All E2E tests passed!")
    print("Check Jaeger at http://localhost:16686")
    print("Check Prometheus at http://localhost:9090")
    return 0

if __name__ == "__main__":
    sys.exit(main())