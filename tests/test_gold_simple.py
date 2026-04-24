#!/usr/bin/env python3
"""
================================================================================
TEST SUITE FOR GOLD SIMPLE - UNIT TESTS
================================================================================
File: tests/test_gold_simple.py

Purpose: Test gold_simple.py module functions
Coverage target: 100%
================================================================================
"""

import pytest
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestGoldSimple:
    """Test gold_simple module functions."""

    def test_create_spark_is_callable(self):
        """Test create_spark can be imported."""
        from gold_simple import create_spark
        assert callable(create_spark)

    def test_create_ollama_udf_is_callable(self):
        """Test create_ollama_udf can be imported."""
        from gold_simple import create_ollama_udf
        assert callable(create_ollama_udf)

    def test_main_is_callable(self):
        """Test main can be imported."""
        from gold_simple import main
        assert callable(main)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])