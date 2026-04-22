#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
OpenTelemetry Instrumentation for GeoAI Medallion Pipeline
================================================================================
File: src/telemetry.py

Purpose:
    Provides distributed tracing and metrics collection for the pipeline
    using OpenTelemetry.

Features:
    - Automatic span creation for each layer (Bronze, Silver, Gold)
    - Custom attributes for data quality tracking
    - Metrics: records processed, errors, duration
    - Export to OTLP-compatible backends (Jaeger, Prometheus, etc.)

Usage:
    from telemetry import setup_telemetry, get_tracer, get_meter
    
    # In your pipeline code
    with get_tracer("bronze").start_as_current_span("fetch_us_accidents") as span:
        span.set_attribute("source", "us_accidents")
        # ... your code ...

Author: GeoAI Principal Data Engineer
Version: 1.0.0
================================================================================
"""

import os
import logging
import time
from typing import Optional, Dict, Any
from functools import wraps
from contextlib import contextmanager

# OpenTelemetry imports - import in stages to avoid cascade failures
OTEL_AVAILABLE = False

# Dummy classes needed for fallback
class DummyTracer:
    def start_as_current_span(self, name): return DummySpan()
    def start_span(self, name): return DummySpan()

class DummySpan:
    def set_attribute(self, k, v): pass
    def set_status(self, ok, msg=''): pass
    def add_event(self, n, a=None): pass
    def record_exception(self, e): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass

class DummyMeter:
    def counter(self, n): return DummyCounter()

class DummyCounter:
    def add(self, v, a=None): pass

# For Status/StatusCode - try to import, fallback
try:
    from opentelemetry.trace import Status, StatusCode
except ImportError:
    class Status:
        def __init__(self, code, message=''):
            pass
    class StatusCode:
        OK = "ok"
        ERROR = "error"
        UNSET = "unset"
        UNAVAILABLE = "unavailable"

# Now try imports
try:
    # First base imports that should always work
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    
    # Optional imports - may not be installed
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    except ImportError:
        OTLPSpanExporter = None
        
    try:
        from opentelemetry.exporter.prometheus import PrometheusMetricsExporter
    except ImportError:
        PrometheusMetricsExporter = None
        
    try:
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter
    except ImportError:
        ConsoleSpanExporter = None
        
    OTEL_AVAILABLE = True
except Exception as e:
    logger.debug(f"OpenTelemetry setup: {e}")
    OTLPSpanExporter = None
    PrometheusMetricsExporter = None
    ConsoleSpanExporter = None


logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


class TelemetryConfig:
    """
    OpenTelemetry configuration for the pipeline.
    """

    # Service identification
    SERVICE_NAME: str = os.getenv("OTEL_SERVICE_NAME", "geo-ai-medallion-pipeline")
    SERVICE_VERSION: str = os.getenv("OTEL_SERVICE_VERSION", "1.0.0")

    # Exporters
    OTEL_EXPORTER_OTLP_ENDPOINT: str = os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
    )
    OTEL_EXPORTER_OTLP_INSECURE: bool = os.getenv(
        "OTEL_EXPORTER_OTLP_INSECURE", "true"
    ).lower() == "true"

    # Console export (useful for debugging)
    OTEL_EXPORT_CONSOLE: bool = os.getenv(
        "OTEL_EXPORT_CONSOLE", "false"
    ).lower() == "true"

    # Prometheus metrics
    OTEL_METRICS_PORT: int = int(os.getenv("OTEL_METRICS_PORT", "8888"))

    # Sampling
    OTEL_TRACE_SAMPLER: str = os.getenv("OTEL_TRACE_SAMPLER", "parentbased_always_on")
    OTEL_TRACE_SAMPLER_PARAM: float = float(os.getenv("OTEL_TRACE_SAMPLER_PARAM", "1.0"))

    # Resource attributes
    LAYER: str = os.getenv("OTEL_LAYER", "unknown")  # bronze, silver, gold
    ENVIRONMENT: str = os.getenv("OTEL_ENVIRONMENT", "development")


# =============================================================================
# TRACING
# =============================================================================


class GeoAITracer:
    """
    Wrapper for OpenTelemetry tracing with GeoAI-specific helpers.
    """

    _instance: Optional["GeoAITracer"] = None
    _tracer: Optional[Any] = None
    _meter: Optional[Any] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not OTEL_AVAILABLE:
            self._tracer = DummyTracer()
            self._meter = DummyMeter()
            return

        # Create resource with service info
        resource = Resource.create({
            SERVICE_NAME: TelemetryConfig.SERVICE_NAME,
            "service.version": TelemetryConfig.SERVICE_VERSION,
            "deployment.environment": TelemetryConfig.ENVIRONMENT,
            "telemetry.sdk.name": "opentelemetry",
        })

        # Setup tracer provider
        provider = TracerProvider(resource=resource)

        # Add OTLP exporter if configured
        if TelemetryConfig.OTEL_EXPORTER_OTLP_ENDPOINT:
            try:
                otlp_exporter = OTLPSpanExporter(
                    endpoint=TelemetryConfig.OTEL_EXPORTER_OTLP_ENDPOINT,
                    insecure=TelemetryConfig.OTEL_EXPORTER_OTLP_INSECURE,
                )
                provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            except Exception as e:
                logger.warning(f"Failed to add OTLP exporter: {e}")

        # Add console exporter for debugging
        if TelemetryConfig.OTEL_EXPORT_CONSOLE:
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

        trace.set_tracer_provider(provider)
        self._tracer = trace.get_tracer(
            TelemetryConfig.SERVICE_NAME,
            TelemetryConfig.SERVICE_VERSION,
        )

        # Setup metrics
        try:
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
            from opentelemetry.exporter.prometheus import PrometheusMetricsExporter

            reader = PeriodicExportingMetricReader(
                PrometheusMetricsExporter(port=TelemetryConfig.OTEL_METRICS_PORT)
            )
            meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
            metrics.set_meter_provider(meter_provider)
            self._meter = metrics.get_meter(TelemetryConfig.SERVICE_NAME)
        except Exception as e:
            logger.warning(f"Failed to setup metrics: {e}")
            self._meter = DummyMeter()

    @property
    def tracer(self) -> Any:
        return self._tracer

    @property
    def meter(self) -> Any:
        return self._meter

    def start_layer_span(
        self,
        layer: str,
        operation: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Start a span for a layer operation.

        Parameters
        ----------
        layer : str
            Layer name (bronze, silver, gold)
        operation : str
            Operation name (e.g., "fetch", "transform", "write")
        attributes : dict, optional
            Additional attributes

        Returns
        -------
        Span
            Configured span
        """
        span = self._tracer.start_span(f"{layer}.{operation}")

        # Add standard attributes
        span.set_attribute("layer", layer)
        span.set_attribute("operation", operation)
        span.set_attribute("environment", TelemetryConfig.ENVIRONMENT)

        # Add custom attributes
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))

        return span

    def record_metrics(
        self,
        layer: str,
        metric_name: str,
        value: float = 1.0,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a metric value.

        Parameters
        ----------
        layer : str
            Layer name
        metric_name : str
            Metric name (e.g., "records.processed", "errors")
        value : float
            Metric value
        attributes : dict, optional
            Metric attributes
        """
        if not hasattr(self._meter, 'counter'):
            return

        try:
            counter = self._meter.create_counter(
                name=f"{layer}.{metric_name}",
                description=f"{metric_name} for {layer} layer",
                unit="1",
            )
            counter.add(value, attributes or {})
        except Exception as e:
            logger.debug(f"Failed to record metric: {e}")


# Global tracer instance
_tracer_instance: Optional[GeoAITracer] = None


def get_tracer() -> GeoAITracer:
    """
    Get the global tracer instance.

    Returns
    -------
    GeoAITracer
        Configured tracer
    """
    global _tracer_instance
    if _tracer_instance is None:
        _tracer_instance = GeoAITracer()
    return _tracer_instance


# =============================================================================
# DECORATORS
# =============================================================================


def traced(layer: str, operation: str, attributes: Optional[Dict[str, Any]] = None):
    """
    Decorator to automatically trace a function.

    Parameters
    ----------
    layer : str
        Layer name (bronze, silver, gold)
    operation : str
        Operation name
    attributes : dict, optional
        Additional span attributes

    Example
    -------
    @traced("bronze", "fetch")
    def fetch_us_accidents():
        ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            attrs = (attributes or {}).copy()
            attrs["function"] = func.__name__

            with tracer.start_layer_span(layer, operation, attrs) as span:
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    if OTEL_AVAILABLE:
                        span.set_status(Status(StatusCode.OK))
                    tracer.record_metrics(layer, "records.processed", 1)
                    return result
                except Exception as e:
                    if OTEL_AVAILABLE:
                        span.set_status(Status(StatusCode.ERROR), str(e))
                    else:
                        span.set_status(False, str(e))
                    span.record_exception(e)
                    tracer.record_metrics(layer, "errors", 1)
                    raise
                finally:
                    duration = time.time() - start_time
                    span.set_attribute("duration_seconds", duration)
                    tracer.record_metrics(layer, "duration_seconds", duration)

        return wrapper
    return decorator


def traced_context(layer: str, operation: str):
    """
    Context manager for tracing.

    Parameters
    ----------
    layer : str
        Layer name
    operation : str
        Operation name

    Example
    -------
    with traced_context("silver", "transform"):
        df = transform_data()
    """
    tracer = get_tracer()
    return tracer.start_layer_span(layer, operation)


# =============================================================================
# PIPELINE-SPECIFIC HELPERS
# =============================================================================


class PipelineTelemetry:
    """
    High-level telemetry helpers for pipeline operations.
    """

    @staticmethod
    def trace_bronze_fetch(source: str):
        """Trace bronze layer data fetch."""
        return traced("bronze", "fetch", {"data_source": source})

    @staticmethod
    def trace_silver_transform(layer: str, dataset: str):
        """Trace silver layer transformation."""
        return traced("silver", "transform", {"dataset": dataset, "type": layer})

    @staticmethod
    def trace_gold_write(table: str):
        """Trace gold layer write."""
        return traced("gold", "write", {"table_name": table})

    @staticmethod
    def trace_llm_inference(model: str, input_tokens: int = 0):
        """Trace LLM inference call."""
        return traced("silver", "llm_inference", {
            "model": model,
            "input_tokens": input_tokens,
        })

    @staticmethod
    def trace_spatial_operation(operation: str, geometry_type: str):
        """Trace spatial/geometric operation."""
        return traced("silver", "spatial", {
            "operation": operation,
            "geometry_type": geometry_type,
        })


# =============================================================================
# SETUP
# =============================================================================


def setup_telemetry(
    service_name: Optional[str] = None,
    otlp_endpoint: Optional[str] = None,
    environment: str = "development",
) -> GeoAITracer:
    """
    Initialize OpenTelemetry for the pipeline.

    Parameters
    ----------
    service_name : str, optional
        Custom service name
    otlp_endpoint : str, optional
        OTLP endpoint URL
    environment : str
        Environment name (development, staging, production)

    Returns
    -------
    GeoAITracer
        Configured tracer
    """
    if service_name:
        TelemetryConfig.SERVICE_NAME = service_name
    if otlp_endpoint:
        TelemetryConfig.OTEL_EXPORTER_OTLP_ENDPOINT = otlp_endpoint
    TelemetryConfig.ENVIRONMENT = environment

    return get_tracer()


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================


def export_spans_to_console() -> None:
    """Enable console span export for debugging."""
    TelemetryConfig.OTEL_EXPORT_CONSOLE = True
    global _tracer_instance
    _tracer_instance = None  # Reset to reinitialize


def create_span_processor(
    exporter_type: str = "console",
    **kwargs,
) -> Optional[Any]:
    """
    Create a custom span processor.

    Parameters
    ----------
    exporter_type : str
        Exporter type (console, otlp, jaeger)
    **kwargs
        Additional exporter arguments

    Returns
    -------
    SpanProcessor or None
    """
    if exporter_type == "console":
        return BatchSpanProcessor(ConsoleSpanExporter())
    elif exporter_type == "otlp":
        return BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=kwargs.get("endpoint", TelemetryConfig.OTEL_EXPORTER_OTLP_ENDPOINT),
                insecure=kwargs.get("insecure", True),
            )
        )
    return None


# =============================================================================
# FLUSH
# =============================================================================


def flush_telemetry() -> None:
    """
    Force flush all telemetry data.
    Useful before shutdown or between pipeline stages.
    """
    if OTEL_AVAILABLE:
        try:
            trace.get_tracer_provider().force_flush()
            metrics.get_meter_provider().force_flush()
        except Exception as e:
            logger.warning(f"Failed to flush telemetry: {e}")
