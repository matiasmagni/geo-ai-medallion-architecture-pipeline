#!/bin/bash
# =============================================================================
# GeoAI Medallion Architecture - Unix Execution Script
# =============================================================================
# This script runs the GeoAI ELT pipelines on Unix/Linux/macOS
#
# Usage:
#   ./scripts/run_pipeline.sh [silver|gold|test|test-l0|test-l1|test-l2|test-l3|init|clean|status]
#
# Requirements:
#   - Docker and Docker Compose installed
#   - .env file configured
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Print colored message
print_msg() {
    local color=$1
    local msg=$2
    echo -e "${color}${msg}${NC}"
}

# Print info
info() {
    print_msg "$BLUE" "[INFO] $1"
}

# Print success
success() {
    print_msg "$GREEN" "[SUCCESS] $1"
}

# Print warning
warn() {
    print_msg "$YELLOW" "[WARN] $1"
}

# Print error
error() {
    print_msg "$RED" "[ERROR] $1"
}

# Check environment
check_env() {
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            warn "No .env file found, copying .env.example"
            cp .env.example .env
            warn "Please edit .env with your configuration"
        else
            error ".env file not found!"
            exit 1
        fi
    fi
    success "Environment check passed"
}

# Docker compose command
docker_compose() {
    docker compose "$@"
}

# =============================================================================
# COMMAND: Initialize infrastructure
# =============================================================================
cmd_init() {
    info "Initializing GeoAI infrastructure..."
    
    check_env
    
    # Build and start services
    info "Building Docker images..."
    docker_compose build
    
    info "Starting services..."
    docker_compose up -d
    
    info "Waiting for services to be ready..."
    sleep 30
    
    cmd_status
    
    success "Infrastructure initialized!"
    info "Access services at:"
    info "  - JupyterLab: http://localhost:8888"
    info "  - MinIO:   http://localhost:9901 (console)"
    info "  - MLflow:  http://localhost:5001"
    info "  - Spark:   http://localhost:9080"
}

# =============================================================================
# COMMAND: Start services
# =============================================================================
cmd_start() {
    info "Starting GeoAI services..."
    docker_compose up -d
    cmd_status
}

# =============================================================================
# COMMAND: Stop services
# =============================================================================
cmd_stop() {
    info "Stopping GeoAI services..."
    docker_compose down
    success "Services stopped"
}

# =============================================================================
# COMMAND: Status
# =============================================================================
cmd_status() {
    echo ""
    echo "==================================================================="
    echo "  GeoAI Services Status"
    echo "==================================================================="
    docker_compose ps
    echo ""
}

# =============================================================================
# COMMAND: Run Silver pipeline
# =============================================================================
cmd_silver() {
    info "Running Silver layer ELT pipeline..."
    
    # Check if Spark is running
    if ! docker_compose ps | grep -q spark; then
        error "Spark not running! Run: $0 init"
        exit 1
    fi
    
    # Run the Silver pipeline
    docker_compose exec -T spark \
        spark-submit \
        --master spark://spark:7077 \
        --deploy-mode client \
        --conf spark.driver.memory=4g \
        --conf spark.executor.memory=4g \
        --packages org.apache.sedona:sedona-spark-3.5_2.12:1.5.1,org.datasyslab:geotools-wrapper:1.5.1-28.2,io.delta:delta-spark_2.12:3.1.0,org.apache.hadoop:hadoop-aws:3.3.4 \
        --repositories https://artifacts.unidata.ucar.edu/repository/unidata-all/ \
        /home/jovyan/src/silver_enrichment.py "$@"
    
    success "Silver pipeline complete!"
}

# =============================================================================
# COMMAND: Run Gold pipeline
# =============================================================================
cmd_gold() {
    info "Running Gold layer ELT pipeline..."
    
    # Check if services running
    if ! docker_compose ps | grep -q spark; then
        error "Spark not running! Run: $0 init"
        exit 1
    fi
    
    # Run the Gold pipeline
    docker_compose exec -T spark \
        spark-submit \
        --master spark://spark:7077 \
        --deploy-mode client \
        --conf spark.driver.memory=4g \
        --conf spark.executor.memory=4g \
        --packages org.apache.sedona:sedona-spark-3.5_2.12:1.5.1,org.datasyslab:geotools-wrapper:1.5.1-28.2,io.delta:delta-spark_2.12:3.1.0,org.apache.hadoop:hadoop-aws:3.3.4 \
        --repositories https://artifacts.unidata.ucar.edu/repository/unidata-all/ \
        /home/jovyan/src/gold_dimensional_modeling.py "$@"
    
    success "Gold pipeline complete!"
}

# =============================================================================
# COMMAND: Run all tests
# =============================================================================
cmd_test() {
    info "Running all tests..."
    python -m pytest tests/ -v --tb=short || true
}

# =============================================================================
# COMMAND: L0 tests
# =============================================================================
cmd_test_l0() {
    info "Running L0 unit tests..."
    python -m pytest tests/test_l0_unit.py -v --tb=short || true
}

# =============================================================================
# COMMAND: L1 tests
# =============================================================================
cmd_test_l1() {
    info "Running L1 integration tests..."
    python -m pytest tests/test_l1_integration.py -v --tb=short || true
}

# =============================================================================
# COMMAND: L2 tests
# =============================================================================
cmd_test_l2() {
    info "Running L2 pipeline tests..."
    python -m pytest tests/test_l2_component.py -v --tb=short || true
}

# =============================================================================
# COMMAND: L3 tests
# =============================================================================
cmd_test_l3() {
    info "Running L3 E2E tests..."
    python -m pytest tests/test_l3_e2e.py -v --tb=short || true
}

# =============================================================================
# COMMAND: Logs
# =============================================================================
cmd_logs() {
    docker_compose logs -f "$@"
}

# =============================================================================
# COMMAND: Clean up
# =============================================================================
cmd_clean() {
    warn "Cleaning up GeoAI resources..."
    
    # Stop services
    docker_compose down -v
    
    # Remove test data
    if [ -d "notebooks/.ipynb_checkpoints" ]; then
        rm -rf notebooks/.ipynb_checkpoints
    fi
    
    success "Cleanup complete!"
}

# =============================================================================
# COMMAND: Shell access
# =============================================================================
cmd_shell() {
    docker_compose exec "$@"
}

# =============================================================================
# COMMAND: Update Heatmap Data
# =============================================================================
cmd_heatmap() {
    info "Exporting Gold data to frontend GeoJSON..."
    
    # 1. Copy script to container
    docker cp scripts/update_frontend_data.py geoai-spark:/home/jovyan/update_frontend_data.py
    
    # 2. Run export in container
    docker exec geoai-spark spark-submit \
        --master spark://spark:7077 \
        --packages org.apache.sedona:sedona-spark-3.5_2.12:1.5.1,org.datasyslab:geotools-wrapper:1.5.1-28.2,io.delta:delta-spark_2.12:3.1.0,org.apache.hadoop:hadoop-aws:3.3.4 \
        --repositories https://artifacts.unidata.ucar.edu/repository/unidata-all/ \
        /home/jovyan/update_frontend_data.py
    
    # 3. Copy back to host
    mkdir -p geo-ai-heatmap/public/data
    docker cp geoai-spark:/home/jovyan/geo-ai-heatmap/public/data/heatmap.geojson geo-ai-heatmap/public/data/heatmap.geojson
    
    success "Frontend data updated! View at http://localhost:3001"
}

# =============================================================================
# MAIN
# =============================================================================
usage() {
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  init          Initialize and start infrastructure"
    echo "  start        Start services"
    echo "  stop         Stop services"
    echo "  status       Show service status"
    echo "  silver       Run Silver pipeline"
    echo "  gold         Run Gold pipeline"
    echo "  heatmap      Export Gold data to frontend heatmap"
    echo "  test         Run all tests"
    echo "  test-l0      Run L0 unit tests"
    echo "  test-l1      Run L1 integration tests"
    echo "  test-l2      Run L2 pipeline tests"
    echo "  test-l3      Run L3 E2E tests"
    echo "  logs         View logs (follows)"
    echo "  clean        Clean up resources"
    echo "  shell       Open shell in container"
    echo ""
    echo "Examples:"
    echo "  $0 init"
    echo "  $0 silver --input-format csv"
    echo "  $0 gold --text-column description"
    echo "  $0 heatmap"
    echo "  $0 logs -f spark"
}

# Parse command
COMMAND="${1:-usage}"
shift || true

case "$COMMAND" in
    init)
        cmd_init "$@"
        ;;
    start)
        cmd_start "$@"
        ;;
    stop)
        cmd_stop "$@"
        ;;
    status)
        cmd_status "$@"
        ;;
    silver)
        cmd_silver "$@"
        ;;
    gold)
        cmd_gold "$@"
        ;;
    heatmap)
        cmd_heatmap "$@"
        ;;
    test)
        cmd_test "$@"
        ;;
    test-l0)
        cmd_test_l0 "$@"
        ;;
    test-l1)
        cmd_test_l1 "$@"
        ;;
    test-l2)
        cmd_test_l2 "$@"
        ;;
    test-l3)
        cmd_test_l3 "$@"
        ;;
    logs)
        cmd_logs "$@"
        ;;
    clean)
        cmd_clean "$@"
        ;;
    shell)
        cmd_shell "$@"
        ;;
    *)
        usage
        exit 1
        ;;
esac