# =============================================================================
# GeoAI Medallion Architecture - Windows Execution Script
# =============================================================================
# This script runs the GeoAI ELT pipelines on Windows (PowerShell)
#
# Usage:
#   .\run_pipeline.ps1 [silver|gold|test|test-l0|test-l1|test-l2|test-l3|init|clean|status]
#
# Requirements:
#   - Docker Desktop for Windows
#   - .env file configured
# =============================================================================

param(
    [Parameter(Position=0)]
    [ValidateSet("init", "start", "stop", "status", "silver", "gold", "test", "test-l0", "test-l1", "test-l2", "test-l3", "logs", "clean", "shell")]
    [string]$Command = "usage",
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Args
)

# Colors for output
$RED = "#FF0000"
$GREEN = "#00FF00"
$YELLOW = "#FFFF00"
$BLUE = "#0000FF"
$NC = ""  # No color

# Project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = $ScriptDir

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Check-Env {
    if (-not (Test-Path ".env")) {
        if (Test-Path ".env.example") {
            Write-Warn "No .env file found, copying .env.example"
            Copy-Item ".env.example" ".env"
            Write-Warn "Please edit .env with your configuration"
        } else {
            Write-Error ".env file not found!"
            exit 1
        }
    }
    Write-Success "Environment check passed"
}

function Invoke-DockerCompose {
    param([string[]]$Args)
    docker compose $Args
}

function Cmd-Init {
    Write-Info "Initializing GeoAI infrastructure..."
    
    Check-Env
    
    Write-Info "Building Docker images..."
    Invoke-DockerCompose build
    
    Write-Info "Starting services..."
    Invoke-DockerCompose up -d
    
    Write-Info "Waiting for services to be ready..."
    Start-Sleep -Seconds 30
    
    Cmd-Status
    
    Write-Success "Infrastructure initialized!"
    Write-Info "Access services at:"
    Write-Info "  - JupyterLab: http://localhost:8888"
    Write-Info "  - MinIO:   http://localhost:9901 (console)"
    Write-Info "  - MLflow:  http://localhost:5001"
    Write-Info "  - Spark:   http://localhost:9080"
}

function Cmd-Start {
    Write-Info "Starting GeoAI services..."
    Invoke-DockerCompose up -d
    Cmd-Status
}

function Cmd-Stop {
    Write-Info "Stopping GeoAI services..."
    Invoke-DockerCompose down
    Write-Success "Services stopped"
}

function Cmd-Status {
    Write-Host ""
    Write-Host "===================================================================" -ForegroundColor Magenta
    Write-Host "  GeoAI Services Status" -ForegroundColor Magenta
    Write-Host "===================================================================" -ForegroundColor Magenta
    Invoke-DockerCompose ps
    Write-Host ""
}

function Cmd-Silver {
    Write-Info "Running Silver layer ELT pipeline..."
    
    # Check if Spark is running
    $status = docker compose ps
    if ($status -notmatch "spark-master") {
        Write-Error "Spark not running! Run: .\run_pipeline.ps1 init"
        exit 1
    }
    
    # Run the Silver pipeline
    Invoke-DockerCompose @("exec", "-T", "spark-master", "spark-submit", "--master", "spark://spark-master:7077", "--deploy-mode", "cluster", "--conf", "spark.driver.memory=4g", "--conf", "spark.executor.memory=4g", "/src/silver_spatial_transform.py") + $Args
    
    Write-Success "Silver pipeline complete!"
}

function Cmd-Gold {
    Write-Info "Running Gold layer ELT pipeline..."
    
    # Check if services running
    $status = docker compose ps
    if ($status -notmatch "spark-master") {
        Write-Error "Spark not running! Run: .\run_pipeline.ps1 init"
        exit 1
    }
    
    # Run the Gold pipeline
    Invoke-DockerCompose @("exec", "-T", "spark-master", "spark-submit", "--master", "spark://spark-master:7077", "--deploy-mode", "cluster", "--conf", "spark.driver.memory=4g", "--conf", "spark.executor.memory=4g", "/src/gold_dimensional_modeling.py") + $Args
    
    Write-Success "Gold pipeline complete!"
}

function Cmd-Test {
    Write-Info "Running all tests..."
    python -m pytest tests/ -v --tb=short 2>$null
}

function Cmd-Test-L0 {
    Write-Info "Running L0 unit tests..."
    python -m pytest tests/L0.py -v --tb=short 2>$null
}

function Cmd-Test-L1 {
    Write-Info "Running L1 integration tests..."
    python -m pytest tests/L1.py -v --tb=short 2>$null
}

function Cmd-Test-L2 {
    Write-Info "Running L2 pipeline tests..."
    python -m pytest tests/L2.py -v --tb=short 2>$null
}

function Cmd-Test-L3 {
    Write-Info "Running L3 E2E tests..."
    python -m pytest tests/L3.py -v --tb=short 2>$null
}

function Cmd-Logs {
    param([string[]]$Args)
    Invoke-DockerCompose @("logs", "-f") + $Args
}

function Cmd-Clean {
    Write-Warn "Cleaning up GeoAI resources..."
    
    # Stop services
    Invoke-DockerCompose down -v
    
    # Remove test data
    if (Test-Path "notebooks\.ipynb_checkpoints") {
        Remove-Item -Recurse -Force "notebooks\.ipynb_checkpoints"
    }
    
    Write-Success "Cleanup complete!"
}

function Cmd-Shell {
    param([string[]]$Args)
    Invoke-DockerCompose @("exec") + $Args
}

function Usage {
    Write-Host "Usage: .\run_pipeline.ps1 <command> [options]"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  init          Initialize and start infrastructure"
    Write-Host "  start        Start services"
    Write-Host "  stop         Stop services"
    Write-Host "  status       Show service status"
    Write-Host "  silver       Run Silver pipeline"
    Write-Host "  gold         Run Gold pipeline"
    Write-Host "  test         Run all tests"
    Write-Host "  test-l0      Run L0 unit tests"
    Write-Host "  test-l1      Run L1 integration tests"
    Write-Host "  test-l2      Run L2 pipeline tests"
    Write-Host "  test-l3      Run L3 E2E tests"
    Write-Host "  logs         View logs (follows)"
    Write-Host "  clean        Clean up resources"
    Write-Host "  shell        Open shell in container"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\run_pipeline.ps1 init"
    Write-Host "  .\run_pipeline.ps1 silver -InputFormat csv"
    Write-Host "  .\run_pipeline.ps1 gold -TextColumn description"
    Write-Host "  .\run_pipeline.ps1 logs -f spark-master"
}

# Main
switch ($Command) {
    "init"       { Cmd-Init @Args }
    "start"      { Cmd-Start @Args }
    "stop"       { Cmd-Stop }
    "status"     { Cmd-Status }
    "silver"     { Cmd-Silver @Args }
    "gold"       { Cmd-Gold @Args }
    "test"       { Cmd-Test }
    "test-l0"    { Cmd-Test-L0 }
    "test-l1"    { Cmd-Test-L1 }
    "test-l2"    { Cmd-Test-L2 }
    "test-l3"    { Cmd-Test-L3 }
    "logs"       { Cmd-Logs @Args }
    "clean"      { Cmd-Clean }
    "shell"      { Cmd-Shell @Args }
    "usage"      { Usage }
    default      { Usage }
}