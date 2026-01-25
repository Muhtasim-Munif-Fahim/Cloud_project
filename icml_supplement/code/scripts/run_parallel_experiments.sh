#!/bin/bash
# =============================================================================
# Multi-TPU Parallel Execution Script for Depth Delusion Experiments
# =============================================================================
#
# This script launches experiments across 3 TPU types simultaneously:
# - v4-32-1: 1B sweep (5 models) + 3B partial (2 models)
# - v4-32-2: 3B partial (3 models)
# - v6e-64:  7B sweep (2 models)
#
# Total: 12 models
#
# Usage:
#   ./run_parallel_experiments.sh [--dry-run]
#
# =============================================================================

set -e

# Configuration
PROJECT_DIR="~/architecture-scaling-laws"
OUTPUT_BASE="gs://icml2026-scaling-data/large_scale_results"
CONFIG_FILE="${PROJECT_DIR}/configs/large_scale_experiments.yaml"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "  DEPTH DELUSION - LARGE SCALE EXPERIMENTS   "
echo "  12 models across 3 TPU types               "

echo "=============================================="
echo ""

# Check for dry run
DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo -e "${YELLOW}[DRY RUN MODE]${NC} Commands will be printed but not executed"
    echo ""
fi

# Function to run command (or print in dry-run mode)
run_or_print() {
    if $DRY_RUN; then
        echo -e "${YELLOW}[DRY RUN]${NC} $@"
    else
        eval "$@"
    fi
}

# =============================================================================
# TPU v4-32: 1B sweep + 2x3B (runs on first TPU)
# =============================================================================

launch_v4_32() {
    echo -e "${GREEN}[v4-32]${NC} Launching 1B sweep (5 models) + 3B partial (2 models)"
    
    local OUTPUT_DIR="${OUTPUT_BASE}/v4_32_${TIMESTAMP}"
    
    # 1B models (sequential, ~48h total)
    for model in "1B_shallow" "1B_medium" "1B_deep" "1B_vdeep" "1B_extreme"; do
        echo -e "${GREEN}[v4-32]${NC} Training $model..."
        run_or_print "python ${PROJECT_DIR}/src/run_large_scale.py \
            --config ${CONFIG_FILE} \
            --scale 1b \
            --model $model \
            --output-dir ${OUTPUT_DIR}/1b \
            --gcs-backup ${OUTPUT_DIR}/1b/"
    done
    
    # 3B models (2 models, ~172h total)
    for model in "3B_shallow" "3B_medium"; do
        echo -e "${GREEN}[v4-32]${NC} Training $model..."
        run_or_print "python ${PROJECT_DIR}/src/run_large_scale.py \
            --config ${CONFIG_FILE} \
            --scale 3b \
            --model $model \
            --output-dir ${OUTPUT_DIR}/3b \
            --gcs-backup ${OUTPUT_DIR}/3b/"
    done
    
    echo -e "${GREEN}[v4-32]${NC} Completed all assigned models"
}

# =============================================================================
# TPU v4-32 (Secondary): 3B partial (runs on second TPU)
# =============================================================================

launch_v4_32_secondary() {
    echo -e "${GREEN}[v4-32-secondary]${NC} Launching 3B partial (3 models)"
    
    local OUTPUT_DIR="${OUTPUT_BASE}/v4_32_sec_${TIMESTAMP}"
    
    # 3B models (3 models)
    for model in "3B_deep" "3B_vdeep" "3B_extreme"; do
        echo -e "${GREEN}[v4-32-secondary]${NC} Training $model..."
        run_or_print "python ${PROJECT_DIR}/src/run_large_scale.py \
            --config ${CONFIG_FILE} \
            --scale 3b \
            --model $model \
            --output-dir ${OUTPUT_DIR}/3b \
            --gcs-backup ${OUTPUT_DIR}/3b/"
    done
    
    echo -e "${GREEN}[v4-32-secondary]${NC} Completed all assigned models"
}

# =============================================================================
# TPU v6e-64: 7B sweep (runs on third TPU)
# =============================================================================

launch_v6e_64() {
    echo -e "${GREEN}[v6e-64]${NC} Launching 7B sweep (2 models)"
    
    local OUTPUT_DIR="${OUTPUT_BASE}/v6e_64_${TIMESTAMP}"
    
    # 7B models (2 models, ~272h total)
    for model in "7B_optimal" "7B_deep"; do
        echo -e "${GREEN}[v6e-64]${NC} Training $model (HEADLINE DEMO)..."
        run_or_print "python ${PROJECT_DIR}/src/run_large_scale.py \
            --config ${CONFIG_FILE} \
            --scale 7b \
            --model $model \
            --output-dir ${OUTPUT_DIR}/7b \
            --gcs-backup ${OUTPUT_DIR}/7b/"
    done
    
    echo -e "${GREEN}[v6e-64]${NC} Completed all assigned models"
}

# =============================================================================
# Main Execution
# =============================================================================

# Detect which TPU we're on and run appropriate experiments
TPU_TYPE="${TPU_TYPE:-auto}"

if [[ "$TPU_TYPE" == "auto" ]]; then
    # Auto-detect based on device count
    DEVICE_COUNT=$(python3 -c "import jax; print(jax.device_count())" 2>/dev/null || echo "0")
    
    if [[ "$DEVICE_COUNT" == "32" ]]; then
        # Could be v4-32 or v5e-32, check chip type
        if python3 -c "import jax; print('v4' in str(jax.devices()[0]))" 2>/dev/null | grep -q "True"; then
            TPU_TYPE="v4_32"
        else
            # Default to v4 if unsure (no v5e)
            TPU_TYPE="v4_32"
        fi
    elif [[ "$DEVICE_COUNT" == "64" ]]; then
        TPU_TYPE="v6e_64"
    fi
fi

echo -e "Detected TPU type: ${GREEN}${TPU_TYPE}${NC}"
echo ""

case $TPU_TYPE in
    "v4_32")
        launch_v4_32
        ;;
    "v4_32_secondary")
        launch_v4_32_secondary
        ;;
    "v6e_64")
        launch_v6e_64
        ;;
    "all")
        # For local testing - run smallest models only
        echo "Running all in sequence (testing mode)..."
        launch_v4_32 &
        launch_v4_32_secondary &
        launch_v6e_64 &
        wait
        ;;
    *)
        echo -e "${RED}Error: Unknown TPU type: $TPU_TYPE${NC}"
        echo "Set TPU_TYPE environment variable to: v4_32, v5e_32, or v6e_64"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "  EXPERIMENT LAUNCH COMPLETE                 "
echo "=============================================="
