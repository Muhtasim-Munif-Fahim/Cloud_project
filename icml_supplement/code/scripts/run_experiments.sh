#!/bin/bash
# Launch scaling experiments on TPU VM
# Usage: ./run_experiments.sh

set -e

PROJECT="time-series-dl-483815"
# Default to v4, override for v6e
TPU_TYPE=${1:-v4}

if [ "$TPU_TYPE" == "v6e" ]; then
    ZONE="us-east5-c"
    TPU_NAME="icml2026-v6e-hero"
else
    ZONE="us-central2-b"
    TPU_NAME="icml2026-v4-primary"
fi

echo "=== Architecture Scaling Laws Experiment ==="
echo "TPU: $TPU_NAME ($ZONE)"
echo ""

# Sync code to all workers
echo "Syncing code to TPU..."
gcloud compute tpus tpu-vm scp \
    --recurse \
    ../src ../configs \
    $TPU_NAME:~/architecture-scaling-laws/ \
    --zone=$ZONE \
    --project=$PROJECT \
    --worker=all

# Install dependencies on all workers
echo "Installing dependencies..."
gcloud compute tpus tpu-vm ssh $TPU_NAME \
    --zone=$ZONE \
    --project=$PROJECT \
    --worker=all \
    --command="pip install -q flax optax datasets transformers tensorflow-cpu"

# Run experiments
echo "Starting experiments..."
gcloud compute tpus tpu-vm ssh $TPU_NAME \
    --zone=$ZONE \
    --project=$PROJECT \
    --worker=all \
    --command="cd ~/architecture-scaling-laws && python src/run_large_scale.py --scale 1b"

echo "Done!"
