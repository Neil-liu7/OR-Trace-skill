#!/bin/bash
set -e

export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export no_proxy="127.0.0.1,localhost"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "============================================"
echo " Running all benchmark pipelines sequentially"
echo "============================================"

echo ""
echo ">>> [1/2] NL4OPT Pipeline"
echo ""
bash scripts/run_nl4opt_pipeline.sh

echo ""
echo ">>> [2/2] OptiBench Pipeline"
echo ""
bash scripts/run_optibench_pipeline.sh

echo ""
echo "============================================"
echo " ALL PIPELINES DONE"
echo "============================================"
