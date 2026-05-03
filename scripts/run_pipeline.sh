#!/bin/bash
# OR-Trace-Skill: End-to-end pipeline
# Usage: bash scripts/run_pipeline.sh <problems.jsonl> <output_dir> [--skip-synthesis]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="${PYTHON:-python3}"
CONFIG="${CONFIG:-$PROJECT_DIR/configs/default.yaml}"

INPUT_FILE="${1:?Usage: $0 <problems.jsonl> <output_dir> [--skip-synthesis]}"
OUTPUT_DIR="${2:?Usage: $0 <problems.jsonl> <output_dir> [--skip-synthesis]}"
SKIP_SYNTHESIS="${3:-}"

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "OR-Trace-Skill Pipeline"
echo "Input:  $INPUT_FILE"
echo "Output: $OUTPUT_DIR"
echo "Config: $CONFIG"
echo "========================================"

# Step 1: Synthesis (optional)
if [ "$SKIP_SYNTHESIS" != "--skip-synthesis" ]; then
    echo ""
    echo "[Step 1] Synthesizing problems..."
    $PYTHON "$SCRIPT_DIR/1_synthesize_problems.py" \
        "$INPUT_FILE" \
        "$PROJECT_DIR/prompts/synthesize.txt" \
        "$OUTPUT_DIR/synthetic_problems.jsonl" \
        --config "$CONFIG"

    # Merge seed + synthetic
    cat "$INPUT_FILE" "$OUTPUT_DIR/synthetic_problems.jsonl" > "$OUTPUT_DIR/all_problems.jsonl"
    TRACE_INPUT="$OUTPUT_DIR/all_problems.jsonl"
else
    echo ""
    echo "[Step 1] Skipped (--skip-synthesis)"
    TRACE_INPUT="$INPUT_FILE"
fi

# Step 2: Generate Think traces
echo ""
echo "[Step 2] Generating Think traces..."
$PYTHON "$SCRIPT_DIR/2_generate_traces.py" \
    "$TRACE_INPUT" \
    "$PROJECT_DIR/prompts/source_cot.txt" \
    "$OUTPUT_DIR/traces.jsonl" \
    --config "$CONFIG"

# Step 3: Verify answers
echo ""
echo "[Step 3] Verifying answers..."
$PYTHON "$SCRIPT_DIR/3_verify_answers.py" \
    "$OUTPUT_DIR/traces.jsonl" \
    "$OUTPUT_DIR/verified_traces.jsonl" \
    --rejected-file "$OUTPUT_DIR/rejected_traces.jsonl"

# Step 4: Distill skills
echo ""
echo "[Step 4] Distilling procedural skills..."
$PYTHON "$SCRIPT_DIR/4_distill_skills.py" \
    "$OUTPUT_DIR/verified_traces.jsonl" \
    "$PROJECT_DIR/prompts/skill_distill.txt" \
    "$OUTPUT_DIR/skills.jsonl" \
    --config "$CONFIG"

# Step 5: Retrieve and infer (on original input as test set)
echo ""
echo "[Step 5] Retrieving skills and running inference..."
$PYTHON "$SCRIPT_DIR/5_retrieve_and_infer.py" \
    "$INPUT_FILE" \
    "$OUTPUT_DIR/skills.jsonl" \
    "$PROJECT_DIR/prompts/skill_infer.txt" \
    "$OUTPUT_DIR/predictions.jsonl" \
    --config "$CONFIG"

# Step 6: Evaluate
echo ""
echo "[Step 6] Evaluating..."
$PYTHON "$SCRIPT_DIR/6_evaluate.py" \
    "$OUTPUT_DIR/predictions.jsonl" \
    --label "OR-Trace-Skill"

echo ""
echo "========================================"
echo "Pipeline complete! Results in $OUTPUT_DIR/"
echo "========================================"
