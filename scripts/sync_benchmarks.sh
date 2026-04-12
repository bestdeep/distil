#!/bin/bash
# Sync benchmark results from eval pod to local state/benchmarks/
# Run periodically (e.g., every 5 min) or after benchmark completion

EVAL_POD="root@213.13.7.110"
EVAL_PORT=6039
REMOTE_DIR="/root/benchmark_results/"
LOCAL_DIR="/home/openclaw/distillation/state/benchmarks/"

mkdir -p "$LOCAL_DIR"

# Only sync summary JSON files (not the full lm_eval output dirs)
rsync -az --timeout=10 \
  -e "ssh -p $EVAL_PORT -o ConnectTimeout=10 -o StrictHostKeyChecking=no" \
  --include='uid_*_summary.json' \
  --exclude='*' \
  "$EVAL_POD:$REMOTE_DIR" "$LOCAL_DIR" 2>/dev/null

# Check if any new summaries arrived
NEW=$(find "$LOCAL_DIR" -name 'uid_*_summary.json' -newer "$LOCAL_DIR/.last_sync" 2>/dev/null | wc -l)
touch "$LOCAL_DIR/.last_sync"
if [ "$NEW" -gt 0 ]; then
    echo "Synced $NEW new benchmark summaries"
fi
