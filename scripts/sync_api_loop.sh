#!/bin/bash
# Run sync 4 times per minute (every ~15s)
# Benchmark sync runs every 5th iteration (~75s)
COUNT=0
while true; do
    /home/openclaw/distillation/scripts/sync_api_state.sh
    COUNT=$((COUNT + 1))
    if [ $((COUNT % 5)) -eq 0 ]; then
        /home/openclaw/distillation/scripts/sync_benchmarks.sh 2>/dev/null
    fi
    sleep 15
done
