#!/usr/bin/env bash
# Weight keeper: a process that holds the model weights in VRAM, so restarts of llama-server, llama-bench and
# llama-perplexity map them (LLAMA_WEIGHT_KEEPER) instead of reading 44 GB from disk (~100 s -> ~1 s).
#   tools/mi50/keeper.sh start|stop|status
# The consumers must use the same build, split and devices (default -sm tensor; SM=layer for layer split, then pass
# the -ts of the consumers); a buffer whose layout, model files or contents differ is loaded from disk as usual.
# Restart the keeper after changing the model.
set -u
cd "$(dirname "$0")/../.."

MODEL=${MODEL:-$HOME/models/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf}
BUILD=${BUILD:-build-sub}
SM=${SM:-tensor}
MANIFEST=${LLAMA_WEIGHT_KEEPER:-/dev/shm/glmv-keeper}
LOG=${KEEPER_LOG:-/tmp/glmv-keeper.log}
PORT=${KEEPER_PORT:-8099}

keeper_pid() {
    pgrep -f "bin/llama-server .*--port $PORT" | head -1
}

case "${1:-status}" in
start)
    if [ -n "$(keeper_pid)" ] && [ -f "$MANIFEST" ]; then
        echo "keeper already running (pid $(keeper_pid))"
        exit 0
    fi
    rm -f "$MANIFEST"
    t0=$(date +%s)
    LLAMA_WEIGHT_KEEPER=$MANIFEST LLAMA_WEIGHT_KEEPER_EXPORT=1 LLAMA_LAZY_PREFETCH=0 HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1} \
        nohup "$BUILD/bin/llama-server" -m "$MODEL" -ngl 999 -sm "$SM" -fa on -fit off \
        --host 127.0.0.1 --port "$PORT" "${@:2}" > "$LOG" 2>&1 &
    for _ in $(seq 1 900); do
        [ -f "$MANIFEST" ] && break
        if [ -z "$(keeper_pid)" ]; then
            echo "keeper died, see $LOG"
            exit 1
        fi
        sleep 1
    done
    echo "keeper ready in $(( $(date +%s) - t0 )) s (pid $(keeper_pid), manifest $MANIFEST)"
    ;;
stop)
    pid=$(keeper_pid)
    [ -n "$pid" ] && kill "$pid" && echo "stopped keeper $pid"
    rm -f "$MANIFEST"
    ;;
status)
    if [ -n "$(keeper_pid)" ] && [ -f "$MANIFEST" ]; then
        echo "keeper running (pid $(keeper_pid))"
    else
        echo "no keeper"
        exit 1
    fi
    ;;
*)
    echo "usage: $0 start|stop|status [extra llama-server args]"
    exit 2
    ;;
esac
