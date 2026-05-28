#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-serve}"
shift || true
case "$MODE" in
  serve)
    exec python3 -m uvicorn pipeline.server:app --host 0.0.0.0 --port "${PORT:-8848}"
    ;;
  run)
    # headless one-shot:  entrypoint.sh run --config /path/run.json
    exec python3 -m pipeline.run_cli "$@"
    ;;
  bash)
    exec /bin/bash
    ;;
  *)
    echo "usage: entrypoint.sh {serve|run --config FILE|bash}" >&2
    exit 2
    ;;
esac
