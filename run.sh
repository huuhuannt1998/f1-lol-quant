#!/usr/bin/env bash
# One-command runner for your Mac (VS Code terminal):
#   bash run.sh          set up .venv if needed, fetch real data, forecast, write data/*.json
#   bash run.sh serve    preview the site at http://localhost:8000
#   bash run.sh test     run the test suite
# Every run's output is saved to cache/last_run.log so Claude can read it.
set -uo pipefail
# Unbuffered stdout: print() lines land in the log in order, not after stderr.
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")"
mkdir -p cache
LOG=cache/last_run.log

if [ ! -x .venv/bin/python ]; then
  echo "== creating .venv"
  python3 -m venv .venv || { echo "python3 not found: install it from python.org or 'brew install python'"; exit 1; }
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt pytest
  if [ "$(uname)" = "Darwin" ] && ! .venv/bin/python -c "import lightgbm" 2>/dev/null; then
    echo "!! LightGBM needs OpenMP on macOS. Optional: 'brew install libomp' (the pipeline runs without it)."
  fi
fi

case "${1:-update}" in
  serve)
    echo "== open http://localhost:8000  (Ctrl+C to stop)"
    .venv/bin/python -m http.server 8000 ;;
  test)
    .venv/bin/python -m pytest -q 2>&1 | tee "$LOG" ;;
  demo)
    .venv/bin/python -m pipeline demo 2>&1 | tee "$LOG" ;;
  *)
    { echo "== run started $(date -u +%FT%TZ) on $(uname -sm), $(.venv/bin/python --version)"
      .venv/bin/python -m pipeline update "${@:2}"
      echo "== exit code $?"; } 2>&1 | tee "$LOG"
    echo
    echo "Done. Preview with: bash run.sh serve   (log saved to $LOG)" ;;
esac
