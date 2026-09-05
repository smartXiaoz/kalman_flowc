#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONDA_EXE=${FLOWCRITIC_CONDA_EXE:-$(command -v conda || true)}
CONDA_ENV=${FLOWCRITIC_CONDA_ENV:-RLS}
PYTHON_EXE=${FLOWCRITIC_PYTHON:-}

if [[ ! -x "$CONDA_EXE" ]]; then
  echo "FLOWCRITIC_CONDA_EXE is not executable: $CONDA_EXE" >&2
  exit 127
fi
if [[ -n "$PYTHON_EXE" && ! -x "$PYTHON_EXE" ]]; then
  echo "FLOWCRITIC_PYTHON is not executable: $PYTHON_EXE" >&2
  exit 127
fi

export PATH="$(dirname "$CONDA_EXE"):${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"

args=(
  --max-rounds "${FLOWCRITIC_MAX_ROUNDS:-12}"
  --planner-retry-delay-sec "${FLOWCRITIC_PLANNER_RETRY_DELAY_SEC:-300}"
  --round-retry-delay-sec "${FLOWCRITIC_ROUND_RETRY_DELAY_SEC:-120}"
)
if [[ ${FLOWCRITIC_CONTINUE_AFTER_STAGNATION:-0} == 1 ]]; then
  args+=(--continue-after-stagnation)
fi
if [[ ${FLOWCRITIC_STOP_ON_INCOMPLETE:-0} == 1 ]]; then
  args+=(--stop-on-incomplete)
fi

if [[ -n "$PYTHON_EXE" ]]; then
  exec "$PYTHON_EXE" -u "$ROOT/scripts/continuous_tune.py" "${args[@]}"
fi
exec "$CONDA_EXE" run --no-capture-output -n "$CONDA_ENV" \
  python -u "$ROOT/scripts/continuous_tune.py" "${args[@]}"
