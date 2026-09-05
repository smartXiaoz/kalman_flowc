#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TEMPLATE="$ROOT/systemd/flowcritic-paper-tuning.service.in"
SERVICE_NAME=flowcritic-paper-tuning.service
USER_SYSTEMD_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
DESTINATION="$USER_SYSTEMD_DIR/$SERVICE_NAME"
ACTION=${1:---install-only}
CONDA_EXE=${FLOWCRITIC_CONDA_EXE:-$(command -v conda || true)}

if [[ ! -f "$TEMPLATE" ]]; then
  echo "service template not found: $TEMPLATE" >&2
  exit 2
fi
if [[ ! -x "$CONDA_EXE" ]]; then
  echo "conda executable not found; set FLOWCRITIC_CONDA_EXE" >&2
  exit 2
fi
PYTHON_EXE=${FLOWCRITIC_PYTHON:-$("$CONDA_EXE" run -n "${FLOWCRITIC_CONDA_ENV:-RLS}" python -c 'import sys; print(sys.executable)' | awk 'NF {last=$0} END {print last}')}
if [[ ! -x "$PYTHON_EXE" ]]; then
  echo "RLS python executable not found; set FLOWCRITIC_PYTHON" >&2
  exit 2
fi
if [[ "$ROOT" == *$'\n'* || "$ROOT" == *'|'* || "$CONDA_EXE" == *$'\n'* || "$CONDA_EXE" == *'|'* || "$PYTHON_EXE" == *$'\n'* || "$PYTHON_EXE" == *'|'* ]]; then
  echo "unsupported path characters in ROOT, FLOWCRITIC_CONDA_EXE, or FLOWCRITIC_PYTHON" >&2
  exit 2
fi

mkdir -p "$USER_SYSTEMD_DIR"
sed \
  -e "s|@ROOT@|$ROOT|g" \
  -e "s|@CONDA_EXE@|$CONDA_EXE|g" \
  -e "s|@PYTHON_EXE@|$PYTHON_EXE|g" \
  "$TEMPLATE" >"$DESTINATION"
systemctl --user daemon-reload

case "$ACTION" in
  --install-only)
    echo "installed $DESTINATION"
    echo "enable later with: systemctl --user enable --now $SERVICE_NAME"
    ;;
  --enable-now)
    systemctl --user enable --now "$SERVICE_NAME"
    systemctl --user --no-pager --full status "$SERVICE_NAME" || true
    ;;
  --disable)
    systemctl --user disable --now "$SERVICE_NAME"
    ;;
  *)
    echo "usage: $0 [--install-only|--enable-now|--disable]" >&2
    exit 2
    ;;
esac
