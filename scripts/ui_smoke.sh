#!/usr/bin/env bash
# Headless-Chrome smoke test of the landscape UI. Usage: scripts/ui_smoke.sh [extra fluxscape args...]
# Env: PORT, OUT (screenshot PNG), CHROME, LOG, MODE (default synthetic), WAIT_TICKS,
#      SHOCK=TICKER:SIZE (also exercise the shock view; TICKER '*' = first active stock).
set -euo pipefail
cd "$(dirname "$0")/.."
PORT=${PORT:-18765}
OUT=${OUT:-/tmp/fluxscape_ui.png}
CHROME=${CHROME:-google-chrome}
LOG=${LOG:-/tmp/fluxscape_serve.log}
PROFILE=$(mktemp -d /tmp/fluxscape_chrome.XXXXXX)
./build/fluxscape --serve --port "$PORT" --mode "${MODE:-synthetic}" --web web "$@" >"$LOG" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true; wait $PID 2>/dev/null || true; rm -rf "$PROFILE"' EXIT
for _ in $(seq 1 "${WAIT_TICKS:-240}"); do
  if curl -s "http://127.0.0.1:$PORT/api/status" | grep -q '"ready":true'; then break; fi
  sleep 0.5
done
URL="http://127.0.0.1:$PORT/?selftest=1"
if [ -n "${SHOCK:-}" ]; then URL="$URL&shock=$SHOCK"; fi
FLAGS=(--headless=new --no-sandbox --user-data-dir="$PROFILE" --enable-unsafe-swiftshader --use-angle=swiftshader
       --window-size=1400,900 --virtual-time-budget=25000)
timeout 120 "$CHROME" "${FLAGS[@]}" --screenshot="$OUT" "$URL" >/dev/null 2>&1 || true
TITLE=$(timeout 120 "$CHROME" "${FLAGS[@]}" --dump-dom "$URL" 2>/dev/null | grep -o '<title>[^<]*</title>' || true)
echo "title: $TITLE"
echo "screenshot: $OUT"
echo "$TITLE" | grep -q 'fluxscape-ok'
