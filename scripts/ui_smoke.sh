#!/usr/bin/env bash
# Headless-Chrome smoke test of the landscape UI. Usage: scripts/ui_smoke.sh [extra fluxscape args...]
# Env: PORT (default: a random free port), OUT (screenshot PNG), CHROME, LOG, MODE (default synthetic),
#      WAIT_TICKS, SHOCK=TICKER:SIZE (also exercise the shock view; TICKER '*' = first active stock).
set -euo pipefail
cd "$(dirname "$0")/.."
PORT=${PORT:-$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')}
OUT=${OUT:-/tmp/fluxscape_ui.png}
CHROME=${CHROME:-google-chrome}
LOG=${LOG:-/tmp/fluxscape_serve.log}
PROFILE=$(mktemp -d /tmp/fluxscape_chrome.XXXXXX)
./build/fluxscape --serve --port "$PORT" --mode "${MODE:-synthetic}" --web web "$@" >"$LOG" 2>&1 &
PID=$!
cleanup() {
  kill "$PID" 2>/dev/null || true
  wait "$PID" 2>/dev/null || true
  pkill -f -- "--user-data-dir=$PROFILE" 2>/dev/null || true
  rm -rf "$PROFILE" 2>/dev/null || true
}
trap cleanup EXIT
# The server prints "serving http://HOST:PORT" only after it has bound the port itself, so a line naming our
# port in our own log (with our PID still alive) proves this process, not a stale one, is answering.
READY=0
for _ in $(seq 1 "${WAIT_TICKS:-240}"); do
  kill -0 "$PID" 2>/dev/null || { echo "server exited:"; cat "$LOG"; exit 1; }
  if grep -q "serving http://[^ ]*:$PORT\b" "$LOG" && curl -s "http://127.0.0.1:$PORT/api/status" | grep -q '"ready":true'; then
    READY=1; break
  fi
  sleep 0.5
done
[ "$READY" = 1 ] || { echo "server not ready on port $PORT"; exit 1; }
URL="http://127.0.0.1:$PORT/?selftest=1"
if [ -n "${SHOCK:-}" ]; then URL="$URL&shock=$SHOCK"; fi
FLAGS=(--headless=new --no-sandbox --user-data-dir="$PROFILE" --enable-unsafe-swiftshader --use-angle=swiftshader
       --window-size=1400,900 --virtual-time-budget=25000)
timeout 120 "$CHROME" "${FLAGS[@]}" --screenshot="$OUT" "$URL" >/dev/null 2>&1 || true
TITLE=$(timeout 120 "$CHROME" "${FLAGS[@]}" --dump-dom "$URL" 2>/dev/null | grep -o '<title>[^<]*</title>' || true)
echo "title: $TITLE"
echo "screenshot: $OUT"
echo "$TITLE" | grep -q 'fluxscape-ok'
