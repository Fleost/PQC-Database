#!/usr/bin/env bash
set -euo pipefail

host="${1:-db}"
port="${2:-5432}"

echo "Waiting for Postgres at $host:$port..."
for i in {1..60}; do
  if command -v pg_isready >/dev/null 2>&1; then
    if pg_isready -h "$host" -p "$port" -q; then
      echo "Postgres is reachable."
      exit 0
    fi
  elif python - "$host" "$port" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])

try:
    with socket.create_connection((host, port), timeout=1):
        pass
except OSError:
    raise SystemExit(1)
raise SystemExit(0)
PY
  then
    echo "Postgres is reachable."
    exit 0
  fi
  sleep 1
done

echo "Timed out waiting for Postgres."
exit 1
