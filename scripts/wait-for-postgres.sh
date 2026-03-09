#!/usr/bin/env bash
set -euo pipefail

host="${1:-db}"
port="${2:-5432}"

echo "Waiting for Postgres at $host:$port..."
for i in {1..60}; do
  if pg_isready -h "$host" -p "$port" -q; then
    echo "Postgres is reachable."
    exit 0
  fi
  sleep 1
done

echo "Timed out waiting for Postgres."
exit 1
