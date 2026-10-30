#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $(id -u) == 0 ]]; then
    echo 'Run as your ordinary HPC user, not root.' >&2
    exit 1
fi
work=${PQC_WORK:-/job}
results=${PQC_OUTPUT:-/results}
mkdir -p "$work/socket" "$results"
test ! -e "$work/pgdata"
export PGHOST="$work/socket" PGPORT=5432 PGUSER=pilot PGDATABASE=postgres
unset PGPASSWORD PGSERVICE PGSERVICEFILE
cleanup() {
    if [[ -f "$work/pgdata/postmaster.pid" ]]; then
        pg_ctl -D "$work/pgdata" -m fast -w stop >>"$results/postgres.log" 2>&1 || true
    fi
}
trap cleanup EXIT
initdb -D "$work/pgdata" --username=pilot --auth-local=trust --auth-host=reject --encoding=UTF8 --no-locale >"$results/initdb.log" 2>&1
pg_ctl -D "$work/pgdata" -l "$results/postgres.log" \
    -o "-c listen_addresses='' -c unix_socket_directories='$PGHOST' -c unix_socket_permissions=0700 -c shared_buffers=128MB" -w start
python -m app.hpc_pilot --datasets "${PQC_DATASETS:-/opt/pilot/datasets}" --output "$results/pilot" "$@"
