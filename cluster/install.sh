#!/usr/bin/env bash
# Run from the delivery branch checkout; no internet or root required here.
set -euo pipefail
cd "$(dirname "$0")"
sha256sum -c payload/SHA256SUMS
if [[ -e pilot.sif ]]; then
    sha256sum -c pilot.sif.sha256
else
    temp=$(mktemp .pilot.sif.XXXXXXXX)
    trap 'rm -f -- "$temp"' EXIT
    cat payload/pilot.sif.part[0-9][0-9][0-9] > "$temp"
    expected=$(awk '{print $1}' pilot.sif.sha256)
    actual=$(sha256sum "$temp" | awk '{print $1}')
    [[ $actual == "$expected" ]] || { echo 'Reassembled image checksum mismatch'; exit 1; }
    # Hard link publishes atomically without overwriting an existing image.
    ln "$temp" pilot.sif
    rm "$temp"
    trap - EXIT
fi
echo 'Package ready. Run: bash submit.sh --quick'
