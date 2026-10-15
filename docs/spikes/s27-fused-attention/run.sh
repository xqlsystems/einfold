#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
#
# SPDX-License-Identifier: Apache-2.0

# Runs spike S27: each measurement in its own process, under GNU time, so
# that the peak resident memory reported is that measurement's alone.
# Usage, from this directory: nix-shell -p duckdb time --run ./run.sh
set -u
BIN=$PWD/target/release/s27-fused-attention
TIME=$(type -P time)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

run() { # label, command...
  local label=$1
  shift
  local out
  out=$(cd "$WORK" && timeout 900 "$TIME" -f "maxrss_kb=%M" "$@" 2>&1)
  echo "$label | $(echo "$out" | grep -E 'ms=|Run Time|maxrss|^[0-9]+,|rror|xhausted|Killed' | tr '\n' ' ' | cut -c1-400)"
}

for L in 1024 2048 4096 8192 16384; do
  for form in window groupby; do
    # DataFusion fails at 8192 (out of memory or past 15 minutes), so stop there.
    if [ "$L" -le 8192 ]; then
      run "L=$L datafusion $form" "$BIN" df "$form" "$L"
    fi
    "$BIN" duck-sql "$form" "$L" > "$WORK/q.sql"
    run "L=$L duckdb $form" duckdb -csv -noheader :memory: -f "$WORK/q.sql"
  done
  for threads in 1 12; do
    run "L=$L fused $threads" "$BIN" fused "$L" "$threads"
  done
done
for L in 32768 65536; do
  run "L=$L fused 12" "$BIN" fused "$L" 12
done
for scale in 1 10 40; do
  "$BIN" accuracy 2048 "$scale"
done
"$BIN" edges
