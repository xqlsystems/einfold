#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
#
# SPDX-License-Identifier: Apache-2.0

# Runs M2's gate: each measurement in its own process, under GNU time, so
# that the peak resident memory reported is that measurement's alone.
# Usage, from this directory, after `cargo build --release`, with GNU time
# on the PATH (or in TIME) and a Python with the `duckdb` package (PYTHON):
#   PYTHON=python3 ./run.sh
set -u
BIN=$(realpath "${BIN:-${CARGO_TARGET_DIR:-target}/release/m2-gate}")
TIME=${TIME:-$(type -P time)}
PYTHON=${PYTHON:-python3}
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# Prints the label, the measurement, and the exit status when it isn't 0:
# 124 for the timeout, or the command's own, such as DuckDB's after it runs
# out of memory.
run() { # label, command...
  local label=$1
  shift
  local out status
  out=$(cd "$WORK" && timeout 900 "$TIME" -f "maxrss_kb=%M" "$@" 2>&1)
  status=$?
  local shown
  shown=$(echo "$out" | grep -E 'ms=|Run Time|maxrss|^[0-9]+,|rror|xhausted|Killed|panicked' | tr '\n' ' ' | cut -c1-400)
  if [ "$status" -ne 0 ]; then
    shown="$shown exit=$status"
  fi
  echo "$label | $shown"
}

for form in window groupby; do
  for l in 1024 4096 8192; do
    run "datafusion $form L=$l" "$BIN" df "$form" "$l" off
  done
  for l in 1024 4096 8192 16384 65536; do
    run "einfold $form L=$l" "$BIN" df "$form" "$l" on
  done
  for l in 1024 4096 8192 16384; do
    "$BIN" duck-sql "$form" "$l" > "$WORK/q.sql"
    run "duckdb $form L=$l" "$PYTHON" "$HERE/duckdb_run.py" "$WORK/q.sql"
  done
done
for scale in 1 10 40; do
  "$BIN" accuracy 2048 "$scale" || exit 1
done
"$BIN" edges
