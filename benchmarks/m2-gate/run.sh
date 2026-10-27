#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
#
# SPDX-License-Identifier: Apache-2.0

# Runs M2's gate: each measurement in its own process, under GNU time, so
# that the peak resident memory reported is that measurement's alone.
# Usage, from this directory, after `cargo build --release`:
#   nix-shell -p duckdb time --run ./run.sh
# or with TIME and DUCKDB pointing at those binaries.
set -u
BIN=${BIN:-${CARGO_TARGET_DIR:-target}/release/m2-gate}
TIME=${TIME:-$(type -P time)}
DUCKDB=${DUCKDB:-$(type -P duckdb)}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

run() { # label, command...
  local label=$1
  shift
  local out
  out=$(cd "$WORK" && timeout 900 "$TIME" -f "maxrss_kb=%M" "$@" 2>&1)
  echo "$label | $(echo "$out" | grep -E 'ms=|Run Time|maxrss|^[0-9]+,|rror|xhausted|Killed' | tr '\n' ' ' | cut -c1-400)"
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
    run "duckdb $form L=$l" "$DUCKDB" -csv -c ".read $WORK/q.sql"
  done
done
for scale in 1 10 40; do
  "$BIN" accuracy 2048 "$scale"
done
"$BIN" edges
