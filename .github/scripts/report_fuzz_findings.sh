#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
#
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from ddx's script of the same name (https://github.com/xqlsystems/ddx,
# the XQL Systems project for automatic differentiation of SQL queries), so the
# two projects' soaks share one log format and one triage.
#
# report_fuzz_findings.sh — triage the nightly simulation-soak logs and, if any
# property failures were found, file (or update) a single GitHub issue with a
# reproducible repro. Invoked by .github/workflows/nightly-fuzz.yml.
#
# Anti-spam policy: at most ONE open `fuzz-finding` issue at a time. If one is
# already open, this appends the new run's findings as a comment instead of
# opening a duplicate — a persistent bug accumulates recurrences on one issue,
# and a fresh issue only opens once a human has triaged and closed the last one.
#
# Usage: report_fuzz_findings.sh <log-dir>
#   Reads every `soak-*.log` under <log-dir>. Requires `gh` authenticated with
#   `issues: write` (the workflow's GITHUB_TOKEN).
#
# Env (provided by the workflow, all optional — used only to enrich the body):
#   GITHUB_RUN_ID, GITHUB_SHA, GITHUB_SERVER_URL, GITHUB_REPOSITORY

set -euo pipefail

LOG_DIR=${1:?"usage: report_fuzz_findings.sh <log-dir>"}

shopt -s nullglob globstar
# With globstar, `**/` also matches the top level, so a log there would be
# listed twice and its failures counted twice; keep each path once.
logs=()
declare -A seen=()
for f in "$LOG_DIR"/**/soak-*.log "$LOG_DIR"/soak-*.log; do
  [ -n "${seen[$f]:-}" ] && continue
  seen[$f]=1
  logs+=("$f")
done
if [ ${#logs[@]} -eq 0 ]; then
  echo "No soak-*.log files found under $LOG_DIR; nothing to report."
  exit 0
fi

# ---------------------------------------------------------------------------
# Tally failures and collect a bounded, readable set of failure blocks.
# ---------------------------------------------------------------------------
total=0
blocks=""       # up to MAX_BLOCKS failure blocks, across all regions
env_line=""     # rustc / os / sha, recorded by the soak step
region_summary=""
MAX_BLOCKS=8
block_count=0

for log in "${logs[@]}"; do
  [ -f "$log" ] || continue
  n=$(grep -c '^FAILURE (seed' "$log" 2>/dev/null || true)
  n=${n:-0}
  total=$((total + n))

  # The soak step records one `ENV ...` line at the top of each log.
  if [ -z "$env_line" ]; then
    env_line=$(grep -m1 '^ENV ' "$log" 2>/dev/null || true)
  fi

  # A soak that hung or was cut short has no `SOAK done` line, and its
  # failures must still be reported: a missing line is no error.
  base=$(grep -m1 '^SOAK start' "$log" 2>/dev/null | sed -n 's/.*base=\([0-9]*\).*/\1/p' || true)
  iters=$(grep -m1 '^SOAK done' "$log" 2>/dev/null | sed -n 's/.*iters=\([0-9]*\).*/\1/p' || true)
  region_summary+="- \`$(basename "$log")\`: base=${base:-?}, iters=${iters:-?}, failures=${n}"$'\n'

  if [ "$n" -gt 0 ] && [ "$block_count" -lt "$MAX_BLOCKS" ]; then
    # Extract each FAILURE block: the header plus its report lines, stopping at
    # the next heartbeat / failure / summary line.
    while IFS= read -r line; do
      case "$line" in
        "===BLOCK===")
          block_count=$((block_count + 1))
          [ "$block_count" -ge "$MAX_BLOCKS" ] && break
          ;;
        *) blocks+="$line"$'\n' ;;
      esac
    # Each block is cut to its first 60 lines of at most 300 characters; the
    # samples as a whole are capped below, after the loop.
    done < <(awk '
      /^FAILURE \(seed/ { if (inblk) print "===BLOCK==="; inblk=1; n=0; print; next }
      inblk == 1 {
        if ($0 ~ /^HEARTBEAT/ || $0 ~ /^SOAK/ || $0 == "") { inblk=0; print "===BLOCK==="; next }
        if (++n <= 60) print substr($0, 1, 300)
        else if (n == 61) print "  …"
      }
      END { if (inblk) print "===BLOCK===" }
    ' "$log")
  fi

  # A soak that ended without its summary crashed, say on a segfault or a
  # panic outside a case, whether or not it recorded failures first: that's a
  # finding too. The watchdog's hang reports, `[stall]` or a failure whose
  # shrinking hung, already explain a missing summary.
  if ! grep -q '^SOAK done' "$log" 2>/dev/null &&
    ! grep -qE '^\[stall\]|a try at shrinking it ran for more than' "$log" 2>/dev/null; then
    total=$((total + 1))
    if [ "$block_count" -lt "$MAX_BLOCKS" ]; then
      blocks+="CRASH ($(basename "$log")): the soak ended without its summary. The log's last lines:"$'\n'
      blocks+="$(tail -n 20 "$log" | cut -c1-300)"$'\n'
      block_count=$((block_count + 1))
    fi
  fi
done

# GitHub rejects an issue body or comment over 65,536 characters, and the
# blocks alone could reach 8 × 60 × 300. Keep the samples well inside it.
if [ "${#blocks}" -gt 40000 ]; then
  blocks="${blocks:0:40000}"$'\n'"… (cut; the full logs are attached to the run)"$'\n'
fi

if [ "$total" -eq 0 ]; then
  echo "Soak completed with 0 property failures. Nothing to file."
  exit 0
fi

echo "Found $total property failure(s) across ${#logs[@]} region log(s); preparing issue."

# ---------------------------------------------------------------------------
# A ready-to-paste repro from the first failing seed: a case follows from its
# seed alone, so EINFOLD_SEED replays it.
# ---------------------------------------------------------------------------
first_seed=$(printf '%s' "$blocks" | sed -n 's/^FAILURE (seed=\([0-9]*\).*/\1/p' | head -1)

# Each soak writes a `REPRO <command>` line with a `{seed}` placeholder, so the
# repro names the soak that failed. The first log with a failure supplies it; a
# log without one falls back to the simulation's own command.
repro=""
for log in "${logs[@]}"; do
  if grep -q '^FAILURE (seed' "$log" 2>/dev/null; then
    repro=$(grep -m1 '^REPRO ' "$log" 2>/dev/null | sed 's/^REPRO //' || true)
    break
  fi
done
if [ -z "$repro" ]; then
  repro='EINFOLD_SEED={seed} cargo test -p einfold-datafusion --test simulation --release -- --nocapture simulation'
fi
# A crash has no seed: the soak logs a case's seed only when it fails.
if [ -n "$first_seed" ]; then
  repro=${repro//\{seed\}/$first_seed}
fi

run_url="${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY:-xqlsystems/einfold}/actions/runs/${GITHUB_RUN_ID:-local}"
date_utc=$(date -u '+%Y-%m-%d %H:%M UTC')

body_file=$(mktemp)
{
  echo "**Nightly simulation-soak fuzz found ${total} property failure(s).**"
  echo
  echo "- Run: [${GITHUB_RUN_ID:-local}](${run_url})"
  echo "- Commit: \`${GITHUB_SHA:-unknown}\`"
  echo "- When: ${date_utc}"
  [ -n "$env_line" ] && echo "- Build: \`${env_line#ENV }\`"
  echo
  echo "### Reproduce"
  echo
  if [ -n "$first_seed" ]; then
    echo "Each failure names its seed; \`EINFOLD_SEED=<seed>\` replays that case"
    echo "alone and prints its SQL and tables. On the same platform as the build"
    echo "above:"
    echo
    echo '```bash'
    echo "$repro"
    echo '```'
  else
    echo "Only crashes were found, and a crash has no seed: the soak logs a"
    echo "case's seed only when it fails. A region's last \`HEARTBEAT\` gives the"
    echo "cases it had run, from its base, before the crash."
  fi
  echo
  echo "The full per-region logs are attached to the workflow run as artifacts."
  echo
  echo "### Per-region summary"
  echo
  printf '%s\n' "$region_summary"
  echo "### Failure samples (up to ${MAX_BLOCKS})"
  echo
  echo '```'
  printf '%s' "$blocks"
  echo '```'
  echo
  echo "> Triage note: each failure starts with its property's tag (\`[rows]\`,"
  echo "> \`[error]\`, \`[schema]\`, …; see simulation.rs). The host is DataFusion"
  echo "> itself, so a failure may be DataFusion's: check whether it reproduces"
  echo "> without einfold before blaming einfold, and file DataFusion's with the"
  echo "> \`upstream\` label."
  echo
  echo "<!-- fuzz-run: ${GITHUB_RUN_ID:-local} -->"
} > "$body_file"

# ---------------------------------------------------------------------------
# Ensure labels exist (idempotent), then dedup: comment on the single open
# fuzz-finding issue if one exists, else create a new one.
# ---------------------------------------------------------------------------
gh label create fuzz-finding --color B60205 \
  --description "Found by the nightly simulation-soak fuzz" 2>/dev/null || true
gh label create needs-triage --color FBCA04 \
  --description "Awaiting human triage" 2>/dev/null || true

existing=$(gh issue list --state open --label fuzz-finding \
  --json number --jq '.[0].number // empty' 2>/dev/null || true)

if [ -n "$existing" ]; then
  echo "Open fuzz-finding issue #${existing} exists; commenting instead of filing a duplicate."
  {
    echo "### Recurred — run ${GITHUB_RUN_ID:-local} (${date_utc})"
    echo
    cat "$body_file"
  } | gh issue comment "$existing" --body-file -
  echo "Commented on #${existing}."
else
  title="Nightly fuzz: ${total} property failure(s) in the simulation soaks"
  url=$(gh issue create --title "$title" \
    --label fuzz-finding --label needs-triage \
    --body-file "$body_file")
  echo "Filed new issue: $url"
fi

rm -f "$body_file"
