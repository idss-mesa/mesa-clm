#!/usr/bin/env bash
# Wait for a pull request's checks and report whether they all passed.
#
#   scripts/wait_for_checks.sh <pr-number|branch> [--repo owner/name] [--min-checks N]
#                              [--timeout SECONDS] [--interval SECONDS]
#
# A wait is settled only when: at least --min-checks checks have registered (an empty
# list means "not started", never "passed"), none is pending or queued, the set of
# checks is identical on two consecutive polls, and none has failed.
# Exit 0: every check passed or was skipped. Exit 1: a check failed. Exit 2: timeout.
set -euo pipefail

ref="${1:?usage: wait_for_checks.sh <pr-number|branch> [--repo owner/name] [--min-checks N] [--timeout S] [--interval S]}"
shift
repo="" min_checks=3 timeout=1800 interval=30
while [ $# -gt 0 ]; do
  case "$1" in
    --repo) repo="$2"; shift 2 ;;
    --min-checks) min_checks="$2"; shift 2 ;;
    --timeout) timeout="$2"; shift 2 ;;
    --interval) interval="$2"; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 64 ;;
  esac
done
repo_args=()
[ -n "$repo" ] && repo_args=(--repo "$repo")

snapshot() {
  # one line per check: "<name>\t<state>" from gh's tab-separated table (name, state,
  # duration, url); states: pass fail pending skipping cancel. gh 2.45 has no --json here.
  gh pr checks "$ref" "${repo_args[@]}" 2>/dev/null | awk -F'\t' 'NF >= 2 {print $1 "\t" $2}' | sort || true
}

start=$(date +%s)
prev=""
while :; do
  now=$(date +%s)
  if [ $((now - start)) -ge "$timeout" ]; then
    echo "timeout after ${timeout}s waiting for checks on $ref" >&2
    [ -n "$prev" ] && printf '%s\n' "$prev" >&2
    exit 2
  fi
  cur="$(snapshot)"
  n=$(printf '%s\n' "$cur" | grep -c . || true)
  pending=$(printf '%s\n' "$cur" | grep -cE $'\t(pending|queued|in_progress)$' || true)
  if [ "$n" -ge "$min_checks" ] && [ "$pending" -eq 0 ] && [ "$cur" = "$prev" ]; then
    printf '%s\n' "$cur"
    if printf '%s\n' "$cur" | grep -qE $'\t(fail|cancel)$'; then
      echo "CI FAILED on $ref" >&2
      exit 1
    fi
    echo "CI settled on $ref: $n checks, none failed"
    exit 0
  fi
  prev="$cur"
  sleep "$interval"
done
