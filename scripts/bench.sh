#!/usr/bin/env bash
# Run the benchmark: the full scenario suite against the guarded and the naive agent pools of
# docker-compose.bench.yml (3 agent + sandbox pairs each, listed in scripts/bench-pool.yaml), k = 5 trials,
# CRM grading on.
#
# Usage:
#   BT_LEDGER_DIR=<dir> BT_BUDGET_USD=<cap> OPENROUTER_API_KEY=<key> scripts/bench.sh [--out runs/<id>] [test options]
#   BT_LEDGER_DIR=<dir> scripts/bench.sh --offline [test options]     # no LLM: scripted agents and personas
#
# Extra options (for example --only smoke, --as-of 2026-10-05, --dry-run, --k 1) are passed on to
# `booking-truth test`; relative paths in them are relative to the repository root. Environment:
#   BT_LEDGER_DIR       required: the cost ledger directory; every agent container appends to its own file there.
#                       On a Linux Docker host it must be writable by the container user (uid 10001).
#   BT_BUDGET_USD       required for live runs: the global spend cap checked by the harness and every agent.
#   OPENROUTER_API_KEY  exported to this script's children as BT_LLM_API_KEY; never written to a file.
#   BT_LLM_MODEL, BT_LLM_PROVIDER, BT_LLM_BASE_URL   optional, passed to the agents and the harness.
#   BT_HARDWARE         the hardware text for the manifest (default: "MacBook Air M5, 24 GB").
#   BT_BENCH_KEEP=1     leave the pool running afterwards (for `docker stats`).
#   DOCKER, UV          the docker and uv executables (default: from PATH, then ~/.local/bin).
# The harness also reads BT_* values from .env in the repository root, as every booking-truth command does, so
# --offline refuses to run while .env sets BT_LLM_API_KEY.
set -euo pipefail

: "${BT_LEDGER_DIR:?set BT_LEDGER_DIR to the cost ledger directory}"
mkdir -p "$BT_LEDGER_DIR"
BT_LEDGER_DIR="$(cd "$BT_LEDGER_DIR" && pwd)"  # absolute, relative to the caller's directory
export BT_LEDGER_DIR

cd "$(dirname "$0")/.."

tool() {
  if [[ -n "${2:-}" ]]; then echo "$2"; elif command -v "$1" >/dev/null 2>&1; then command -v "$1"
  elif [[ -x "$HOME/.local/bin/$1" ]]; then echo "$HOME/.local/bin/$1"; else echo "error: $1 not found" >&2; exit 2; fi
}
DOCKER="$(tool docker "${DOCKER:-}")"
UV="$(tool uv "${UV:-}")"
COMPOSE=("$DOCKER" compose -f docker-compose.bench.yml)

offline=0
args=()
for arg in "$@"; do
  if [[ "$arg" == "--offline" ]]; then offline=1; else args+=("$arg"); fi
done

if [[ "$offline" == 1 ]]; then
  unset OPENROUTER_API_KEY
  export BT_LLM_API_KEY=""  # compose passes this empty value to the agents, overriding .env
  # The harness reads .env too, and there an empty exported value does not override a key, so ask the
  # settings loader itself whether the harness would run offline.
  rc=0
  "$UV" run --frozen python -c \
    'import sys; from booking_truth.config import load_settings; sys.exit(0 if load_settings().offline else 3)' \
    || rc=$?
  if [[ "$rc" == 3 ]]; then
    echo "error: --offline, but .env sets BT_LLM_API_KEY; remove it from .env or run without --offline" >&2
    exit 2
  elif [[ "$rc" != 0 ]]; then
    echo "error: cannot load the BT_* settings (see above)" >&2
    exit 2
  fi
else
  if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "error: OPENROUTER_API_KEY is not set (or pass --offline for a run without an LLM)" >&2
    exit 2
  fi
  : "${BT_BUDGET_USD:?set BT_BUDGET_USD: live runs need a spend cap}"
  export BT_BUDGET_USD
  # This process and its children only: compose reads it for the agents, the harness for personas.
  export BT_LLM_API_KEY="$OPENROUTER_API_KEY"
fi

keep="${BT_BENCH_KEEP:-0}"
cleanup() {
  if [[ "$keep" != 1 ]]; then
    echo "stopping the benchmark pool..."
    "${COMPOSE[@]}" down --remove-orphans >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

echo "starting the benchmark pool (3 guarded + 3 naive pairs)..."
# Fresh containers every run, even when a pool kept with BT_BENCH_KEEP=1 is still up: no agent state carries over.
"${COMPOSE[@]}" up -d --build --force-recreate --wait --wait-timeout 180

wait_for() {  # url [header]
  local url="$1" header="${2:-}" i
  for i in $(seq 1 60); do
    if curl -fsS -o /dev/null ${header:+-H "$header"} "$url"; then return 0; fi
    sleep 1
  done
  echo "error: $url did not become healthy" >&2
  return 1
}
for port in 8201 8202 8203 8211 8212 8213; do
  wait_for "http://127.0.0.1:$port/healthz" "Authorization: Bearer ${BT_API_KEY:-dev-local-key}"
done
for port in 8301 8302 8303 8311 8312 8313; do
  wait_for "http://127.0.0.1:$port/_ui"
done

pool="scripts/bench-pool.yaml"  # the ports of docker-compose.bench.yml

pool_stats() {
  # shellcheck disable=SC2046
  "$DOCKER" stats --no-stream --format 'table {{.Name}}\t{{.MemUsage}}\t{{.CPUPerc}}' $("${COMPOSE[@]}" ps -q)
}
pool_stats

echo "running booking-truth test..."
status=0
"$UV" run --frozen booking-truth test \
  --pool "$pool" \
  --k 5 \
  --grade-crm \
  --hardware "${BT_HARDWARE:-MacBook Air M5, 24 GB}" \
  ${args[@]+"${args[@]}"} || status=$?

pool_stats
exit "$status"
