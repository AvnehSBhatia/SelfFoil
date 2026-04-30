#!/usr/bin/env bash
# Train a configurable slice of the catalog, evaluate, optional data-efficiency, plots.
# Use FULL_SUITE=1 to train every entry in ablation_suite.catalog (long).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
export PYTHONPATH="${REPO}"

if [[ -f "${REPO}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${REPO}/.venv/bin/activate"
fi

pick_python_with_torch() {
  local candidates=()
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    candidates+=("${PYTHON_BIN}")
  fi
  candidates+=(
    "${REPO}/.venv/bin/python"
    "/opt/venv/bin/python"
    "python"
    "python3"
  )
  for py in "${candidates[@]}"; do
    if command -v "${py}" >/dev/null 2>&1; then
      if "${py}" -c "import torch" >/dev/null 2>&1; then
        echo "${py}"
        return 0
      fi
    fi
  done
  return 1
}

if ! PYTHON_EXE="$(pick_python_with_torch)"; then
  echo "ERROR: Could not find a Python interpreter with torch installed."
  echo "Tried: PYTHON_BIN, ${REPO}/.venv/bin/python, /opt/venv/bin/python, python, python3"
  exit 1
fi
echo "Using Python: ${PYTHON_EXE}"

SUITE="${REPO}/ablation_suite"
EPOCHS="${EPOCHS:-80}"
DEVICE="${DEVICE:-cuda}"
EFF="${RUN_DATA_EFFICIENCY:-1}"
BATCH="${BATCH:-64000}"
LR="${LR:-3e-4}"
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
LR_MIN_FACTOR="${LR_MIN_FACTOR:-0.1}"
AUX_RAMP_EPOCHS="${AUX_RAMP_EPOCHS:-10}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-8}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-5e-4}"
EARLY_STOP_MONITOR="${EARLY_STOP_MONITOR:-val_geo}"
COMPILE="${COMPILE:-1}"
COMPILE_BACKEND="${COMPILE_BACKEND:-auto}"
WARMUP="${WARMUP:-1}"
LOG_EVERY="${LOG_EVERY:-20}"
MAX_ROWS="${MAX_ROWS:-}"

mkdir -p "${SUITE}/logs" "${SUITE}/figures"

TRAIN=(
  "${PYTHON_EXE}" "${SUITE}/scripts/train_one.py"
  --device "${DEVICE}"
  --epochs "${EPOCHS}"
  --batch "${BATCH}"
  --lr "${LR}"
  --lr-schedule "${LR_SCHEDULE}"
  --lr-min-factor "${LR_MIN_FACTOR}"
  --aux-ramp-epochs "${AUX_RAMP_EPOCHS}"
  --early-stop-patience "${EARLY_STOP_PATIENCE}"
  --early-stop-min-delta "${EARLY_STOP_MIN_DELTA}"
  --early-stop-monitor "${EARLY_STOP_MONITOR}"
  --log-every "${LOG_EVERY}"
  --compile-backend "${COMPILE_BACKEND}"
  --suite-root "${SUITE}"
)
if [[ "${COMPILE}" == "1" ]]; then
  TRAIN+=(--compile)
fi
if [[ "${WARMUP}" == "1" ]]; then
  TRAIN+=(--warmup)
fi
if [[ -n "${MAX_ROWS}" ]]; then
  TRAIN+=(--max-rows "${MAX_ROWS}")
fi
EVAL=("${PYTHON_EXE}" "${SUITE}/scripts/evaluate.py" --device "${DEVICE}" --suite-root "${SUITE}")

if [[ "${FULL_SUITE:-0}" == "1" ]]; then
  read -r -a IDS <<< "$("${PYTHON_EXE}" -c "from ablation_suite.catalog import all_run_ids; print(' '.join(all_run_ids()))")"
else
  IDS=(
    baseline/no_physics
    baseline/no_routing
    baseline/no_delta
    physics/reference
    physics/no_bl
    physics/no_circulation
    delta/reference
    delta/identity_delta
    interaction/reference
    interaction/linear_only
    iteration/reference
    iteration/single_pass
    encoder/frozen
    encoder/finetune
    loss/reference
    loss/geo_only
    numerics/reference
    numerics/simpson
  )
fi

for rid in "${IDS[@]}"; do
  cat="${rid%%/*}"
  slug="${rid#*/}"
  mkdir -p "${SUITE}/runs/${cat}/${slug}"
  "${TRAIN[@]}" --run-id "${rid}"
done

if [[ "${EFF}" == "1" ]]; then
  : > "${SUITE}/logs/efficiency.jsonl"
  for fr in 0.25 0.5 0.75 1.0; do
    "${TRAIN[@]}" --run-id baseline/full --frac-train "${fr}" --epochs "${EPOCHS}"
    "${EVAL[@]}" --run-ids baseline/full --metrics-path "${SUITE}/logs/efficiency.jsonl"
  done
fi
"${TRAIN[@]}" --run-id baseline/full --frac-train 0.8 --epochs "${EPOCHS}"

if [[ -f "${SUITE}/runs/baseline/full/model.pt" ]]; then
  mkdir -p "${SUITE}/runs/baseline/base"
  cp -f "${SUITE}/runs/baseline/full/model.pt" "${SUITE}/runs/baseline/base/model.pt"
  for f in history.json train_summary.json; do
    if [[ -f "${SUITE}/runs/baseline/full/${f}" ]]; then
      cp -f "${SUITE}/runs/baseline/full/${f}" "${SUITE}/runs/baseline/base/${f}"
    fi
  done
fi

: > "${SUITE}/logs/metrics.jsonl"
"${PYTHON_EXE}" "${SUITE}/scripts/evaluate.py" --suite-root "${SUITE}"

"${PYTHON_EXE}" "${SUITE}/scripts/plot_results.py" --suite-root "${SUITE}"
echo "Done. All figures: ${REPO}/figures/ | Metrics: ${SUITE}/logs/metrics.jsonl"
