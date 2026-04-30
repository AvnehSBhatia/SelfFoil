#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"
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
  candidates+=("${REPO}/.venv/bin/python" "/opt/venv/bin/python" "python" "python3")
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
  exit 1
fi
echo "Using Python: ${PYTHON_EXE}"

DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-80}"
BATCH="${BATCH:-128000}"
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
DROPOUT_START="${DROPOUT_START:-0.05}"

BASE_ARGS=(
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
  --dropout-start "${DROPOUT_START}"
  --log-every "${LOG_EVERY}"
  --compile-backend "${COMPILE_BACKEND}"
)
if [[ "${COMPILE}" == "1" ]]; then
  BASE_ARGS+=(--compile)
fi
if [[ "${WARMUP}" == "1" ]]; then
  BASE_ARGS+=(--warmup)
fi
if [[ -n "${MAX_ROWS}" ]]; then
  BASE_ARGS+=(--max-rows "${MAX_ROWS}")
fi

echo "==> [1/2] Train baseline WHISP (encoders must already exist)"
"${PYTHON_EXE}" scripts/train_whisp.py "${BASE_ARGS[@]}"

echo "==> [2/2] Train/evaluate ablation suite"
export PYTHON_BIN="${PYTHON_EXE}"
export DEVICE EPOCHS BATCH LR LR_SCHEDULE LR_MIN_FACTOR AUX_RAMP_EPOCHS
export EARLY_STOP_PATIENCE EARLY_STOP_MIN_DELTA EARLY_STOP_MONITOR COMPILE COMPILE_BACKEND WARMUP LOG_EVERY
export DROPOUT_START
if [[ -n "${MAX_ROWS}" ]]; then
  export MAX_ROWS
fi
bash ablation_suite/scripts/run_all.sh

echo "Done."
