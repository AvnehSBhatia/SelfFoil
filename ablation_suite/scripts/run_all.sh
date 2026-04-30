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

SUITE="${REPO}/ablation_suite"
EPOCHS="${EPOCHS:-80}"
DEVICE="${DEVICE:-cuda}"
EFF="${RUN_DATA_EFFICIENCY:-1}"

mkdir -p "${SUITE}/logs" "${SUITE}/figures"

TRAIN=(python "${SUITE}/scripts/train_one.py" --device "${DEVICE}" --epochs "${EPOCHS}" --suite-root "${SUITE}")
EVAL=(python "${SUITE}/scripts/evaluate.py" --device "${DEVICE}" --suite-root "${SUITE}")

if [[ "${FULL_SUITE:-0}" == "1" ]]; then
  read -r -a IDS <<< "$(python -c "from ablation_suite.catalog import all_run_ids; print(' '.join(all_run_ids()))")"
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
python "${SUITE}/scripts/evaluate.py" --suite-root "${SUITE}"

python "${SUITE}/scripts/plot_results.py" --suite-root "${SUITE}"
echo "Done. All figures: ${REPO}/figures/ | Metrics: ${SUITE}/logs/metrics.jsonl"
