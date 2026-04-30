# WHISP ablation suite — results

## Layout

- **Catalog:** `ablation_suite/catalog.py` — all `run_id`s (`category/slug`) and merged `{spec, train}` knobs.
- **Checkpoints:** `ablation_suite/runs/<category>/<slug>/model.pt` plus `history.json`, `train_summary.json`.
- **Metrics:** `ablation_suite/logs/metrics.jsonl` — each line includes `model` (= run_id), `category`, `slug`, `cst_error`, `cl_error`, `cd_error`, `epoch`, `trainable_params`, optional `frac_train`.
- **Figures (repo root):** all PNGs and plot outputs go under **`figures/`** (see `core/figures_path.py`). Ablation plotting writes names like `ablation_suite_*.png`, `ablation_train_curves_*.png`; evaluation writes `eval_ablation_*.png`.

`cd_error` in metrics is the validation **mean-baseline** MSE for `Cd` (scalar stress test, not a Cd head).

## Re-run

```bash
bash ablation_suite/scripts/run_all.sh
```

Train the full catalog: `FULL_SUITE=1 bash ablation_suite/scripts/run_all.sh`.
