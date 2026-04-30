# WHISP ablation suite

Catalog-driven ablations for **WHISP** (Weighted Hydrodynamic Iterative Structured Propagation). Each run is `runs/<category>/<slug>/` (see `ablation_suite/catalog.py`).

## Requirements

- Repository root on `PYTHONPATH` (the run script sets this).
- PyTorch, NumPy, Matplotlib.
- Pair-encoder checkpoints under `models/encoder_*_alpha.pt` (not required for `encoder/scratch` / `baseline/no_pretrain`).
- Tensor cache `models/original_tensors.pt` (same defaults as `scripts/train_whisp.py`).

## One command (subset of catalog)

```bash
bash ablation_suite/scripts/run_all.sh
```

- `FULL_SUITE=1` — train **every** catalog entry (very long).
- `EPOCHS=40`, `DEVICE=cuda`, `RUN_DATA_EFFICIENCY=0` — optional env overrides.

## Single run

```bash
export PYTHONPATH=/path/to/SelfFoil
python ablation_suite/scripts/train_one.py --run-id physics/no_bl --device auto
python ablation_suite/scripts/evaluate.py
python ablation_suite/scripts/plot_results.py
```

Legacy baseline names still work:

```bash
python ablation_suite/scripts/train_one.py --model no_physics --device auto
```

See `results_summary.md` for metrics, checkpoint layout, and figure names (including per-category bar charts).
