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
- `EPOCHS=40`, `RUN_DATA_EFFICIENCY=0` — optional. **Default device is CUDA** in `run_all.sh`; without an NVIDIA GPU use `DEVICE=auto`, `cpu`, or `mps`.

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

See `results_summary.md` for metrics and checkpoint layout. **All plots** (ablations, `train_whisp`, `train_autoencoders`, `test/eval_autoencoders`, eval scatters) are written under the repo **`figures/`** directory (`core.figures_path.figures_dir()`).
