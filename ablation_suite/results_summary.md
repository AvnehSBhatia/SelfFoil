# WHISP ablation suite — results

## Layout

- **Catalog:** `ablation_suite/catalog.py` — all `run_id`s (`category/slug`) and merged `{spec, train}` knobs.
- **Checkpoints:** `runs/<category>/<slug>/model.pt` plus `history.json`, `train_summary.json`.
- **Metrics:** `logs/metrics.jsonl` — each line includes `model` (= run_id), `category`, `slug`, `cst_error`, `cl_error`, `cd_error`, `epoch`, `trainable_params`, optional `frac_train`.
- **Figures (300 dpi PNG):** `figures/ablation_barplot.png` (all runs), `figures/<category>_bar.png` (per-axis), `convergence_curves.png`, `error_vs_model_size.png`, `suite_manifest.txt`.

## Axes covered (high level)

| Category      | Role |
|---------------|------|
| `baseline`    | Structural toggles (full physics stack, routing, delta coupling, depth, encoders). |
| `physics`     | Fidelity / stress (no BL march, no circulation channel, no skin-friction term in residual, shuffled chord grid). |
| `delta`       | Iterative correction (identity, Gaussian noise, linear map, frozen MLP, sign-only STE). |
| `interaction` | Tensor interaction (concat MLP, Hadamard, attention-style mixing, no cross-emb, shared B). |
| `iteration`   | Outer depth / order / decay. |
| `latent`      | Corruptions on the latent path into Δ (noise, shuffle, scalar collapse, widened MLP input). |
| `encoder`     | Frozen / finetune / scratch / partial weight-freeze / distillation toward frozen teacher embeddings. |
| `data`        | Fraction of airfoils, input noise, α subsampling on train indices, low-Re train → high-Re val. |
| `loss`        | geo-only, aero-only (Cl), no-geo, stronger NS weight, annealed physics weights. |
| `numerics`    | Trapezoid vs Simpson integrals, chord resolution, Chebyshev-style non-uniform stations. |

`cd_error` in metrics is still the **validation mean baseline** MSE for `Cd` (scalar stress test, not a Cd head).

## Re-run

```bash
bash ablation_suite/scripts/run_all.sh
```

Train everything in the catalog: `FULL_SUITE=1 bash ablation_suite/scripts/run_all.sh`.
