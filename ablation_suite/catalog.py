"""
Publishable WHISP ablation catalog: categories map to runs/<category>/<slug>/.

Each entry is a JSON-serializable spec merged into WHISP + train loop.
"""

from __future__ import annotations

from typing import Any

# Default training knobs (overridden per-run where present)
_DEFAULT_TRAIN: dict[str, Any] = {
    "loss_profile": "full",
    "lambda_ns": None,
    "lambda_cl_gamma": None,
    "input_noise_std": 0.0,
    "anneal_physics": False,
    "extrapolation": False,
    "re_log_train_max": None,
    "re_log_val_min": None,
    "sparse_aoa_stride": 1,
}


def _spec(
    category: str,
    slug: str,
    *,
    train: dict[str, Any] | None = None,
    **model: Any,
) -> dict[str, Any]:
    t = {**_DEFAULT_TRAIN, **(train or {})}
    return {"category": category, "slug": slug, "train": t, **model}


def run_id(category: str, slug: str) -> str:
    return f"{category}/{slug}"


# --- Baseline (structural ablations you already had) ---
BASELINE_SPECS: dict[str, dict[str, Any]] = {
    run_id("baseline", "full"): _spec("baseline", "full"),
    run_id("baseline", "base"): _spec("baseline", "base", train={}),  # same arch as full; copy ckpt in run_all
    run_id("baseline", "no_physics"): _spec("baseline", "no_physics", use_physics=False, use_delta=False),
    run_id("baseline", "no_routing"): _spec("baseline", "no_routing", routing="mean_h"),
    run_id("baseline", "no_delta"): _spec("baseline", "no_delta", use_delta=False),
    run_id("baseline", "shared_outer"): _spec("baseline", "shared_outer", shared_outer=True),
    run_id("baseline", "single_pass"): _spec("baseline", "single_pass", n_outer=1, n_inner=1),
    run_id("baseline", "no_pretrain"): _spec("baseline", "no_pretrain", encoder_mode="scratch"),
}

# --- 6. Physics fidelity ---
PHYSICS_SPECS: dict[str, dict[str, Any]] = {
    run_id("physics", "reference"): _spec("physics", "reference", physics_fidelity="full"),
    run_id("physics", "no_bl"): _spec("physics", "no_bl", physics_fidelity="no_bl"),
    run_id("physics", "no_circulation"): _spec("physics", "no_circulation", physics_fidelity="no_circulation"),
    run_id("physics", "no_energy_term"): _spec("physics", "no_energy_term", physics_fidelity="no_energy_term"),
    run_id("physics", "shuffled_x_grid"): _spec("physics", "shuffled_x_grid", physics_fidelity="shuffled_x"),
}

# --- 7. Delta mechanism ---
DELTA_SPECS: dict[str, dict[str, Any]] = {
    run_id("delta", "reference"): _spec("delta", "reference", delta_mode="mlp"),
    run_id("delta", "identity_delta"): _spec("delta", "identity_delta", delta_mode="identity"),
    run_id("delta", "random_delta"): _spec("delta", "random_delta", delta_mode="random", random_delta_std=0.05),
    run_id("delta", "linear_delta"): _spec("delta", "linear_delta", delta_mode="linear"),
    run_id("delta", "frozen_delta"): _spec("delta", "frozen_delta", delta_mode="mlp", freeze_delta=True),
    run_id("delta", "sign_only_delta"): _spec("delta", "sign_only_delta", delta_mode="sign"),
}

# --- 8. Interaction structure ---
INTERACTION_SPECS: dict[str, dict[str, Any]] = {
    run_id("interaction", "reference"): _spec("interaction", "reference", interaction="bilinear"),
    run_id("interaction", "linear_only"): _spec("interaction", "linear_only", interaction="linear_concat"),
    run_id("interaction", "hadamard_only"): _spec("interaction", "hadamard_only", interaction="hadamard"),
    run_id("interaction", "attention_only"): _spec("interaction", "attention_only", interaction="attention"),
    run_id("interaction", "no_cross_emb"): _spec("interaction", "no_cross_emb", interaction="no_cross"),
    run_id("interaction", "full_shared_B"): _spec(
        "interaction", "full_shared_B", interaction="bilinear", shared_B_matrix=True
    ),
}

# --- 9. Iterative inference ---
ITERATION_SPECS: dict[str, dict[str, Any]] = {
    run_id("iteration", "reference"): _spec("iteration", "reference", n_outer=3, n_inner=5, outer_decay=1.0, outer_order="forward"),
    run_id("iteration", "single_pass"): _spec("iteration", "single_pass", n_outer=1, n_inner=1),
    run_id("iteration", "five_iter"): _spec("iteration", "five_iter", n_outer=5, n_inner=5),
    run_id("iteration", "decayed_iters"): _spec("iteration", "decayed_iters", n_outer=3, outer_decay=0.7),
    run_id("iteration", "reversed_iters"): _spec("iteration", "reversed_iters", n_outer=3, outer_order="reversed"),
}

# --- 10. Latent physics vector p ---
LATENT_SPECS: dict[str, dict[str, Any]] = {
    run_id("latent", "reference"): _spec("latent", "reference", latent_p_mode="full"),
    run_id("latent", "no_p"): _spec("latent", "no_p", latent_p_mode="zeros"),
    run_id("latent", "noise_p"): _spec("latent", "noise_p", latent_p_mode="noise", p_noise_std=0.1),
    run_id("latent", "shuffled_p"): _spec("latent", "shuffled_p", latent_p_mode="shuffle"),
    run_id("latent", "scalar_p"): _spec("latent", "scalar_p", latent_p_mode="scalar"),
    run_id("latent", "expanded_p"): _spec("latent", "expanded_p", latent_p_mode="full", delta_mode="expanded"),
}

# --- 11. Encoder regime ---
ENCODER_SPECS: dict[str, dict[str, Any]] = {
    run_id("encoder", "frozen"): _spec("encoder", "frozen", encoder_mode="frozen"),
    run_id("encoder", "finetune"): _spec("encoder", "finetune", encoder_mode="finetune"),
    run_id("encoder", "scratch"): _spec("encoder", "scratch", encoder_mode="scratch"),
    run_id("encoder", "partial_freeze"): _spec("encoder", "partial_freeze", encoder_mode="partial_freeze"),
    run_id("encoder", "distill_enc"): _spec("encoder", "distill_enc", encoder_mode="finetune", distill_weight=0.2),
}

# --- 12. Data regime (mostly train loop; model stays reference) ---
DATA_SPECS: dict[str, dict[str, Any]] = {
    run_id("data", "full_data"): _spec("data", "full_data", train={"frac_train": 1.0}),
    run_id("data", "low_data_10"): _spec("data", "low_data_10", train={"frac_train": 0.1}),
    run_id("data", "low_data_25"): _spec("data", "low_data_25", train={"frac_train": 0.25}),
    run_id("data", "high_noise"): _spec("data", "high_noise", train={"input_noise_std": 0.05}),
    run_id("data", "sparse_aoa"): _spec("data", "sparse_aoa", train={"sparse_aoa_stride": 3}),
    run_id("data", "extrapolation"): _spec(
        "data",
        "extrapolation",
        train={
            "extrapolation": True,
            "re_log_train_max": 6.5,
            "re_log_val_min": 6.5,
        },
    ),
}

# --- 13. Loss structure ---
LOSS_SPECS: dict[str, dict[str, Any]] = {
    run_id("loss", "reference"): _spec("loss", "reference", train={"loss_profile": "full"}),
    run_id("loss", "geo_only"): _spec("loss", "geo_only", train={"loss_profile": "geo_only"}),
    run_id("loss", "aero_only"): _spec("loss", "aero_only", train={"loss_profile": "aero_only"}),
    run_id("loss", "no_geo"): _spec("loss", "no_geo", train={"loss_profile": "no_geo"}),
    run_id("loss", "weighted_ns_high"): _spec("loss", "weighted_ns_high", train={"loss_profile": "full", "lambda_ns": 0.25}),
    run_id("loss", "annealed_physics"): _spec("loss", "annealed_physics", train={"loss_profile": "full", "anneal_physics": True}),
}

# --- 14. Numerics / discretization ---
NUMERICS_SPECS: dict[str, dict[str, Any]] = {
    run_id("numerics", "reference"): _spec("numerics", "reference", integration="trapz", physics_nx=32),
    run_id("numerics", "simpson"): _spec("numerics", "simpson", integration="simpson"),
    run_id("numerics", "coarse_grid"): _spec("numerics", "coarse_grid", physics_nx=16),
    run_id("numerics", "fine_grid"): _spec("numerics", "fine_grid", physics_nx=64),
    run_id("numerics", "adaptive_grid"): _spec("numerics", "adaptive_grid", physics_nx=32, adaptive_x_grid=True),
}

VARIANT_CATALOG: dict[str, dict[str, Any]] = {}
for d in (
    BASELINE_SPECS,
    PHYSICS_SPECS,
    DELTA_SPECS,
    INTERACTION_SPECS,
    ITERATION_SPECS,
    LATENT_SPECS,
    ENCODER_SPECS,
    DATA_SPECS,
    LOSS_SPECS,
    NUMERICS_SPECS,
):
    VARIANT_CATALOG.update(d)

# Aliases: baseline/base uses same weights file as full (handled in run_all); training uses variant full
VARIANT_CATALOG[run_id("baseline", "base")] = {**VARIANT_CATALOG[run_id("baseline", "full")], "slug": "base", "category": "baseline"}


def all_run_ids() -> tuple[str, ...]:
    return tuple(sorted(VARIANT_CATALOG.keys()))


def get_spec(run_key: str) -> dict[str, Any]:
    if run_key not in VARIANT_CATALOG:
        raise KeyError(f"Unknown run_id {run_key!r}. See ablation_suite.catalog.VARIANT_CATALOG")
    return VARIANT_CATALOG[run_key]


def category_of(run_key: str) -> str:
    return str(get_spec(run_key)["category"])


def slug_of(run_key: str) -> str:
    return str(get_spec(run_key)["slug"])


def run_path_parts(run_key: str) -> tuple[str, str]:
    return category_of(run_key), slug_of(run_key)
