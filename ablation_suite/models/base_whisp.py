"""Back-compat exports: ablation WHISP lives in `whisp_ablated`."""

from __future__ import annotations

from ablation_suite.catalog import all_run_ids

from .whisp_ablated import WHISPAblated, merge_model_spec

WHISP = WHISPAblated
VARIANT_SLUGS = all_run_ids()


def build_whisp(encoder_ckpts: object, *, spec: dict, **kwargs: object) -> WHISPAblated:
    if kwargs:
        raise TypeError("build_whisp only accepts spec=...; use catalog entries.")
    return WHISPAblated(encoder_ckpts, spec)  # type: ignore[arg-type]


__all__ = ["WHISP", "WHISPAblated", "VARIANT_SLUGS", "build_whisp", "merge_model_spec"]
