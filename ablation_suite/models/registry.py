"""Run identifiers for the ablation suite (`category/slug` paths under `runs/`)."""

from __future__ import annotations

from ablation_suite.catalog import all_run_ids, get_spec, run_path_parts

RUN_SLUGS: tuple[str, ...] = all_run_ids()


def resolve_run_key(run_id: str | None, legacy_model: str | None) -> str:
    if run_id and legacy_model:
        raise ValueError("Pass only one of --run-id or --model.")
    if run_id:
        return run_id
    if legacy_model:
        if legacy_model == "base":
            return "baseline/base"
        return f"baseline/{legacy_model}"
    raise ValueError("Provide --run-id <category>/<slug> or --model <baseline_slug>.")


def spec_for_run_key(run_key: str) -> dict:
    return get_spec(run_key)


__all__ = ["RUN_SLUGS", "resolve_run_key", "spec_for_run_key", "run_path_parts"]
