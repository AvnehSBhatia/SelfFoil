"""Single canonical directory for all project figures (PNG, etc.)."""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def figures_dir(create: bool = True) -> Path:
    """Repository-root `figures/` (same for ablations, training scripts, and tests)."""
    p = _REPO / "figures"
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def repo_root() -> Path:
    return _REPO
