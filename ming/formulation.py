"""Bind this product to the shared winning formulation.

Architecture is gmix. Knobs come from ``winning_formulation()`` and stay
provisional until ParticleGAN #38 crowns a full live-leaderboard winner.
This module does not copy RoutedMLP, GradRegularizer, or the loss math.
"""

from __future__ import annotations

from typing import Any

from ming.defaults import ARCHITECTURE, FORMULATION_ENTRY

_INSTALL = (
    "ming-particle-sliders requires particle-sliders-core "
    "(import particle_sliders.winning_formulation). "
    "Install requirements.txt. The pin is HyperGAN/particle-sliders "
    "pull request 132 (branch cursor/particle-sliders-shared-core-030c) "
    "until that PR merges. Do not fall back to concept-slider-core."
)


def load_winning() -> Any:
    """Return the shared stamp. Raises if the package is missing or not gmix."""
    try:
        from particle_sliders import winning_formulation
    except ImportError as exc:
        raise ImportError(_INSTALL) from exc
    stamp = winning_formulation()
    critic = str(stamp.spec["critic"])
    if critic != ARCHITECTURE:
        raise RuntimeError(
            f"Ming architecture is {ARCHITECTURE}; "
            f"winning_formulation() returned critic {critic!r} (stamp {stamp.id})."
        )
    return stamp


def formulation_record(stamp: Any | None = None) -> dict[str, Any]:
    """JSON-safe view of the live stamp. Not a second copy of the game."""
    stamp = load_winning() if stamp is None else stamp
    losses = stamp.losses()
    edit_rms = float(stamp.spec["edit_rms_target"])
    return {
        "entry": FORMULATION_ENTRY,
        "id": stamp.id,
        "family": stamp.family,
        "architecture": ARCHITECTURE,
        "critic": str(stamp.spec["critic"]),
        "provisional": bool(stamp.provisional),
        "winner_source": stamp.winner_source,
        "related_search": stamp.related_search,
        "winner_gate": stamp.winner_gate,
        "adapter_rank": int(stamp.spec["adapter_rank"]),
        "adapter_alpha": float(stamp.spec["adapter_alpha"]),
        "noise_hold_ratio": float(stamp.spec["noise_hold_ratio"]),
        "g_lr": float(stamp.spec["g_lr"]),
        "noise_std_step0": float(stamp.noise_std_at(0, edit_rms)),
        "loss_modules": [fn.__module__ for fn in losses],
    }
