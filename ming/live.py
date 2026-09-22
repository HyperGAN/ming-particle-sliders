"""Load plan for inclusionAI/Ming-Image-0.1-Design-Layer.

Weight loading is stubbed. This module does not call ``from_pretrained``
and does not download Hub files. Companion infer code lives in
inclusionAI/Ming-Image (``--task layer-decompose``).
"""

from __future__ import annotations

from ming.defaults import (
    COMPANION_REPO,
    DTYPE,
    GUIDANCE,
    HUB_SUBFOLDERS,
    MODEL_ID,
    RESOLUTION,
    RESOLUTION_NOTE,
    SAMPLE_STEPS,
    TASK,
    VRAM_GIB,
)


def load_plan(model_id: str = MODEL_ID) -> dict:
    """Describe the Hub layout. Does not touch the network."""
    return {
        "model_id": model_id,
        "task": TASK,
        "subfolders": list(HUB_SUBFOLDERS),
        "dtype": DTYPE,
        "sample_steps": SAMPLE_STEPS,
        "guidance": GUIDANCE,
        "resolution": RESOLUTION,
        "resolution_note": RESOLUTION_NOTE,
        "vram_gib": VRAM_GIB,
        "companion_repo": COMPANION_REPO,
        "downloads": False,
        "stub": True,
    }


def load_ming_pipeline(*args: object, **kwargs: object) -> None:
    """Refuse to fetch weights. Present so a later live loader has one seam."""
    del args, kwargs
    plan = load_plan()
    folders = ", ".join(plan["subfolders"])
    raise NotImplementedError(
        "Ming weight loading is stubbed. This product does not download "
        f"{plan['model_id']} ({folders}). Companion infer code: {COMPANION_REPO} "
        "(python infer.py --task layer-decompose). CPU smoke uses --dummy."
    )
