"""ComfyUI node for a Ming design-layer LoRA.

Drop this repo in ``ComfyUI/custom_nodes``. The diffusion weights are
``inclusionAI/Ming-Image-0.1-Design-Layer`` (see COMFYUI.md). Adapter
files, when they exist, go in ``ComfyUI/models/loras/``. This module
does not download either.

Strength 0 returns the cloned model unchanged. A non-zero strength
loads a local safetensors LoRA and refuses files that do not look like
a diffusion LoRA.
"""

from __future__ import annotations

import math
from pathlib import Path

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}


def validate_strength(strength: float) -> float:
    value = float(strength)
    if not math.isfinite(value) or not 0.0 <= value <= 5.0:
        raise ValueError("Strength must be between 0 and 5")
    return value


def diffusion_class_ok(class_name: str) -> bool:
    return "ming" in str(class_name).lower()


def _looks_like_lora(keys: list[str]) -> bool:
    if not keys:
        return False
    markers = ("lora_a", "lora_b", "lora_up", "lora_down")
    lowered = [key.lower() for key in keys]
    return any(any(marker in key for marker in markers) for key in lowered)


class MingDesignLayerLora:
    """Apply a LoRA on the Ming design-layer diffusion model."""

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths

        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "NTC/Ming"

    def load(self, model, lora_name, strength):
        import folder_paths

        value = validate_strength(strength)
        clone = model.clone()
        if value == 0.0:
            return (clone,)
        transformer = clone.model.diffusion_model
        class_name = transformer.__class__.__name__
        if not diffusion_class_ok(class_name):
            raise ValueError(
                "Use a Ming design-layer diffusion model "
                "(inclusionAI/Ming-Image-0.1-Design-Layer), "
                f"not {class_name}"
            )
        path = Path(folder_paths.get_full_path_or_raise("loras", lora_name))
        if path.suffix != ".safetensors":
            raise ValueError(f"Ming LoRA must be a .safetensors file, got {path.name}")
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
        if not _looks_like_lora(keys):
            raise ValueError(
                f"{path.name} has no LoRA tensors. This node loads a LoRA "
                "trained by scripts/train_ming.py."
            )
        patches = getattr(clone, "patches", None)
        if isinstance(patches, dict):
            patches.setdefault("ming_design_layer_lora", []).append(
                {"path": str(path), "strength": value}
            )
        return (clone,)


NODE_CLASS_MAPPINGS = {"NTCMingDesignLayerLora": MingDesignLayerLora}
NODE_DISPLAY_NAME_MAPPINGS = {
    "NTCMingDesignLayerLora": "Ming Design-Layer LoRA (ntc-ai)",
}
