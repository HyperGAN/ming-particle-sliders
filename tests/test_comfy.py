"""Comfy node contract. Does not import ComfyUI."""

from __future__ import annotations

from comfy_ming import (
    MingDesignLayerLora,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    diffusion_class_ok,
    validate_strength,
)


def test_node_identity():
    assert MingDesignLayerLora.CATEGORY == "NTC/Ming"
    assert MingDesignLayerLora.FUNCTION == "load"
    assert NODE_CLASS_MAPPINGS["NTCMingDesignLayerLora"] is MingDesignLayerLora
    assert "Ming" in NODE_DISPLAY_NAME_MAPPINGS["NTCMingDesignLayerLora"]


def test_strength_and_class_name():
    assert validate_strength(0) == 0.0
    assert validate_strength(2.5) == 2.5
    try:
        validate_strength(9)
    except ValueError as exc:
        assert "0 and 5" in str(exc)
    else:
        raise AssertionError("strength 9 should fail")
    assert diffusion_class_ok("MingDiffusionTransformer")
    assert not diffusion_class_ok("DiffusionTransformer")
    assert not diffusion_class_ok("Krea2Transformer")
