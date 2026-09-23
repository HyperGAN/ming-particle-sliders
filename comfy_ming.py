"""ComfyUI node for Ming particle sliders.

Execution of the routed adapter stays in ``particle_sliders``. This module
only names the node and refuses to load unpublished product weights.
"""

from ming.surfaces import COMFY_CATEGORY, COMFY_NODE_CLASS, HUB_MODEL_ID, HUB_PUBLISHED, stamp


class MingParticleSlider:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "adapter_name": ("STRING", {"default": ""}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = COMFY_CATEGORY

    def load(self, model, adapter_name, strength):
        loaded = stamp()
        low, high = loaded.spec["recommended_range"]
        if strength < float(low) or strength > float(high):
            raise ValueError(
                f"strength {strength} is outside the stamp recommended range {(low, high)}"
            )
        if not HUB_PUBLISHED:
            raise RuntimeError(
                f"{COMFY_NODE_CLASS} has no weights yet. "
                f"{HUB_MODEL_ID} is not published. "
                f"architecture_id={loaded.architecture_id} "
                f"formulation_id={loaded.formulation_id}"
            )
        raise RuntimeError("Ming Comfy execution is not wired in this scaffold")


NODE_CLASS_MAPPINGS = {COMFY_NODE_CLASS: MingParticleSlider}
NODE_DISPLAY_NAME_MAPPINGS = {COMFY_NODE_CLASS: "Ming Particle Slider"}
