"""Hub, Comfy, and sampling constants for the Ming product.

Game math is not defined here. Call ``stamp()``.
"""

from particle_sliders import winning_formulation

# Upstream base. Not the product weight repo.
BASE_MODEL_ID = "inclusionAI/Ming-Image-0.1-Design-Layer"

# Placeholder product Hub id. The repository is not published yet.
HUB_MODEL_ID = "ntc-ai/ming-particle-sliders"
HUB_PUBLISHED = False

COMFY_NODE_CLASS = "MingParticleSlider"
COMFY_CATEGORY = "NTC/Ming"

# Layer-decomposition defaults from the Ming-Image README. Not a slider recipe.
BASE_STEPS = 12
BASE_CFG = 2.0
BASE_RESOLUTION = 1024


def stamp():
    """Return the pinned gmix winning-formulation stamp."""
    return winning_formulation()
