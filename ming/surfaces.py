"""Hub, Comfy, and sampling constants for the Ming product.

Game math is not defined here. Call ``stamp()``.
"""

from .cohort import CohortError, require_cohort
from .constants import BASE_MODEL_ID

# Upstream base. Not the product weight repo.


def winning_formulation():
    try:
        from particle_sliders import winning_formulation as shared
    except ImportError as exc:
        raise CohortError('Ming shared core is unavailable; install the stable product in its own environment') from exc
    return shared()

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
    """Validate and return the pinned shared formulation metadata.

    This check does not construct a trainer or qualify a Ming backend.
    """
    require_cohort()
    loaded = winning_formulation()
    loaded.require(loaded.as_dict())
    return loaded
