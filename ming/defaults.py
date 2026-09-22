"""Pinned Ming-Image-0.1-Design-Layer defaults. Importing this hits no network."""

from __future__ import annotations

MODEL_ID = "inclusionAI/Ming-Image-0.1-Design-Layer"
TASK = "layer-decompose"
PIPELINE_TAG = "image-text-to-image"
LIBRARY = "diffusers"
LICENSE_NAME = "mit"
ATTRIBUTION = "inclusionAI"

HUB_SUBFOLDERS = (
    "transformer",
    "vae",
    "scheduler",
    "connector",
    "mllm",
    "mlp",
)
TRANSFORMER_CLASS = "DiffusionTransformer"
VAE_CLASS = "AutoencoderKLQwenImage"
SCHEDULER_CLASS = "FlowMatchEulerDiscreteScheduler"
SCHEDULER_SHIFT = 6.0
MLLM_ARCHITECTURE = "BailingMM2NativeForConditionalGeneration"
CONNECTOR_ARCHITECTURE = "Qwen2ForCausalLM"

# Model card. 1024 is the recommended bucket; 512 is the faster bucket.
# Output preserves the input image aspect ratio.
SAMPLE_STEPS = 12
GUIDANCE = 2.0
RESOLUTION = 1024
RESOLUTION_FAST = 512
RESOLUTION_NOTE = (
    "Working-resolution bucket 1024 (recommended) or 512 for faster layer "
    "decomposition. The output preserves the input image's aspect ratio."
)
DTYPE = "bfloat16"
VRAM_GIB = 80
NUM_LAYERS_EXAMPLE = 6

HUB_CARD = "https://huggingface.co/inclusionAI/Ming-Image-0.1-Design-Layer"
HUB_LICENSE = (
    "https://huggingface.co/inclusionAI/Ming-Image-0.1-Design-Layer/blob/main/LICENSE"
)
COMPANION_REPO = "https://github.com/inclusionAI/Ming-Image"
PARTICLE_SLIDERS_REPO = "https://github.com/HyperGAN/particle-sliders"
PARTICLE_SLIDERS_PR = "https://github.com/HyperGAN/particle-sliders/pull/132"
PARTICLE_SLIDERS_BRANCH = "cursor/particle-sliders-shared-core-030c"
PARTICLE_SLIDERS_REVISION = "b3a100e5184d70a94f1a7562b71c10b81f009591"
PARTICLEGAN_PR = "https://github.com/255BITS/ParticleGAN/pull/38"
CONCEPTMOD_REPO = "https://github.com/HyperGAN/conceptmod"
KREA2_PRODUCT = "https://github.com/HyperGAN/krea2-particle-sliders"
ANIMA_PRODUCT = "https://github.com/HyperGAN/anima-particle-sliders"

ARCHITECTURE = "gmix"
FORMULATION_ENTRY = "winning_formulation"
PRODUCT_NAME = "ming-particle-sliders"

CONTROL_PROMPT = "a bowl of fruit on a table"
HOLD_WEIGHT = 0.1
SAMPLE_SCALES = (0.0, 0.25, 0.5, 1.0)
DEFAULT_PROMPTS = "configs/ming/prompts-layer.yaml"
DEFAULT_CONFIG = "configs/ming/config-layer.yaml"
