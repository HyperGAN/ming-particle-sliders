# ComfyUI

This product owns the ComfyUI surface. The node class is `MingParticleSlider`,
displayed as **Ming Particle Slider**, category `NTC/Ming`. The module is
`comfy_ming.py`. A ComfyUI custom-node checkout of this repository loads it
through `__init__.py`.

The node reads the shared stamp with `particle_sliders.winning_formulation()`.
It does not implement routed particles, the global-mix critic, or a
ParticleGAN regularizer. Those constructors stay on the stamp
(`bridge()`, `critic()`, `regularizer()`, `losses()`).

`ntc-ai/ming-particle-sliders` is the intended Hub id for product weights.
That repository is not published yet, so `load` stops before it touches a
checkpoint. Base Ming weights remain
`inclusionAI/Ming-Image-0.1-Design-Layer` and are not bundled here.

Upstream layer-decomposition sampling, from the Ming-Image README, is 12
steps, CFG 2.0, and the 1024 working-resolution bucket (512 is the faster
bucket). Those numbers are base-model settings, not a slider recipe.
