# ComfyUI

This product owns the ComfyUI surface. The node class is `MingParticleSlider`,
displayed as **Ming Particle Slider**, category `NTC/Ming`. The module is
`comfy_ming.py`. A ComfyUI custom-node checkout of this repository loads it
through `__init__.py`.

The node validates metadata with `particle_sliders.winning_formulation()`
and `stamp.require()`. It does not construct or execute routed particles,
a critic, or a ParticleGAN regularizer. Comfy's package loader and an
installed `comfy_ming` module both resolve the product imports.

`ntc-ai/ming-particle-sliders` is the intended Hub id for product weights.
That repository is not published yet, so `load` stops before it touches a
checkpoint. Base Ming weights remain
`inclusionAI/Ming-Image-0.1-Design-Layer` and are not bundled here.

The strength field is a metadata range check only. It does not establish
signed adapter semantics or apply an adapter. Non-finite strengths are
rejected. Native loading and sampling need the gates in
[the capability audit](docs/capability-audit.md).

Upstream layer-decomposition sampling, from the Ming-Image README, is 12
steps, CFG 2.0, and the 1024 working-resolution bucket (512 is the faster
bucket). Those numbers are base-model settings, not a slider recipe.
