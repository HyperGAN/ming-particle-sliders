# Formulation

`ming-particle-sliders` owns the Ming product surfaces: the Hub id for
slider weights, the ComfyUI node, prompt and sampling notes for
`inclusionAI/Ming-Image-0.1-Design-Layer`, and the train/infer entrypoints
in `ming/` and `scripts/`.

The game those entrypoints train is `winning_formulation()` from the pinned
[`particle-sliders-core`](https://github.com/HyperGAN/particle-sliders/tree/a119ca1ecd3d5d6c437065839d22739b04f2f4d8/packages/particle-sliders-core)
package:

```python
from particle_sliders import winning_formulation

stamp = winning_formulation()
```

`stamp.architecture_id` is `gmix` (`gmix_architecture()` / `gmix_recipe()`):
routed particles and a global-mix critic. That structure stays in the shared
core. This repository does not copy `formulation.py`, `reference.py`, or a
second `GradRegularizer`.

The parameter overlay is whatever `CURRENT_FORMULATION` the pin exports.
At commit `a119ca1ecd3d5d6c437065839d22739b04f2f4d8` that overlay is the
provisional `particle-gmix-1600-v2` record (`stamp.formulation_provisional`
is true). When ParticleGAN pull request 38 crowns a full live-leaderboard
winner (9 trained toys and all 29 bounds), the core replaces
`CURRENT_FORMULATION` and this product picks it up by bumping the pin in
`requirements.txt` and `core.lock.json`. Products keep calling
`winning_formulation()`. Copying the knobs into this repo is a fork:
`stamp.require()` raises on formulation drift.

`particlegan` (cap, RpGAN loss, particle VIC) is a transitive dependency of
`particle-sliders-core`. Do not pin ParticleGAN again here and do not vendor
its builders. `stamp.regularizer()`, `stamp.losses()`, `stamp.bridge()`, and
`stamp.critic()` are the product-facing constructors.

Formulation toys and the DSL board stay in
[HyperGAN/conceptmod](https://github.com/HyperGAN/conceptmod). A new Ming
experiment lands there; a reusable primitive lands in
[HyperGAN/particle-sliders](https://github.com/HyperGAN/particle-sliders);
only the winning recipe comes back here as a config and a docs change.

Model-surface keys (`g_lr`, `d_lr`, `particle_lr`, `adv_batch`, and the
other names on `stamp.model_surface_keys`) may be overridden through
`stamp.require()` once a Ming recipe exists. This scaffold does not publish
one. Hub ids, the Comfy class name, and the train/infer wiring stay in this
repo. See [shared-core.md](https://github.com/HyperGAN/particle-sliders/blob/a119ca1ecd3d5d6c437065839d22739b04f2f4d8/docs/shared-core.md).
