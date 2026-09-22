# Formulation

`ming-particle-sliders` owns **model integration and recipes** for sliders on `inclusionAI/Ming-Image-0.1-Design-Layer`: the Hub id, the six subfolders (`transformer`, `vae`, `scheduler`, `connector`, `mllm`, `mlp`), the design-layer sample card (12 steps, guidance 2.0, resolution 1024 or 512, BF16, about 80 GiB), the UNI prompt card under `configs/ming/`, the train/infer entrypoints in `ming/` and `scripts/`, and the Comfy node `MingDesignLayerLora`.

The **architecture is gmix**. The **formulation** is whatever `winning_formulation()` returns from `particle-sliders-core`:

```python
from particle_sliders import winning_formulation

stamp = winning_formulation()
```

Knobs on that stamp are **provisional** until [ParticleGAN #38](https://github.com/255BITS/ParticleGAN/pull/38) crowns a winner on the full live leaderboard (9 trained toys and all 29 live bounds). Partial wins do not count. Today's body is `particle-gmix-1600-v2` and `stamp.provisional` is true. This product does not freeze those knobs in a local recipe. When the crown lands, bump the pin in `requirements.txt`. Keep calling `winning_formulation()`.

`ming/train.py` refuses a stamp whose `critic` is not `gmix`.

Formulation toys, ablations, and pass/fail gates stay in [HyperGAN/conceptmod](https://github.com/HyperGAN/conceptmod). Do not add them here, and do not put product experiments in ParticleGAN.

ParticleGAN is the core primitive. This product does not fork those primitives and does not vendor a second copy of `RoutedMLP` or `GradRegularizer`. The dummy CPU path calls `winning_formulation()`, `stamp.noise_std_at`, and `stamp.losses()`. It does not reimplement them.

The install pin is `particle-sliders-core` from HyperGAN/particle-sliders, subdirectory `packages/particle-sliders-core`. Until [pull request 132](https://github.com/HyperGAN/particle-sliders/pull/132) merges, the pin is commit `b3a100e5184d70a94f1a7562b71c10b81f009591` on branch `cursor/particle-sliders-shared-core-030c`. Do not use `concept-slider-core` as the product base.

Sibling products keep their own Hub ids and sample cards: [krea2-particle-sliders](https://github.com/HyperGAN/krea2-particle-sliders), [anima-particle-sliders](https://github.com/HyperGAN/anima-particle-sliders). They should call the same `winning_formulation()` entry.
