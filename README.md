# Ming-Image Design-Layer particle sliders

Product repository for particle sliders on
[inclusionAI/Ming-Image-0.1-Design-Layer](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design-Layer).
The layout follows [anima-particle-sliders](https://github.com/HyperGAN/anima-particle-sliders)
and [krea2-particle-sliders](https://github.com/HyperGAN/krea2-particle-sliders):
this repo owns the Hub id, the ComfyUI node, and the train/infer entrypoints.
[particle-sliders-core](https://github.com/HyperGAN/particle-sliders/tree/4be0b5b29d2e2c741431f468d71ba8a7c884b724/packages/particle-sliders-core)
owns gmix and `winning_formulation()`.

This is a product scaffold. There is no native Ming slider trainer, adapter
runtime, checkpoint format, or sampler here. No slider weights or sample
gallery are available. A successful dry-run validates shared formulation
metadata; it does not qualify a Ming GAN implementation.

## Base model

`inclusionAI/Ming-Image-0.1-Design-Layer` decomposes a flattened design image
into transparent RGBA layers. The companion code is
[inclusionAI/Ming-Image](https://github.com/inclusionAI/Ming-Image). The model
card states an MIT license. Upstream layer-decomposition sampling is 12 steps,
CFG 2.0, and a 1024 working-resolution bucket (512 is the faster bucket), in
BF16, on a GPU with 80 GiB. Those settings belong to the base model. This
repository does not redistribute its weights.

## Shared core

Pin this commit. Do not pin `mikkel/sliders-conceptmod`, and do not vendor a
ParticleGAN checkout next to the product.

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` installs:

```text
particle-sliders-core @ git+https://github.com/HyperGAN/particle-sliders.git@4be0b5b29d2e2c741431f468d71ba8a7c884b724#subdirectory=packages/particle-sliders-core
```

`particlegan` comes in transitively from that package. Train, infer, and
Comfy call `particle_sliders.winning_formulation` and then `stamp.require()`
to check metadata. They do not construct or execute a GAN, routed adapter,
critic, or regularizer.
The formulation note is [FORMULATION.md](FORMULATION.md). The pin record is
[core.lock.json](core.lock.json).

This package baseline uses shared core 0.6.0 with released ParticleGAN
0.9.1. The historical core 0.3.0 scaffold pin is preserved separately in
the lock record. This updates metadata compatibility; it does not enable
native Ming execution or retarget any training run or checkpoint.

## Hub

Product slider weights, when they exist, belong on Hugging Face at
**`ntc-ai/ming-particle-sliders`**. That repository is **not published yet**.
Until it is, `HUB_PUBLISHED` in `ming/surfaces.py` stays false and the Comfy
node refuses to load an adapter.

| Weights | Hub id | Status |
|---|---|---|
| Base Ming-Image-0.1-Design-Layer | [`inclusionAI/Ming-Image-0.1-Design-Layer`](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design-Layer) | Upstream. Not owned by this product. |
| Ming particle sliders | `ntc-ai/ming-particle-sliders` | Placeholder. Not published. This repo will own the card, files, and provenance. |

This GitHub repository is the source for the product surfaces. It does not
host checkpoints.

## Train and infer

```bash
python scripts/train_ming.py --dry-run
python scripts/infer_ming.py --dry-run
python -m pip install '.[test]'
ming-train --dry-run
ming-infer --dry-run
python -m pytest
```

The installed commands and `python -m ming.train` / `python -m ming.infer`
also work outside a source checkout. `--dry-run` imports the stamp on CPU
and prints `architecture_id`, `formulation_id`, the base model id, the
unpublished Hub id, and `execution=scaffold-only`. Without
`--dry-run` the commands exit before any download or update. Projection
hooks for the Ming transformer are not wired.

No MSE slider training path exists to replace. Conditioning, signed slider
application, active parameter selection, sampling, save/load, and resume
remain unimplemented. See [the capability audit](docs/capability-audit.md)
for the native implementation gates.

## ComfyUI

The node is **Ming Particle Slider** (`MingParticleSlider`) under `NTC/Ming`.
See [COMFYUI.md](COMFYUI.md).

## License

Independently authored source in this repository is [MIT](LICENSE). See
[NOTICE](NOTICE). The MIT license does not relicense the inclusionAI base
weights.
