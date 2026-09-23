# Ming-Image Design-Layer particle sliders

Product repository for particle sliders on
[inclusionAI/Ming-Image-0.1-Design-Layer](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design-Layer).
The layout follows [anima-particle-sliders](https://github.com/HyperGAN/anima-particle-sliders)
and [krea2-particle-sliders](https://github.com/HyperGAN/krea2-particle-sliders):
this repo owns the Hub id, the ComfyUI node, and the train/infer entrypoints.
[particle-sliders-core](https://github.com/HyperGAN/particle-sliders/tree/4340e28bed388d50800c469525b460a108091da0/packages/particle-sliders-core)
owns gmix and `winning_formulation()`.

No slider weights are in this scaffold. There is no sample gallery yet.

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
particle-sliders-core @ git+https://github.com/HyperGAN/particle-sliders.git@4340e28bed388d50800c469525b460a108091da0#subdirectory=packages/particle-sliders-core
```

`particlegan` comes in transitively from that package. Train and infer call
`particle_sliders.winning_formulation` and then `stamp.require()`. They do
not reimplement routed particles, the global-mix critic, or the regularizer.
The formulation note is [FORMULATION.md](FORMULATION.md). The pin record is
[core.lock.json](core.lock.json).

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
python -m pytest
```

`--dry-run` imports the stamp on CPU and prints `architecture_id`,
`formulation_id`, the base model id, and the unpublished Hub id. Without
`--dry-run` the commands exit before any download or update. Projection
hooks for the Ming transformer are not wired.

## ComfyUI

The node is **Ming Particle Slider** (`MingParticleSlider`) under `NTC/Ming`.
See [COMFYUI.md](COMFYUI.md).

## License

Independently authored source in this repository is [MIT](LICENSE). See
[NOTICE](NOTICE). The MIT license does not relicense the inclusionAI base
weights.
