# Ming-Image Design-Layer particle sliders

Product repository for particle sliders on
[inclusionAI/Ming-Image-0.1-Design-Layer](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design-Layer).
The layout follows [anima-particle-sliders](https://github.com/HyperGAN/anima-particle-sliders)
and [krea2-particle-sliders](https://github.com/HyperGAN/krea2-particle-sliders):
this repo owns the Hub id, the ComfyUI node, and the train/infer entrypoints.
[particle-sliders-core](https://github.com/HyperGAN/particle-sliders/tree/3c25f785d14073e6f627c0fa9d2297a041c79913/packages/particle-sliders-core)
owns gmix and `winning_formulation()`.

The installed CLI and ComfyUI surfaces remain a product scaffold. An
experimental [staged velocity binding](docs/native-binding.md) supplies native
model glue to the real shared response/GAN APIs. Its separate core 0.12
candidate lane executes an actual HyperGAN develop/HNDL field critic on tiny
CPU fixtures. Pretrained Ming, multiframe training and production sampling
remain unqualified. No slider weights or sample gallery are available.

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
particle-sliders-core @ git+https://github.com/HyperGAN/particle-sliders.git@3c25f785d14073e6f627c0fa9d2297a041c79913#subdirectory=packages/particle-sliders-core
```

`particlegan` comes in transitively from that package. Train, infer, and
Comfy call `particle_sliders.winning_formulation` and then `stamp.require()`
to check metadata. They do not construct or execute a GAN, routed adapter,
critic, or regularizer. Every check also verifies the installed core version,
immutable origin and locked source hashes. The wheel includes its lock records.
The formulation note is [FORMULATION.md](FORMULATION.md). The pin record is
[core.lock.json](core.lock.json).

This package baseline uses shared core 0.11.0 with released ParticleGAN
0.9.3. The historical core 0.3.0 scaffold pin is preserved separately in
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
`--dry-run` the commands exit before any download or update. These commands
do not attach Ming transformer projection hooks.

No MSE slider training path exists to replace. The experimental Python binding
owns the already attached shared native LoRA leaves and frozen numerical
conditioning. Shared APIs own GAN updates, exports and exact checkpoint restore.
CLI loading, production conditioning, signed semantic axes, multiframe
training and rendering remain pending. See [the capability audit](docs/capability-audit.md).

## ComfyUI

The node is **Ming Particle Slider** (`MingParticleSlider`) under `NTC/Ming`.
See [COMFYUI.md](COMFYUI.md).

## License

Independently authored source in this repository is [MIT](LICENSE). See
[NOTICE](NOTICE). The MIT license does not relicense the inclusionAI base
weights.

## 2026-10-31 scaffold compatibility refresh

The existing draft now selects the public shared core0.11.0 commit above.
The prior core0.6.0 package identity and receipts remain nested in core.lock.json;
this is a new metadata-only package baseline. The proposed core0.12.0/PG0.9.4
shared cohort is separate and is not selected by the default installer.
Native CLI and Comfy operations remain unwired and refuse execution; no
model weights were downloaded or pretrained model training run.

Version 0.1.1 adds staged native glue and runtime source admission without
changing that default dependency. Python 3.11 is now declared explicitly,
matching the selected core's existing minimum. No workflow publishes a package
or weights, and no model loader or download runs on import.

The optional candidate wheel can now resolve exact public draft core PR148,
HyperGAN develop and HNDL normally. It has explicit `+native012` metadata and
runs in a separate environment; see [native binding](docs/native-binding.md).
Actual PG/HG/HNDL source and declared dependency admission, checkpoint cleanup,
and exact frame-count validation are documented in
[review corrections](docs/review-fixes-20261031.md).
