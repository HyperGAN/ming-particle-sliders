# Ming-Image Design-Layer particle sliders

Explicit LR controls and qualification scope: [LEARNING-RATES.md](LEARNING-RATES.md).

Product repository for particle sliders on
[inclusionAI/Ming-Image-0.1-Design-Layer](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design-Layer).
The layout follows [anima-particle-sliders](https://github.com/HyperGAN/anima-particle-sliders)
and [krea2-particle-sliders](https://github.com/HyperGAN/krea2-particle-sliders):
this repo owns the Hub id, the ComfyUI node, and the train/infer entrypoints.
[particle-sliders-core](https://github.com/HyperGAN/particle-sliders/tree/ddb41423bcefc1fe76cd94d0f9b367353b3ea137/packages/particle-sliders-core)
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
python -m pip install .
```

This installs the stable product identity and the exact core dependency recorded
in `requirements.txt`:

```text
particle-sliders-core @ git+https://github.com/HyperGAN/particle-sliders.git@ddb41423bcefc1fe76cd94d0f9b367353b3ea137#subdirectory=packages/particle-sliders-core
```

`particlegan` comes in transitively from that package. Train, infer, and
Comfy call `particle_sliders.winning_formulation` and then `stamp.require()`
to check metadata. They do not construct or execute a GAN, routed adapter,
critic, or regularizer. Every check also verifies the installed core version,
immutable origin and locked source hashes. The wheel includes its lock records.
The formulation note is [FORMULATION.md](FORMULATION.md). The pin record is
[core.lock.json](core.lock.json).

This package baseline uses shared core 0.12.0 with released ParticleGAN
0.9.4. The historical core 0.3.0 scaffold pin is preserved separately in
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
python -m pip install '.[test]'
python scripts/train_ming.py --dry-run
python scripts/infer_ming.py --dry-run
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

## Current PG 0.9.4 upgrade

Both current distributions now use the canonical merged core at
`ddb41423bcefc1fe76cd94d0f9b367353b3ea137`: default
`ming-particle-sliders==0.2.0` and isolated
`ming-particle-sliders-native==0.2.0.dev2`. The default CLI/Comfy scaffold still
refuses live execution. The native candidate remains a separate environment
and namespace with unchanged native qualification gates. See
[current installation, source closure and historical reproduction](docs/pg094-current.md).

The previous public and native lock records and qualification receipts remain
archived. They do not qualify this new source. Actual-diff Claude review and
RETURN remain required before landing.
