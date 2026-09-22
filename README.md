# Ming Particle Sliders

Product repo for particle sliders on **[inclusionAI/Ming-Image-0.1-Design-Layer](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design-Layer)**. The GitHub repo is **[ming-particle-sliders](https://github.com/HyperGAN/ming-particle-sliders)**, in the same pattern as [krea2-particle-sliders](https://github.com/HyperGAN/krea2-particle-sliders) and [anima-particle-sliders](https://github.com/HyperGAN/anima-particle-sliders).

This repo **is** the product. It owns the Hub id, the Ming train and infer surfaces, the UNI prompt cards, the sample card, and the Comfy node. The particle game is not copied here. Training calls `winning_formulation()` from **particle-sliders-core** (`import particle_sliders`), the shared package in [HyperGAN/particle-sliders](https://github.com/HyperGAN/particle-sliders) ([pull request 132](https://github.com/HyperGAN/particle-sliders/pull/132)).

This repo supports CPU smoke checks. It does not ship slider weights, and it does not download Ming weights. Weight loading is stubbed.

## Base model

| | |
|---|---|
| Hub | [`inclusionAI/Ming-Image-0.1-Design-Layer`](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design-Layer) |
| Task | image-text-to-image, `layer-decompose`, RGBA graphic design |
| Library | diffusers |
| License | MIT. Attribute **inclusionAI**. See [NOTICE](NOTICE). |
| Hub layout | `transformer/`, `vae/`, `scheduler/`, `connector/`, `mllm/`, `mlp/` |
| Companion infer | [inclusionAI/Ming-Image](https://github.com/inclusionAI/Ming-Image) (`--task layer-decompose`) |
| Lock | [`configs/ming/model.lock.json`](configs/ming/model.lock.json) |

Ming-Image-0.1-Design-Layer decomposes a flattened design image into a requested number of RGBA layers from an input image and a layer plan. The released card example is a six-layer greeting card. When `--prompt` is set, the layer count in that prompt controls the output count. Standalone outputs are RGBA PNGs.

## Sample card

Locked from the model card. Do not treat these as a tuned slider result.

| | |
|---|---|
| Resolution | **1024** recommended, or **512** for a faster bucket. Output keeps the input aspect ratio. |
| Sampling steps | **12** |
| CFG / guidance | **2.0** |
| Precision | **BF16** (`bfloat16`) |
| Hardware | one CUDA GPU with about **80 GiB** VRAM (the card's validated configuration) |

Starter yaml: [`configs/ming/config-layer.yaml`](configs/ming/config-layer.yaml) (`steps: 12`, `guidance: 2.0`, resolution note). Prompts: [`configs/ming/prompts-layer.yaml`](configs/ming/prompts-layer.yaml).

```python
# Companion repo, not this package. Weights stay on the Hub.
# python infer.py \
#   --model inclusionAI/Ming-Image-0.1-Design-Layer \
#   --task layer-decompose \
#   --resolution 1024
```

`ming/live.py` records that load plan (`transformer`, `vae`, `scheduler`, `connector`, `mllm`, `mlp`) and refuses to fetch files.

## Shared formulation

Architecture is **gmix**. The product does not restate the knobs.

```python
from particle_sliders import winning_formulation

stamp = winning_formulation()
stamp.spec["critic"]  # "gmix"
```

Today's body is the provisional stamp `particle-gmix-1600-v2` (`stamp.provisional` is true). The next winner is whatever scores best when [ParticleGAN #38](https://github.com/255BITS/ParticleGAN/pull/38) finishes on the full live leaderboard. Products keep calling `winning_formulation()` and bump the pin. This repo does not vendor `RoutedMLP` or `GradRegularizer`.

Install pin, until pull request 132 merges (branch `cursor/particle-sliders-shared-core-030c`):

```text
particle-sliders-core @ git+https://github.com/HyperGAN/particle-sliders.git@b3a100e5184d70a94f1a7562b71c10b81f009591#subdirectory=packages/particle-sliders-core
```

`concept-slider-core` is not the product base. Short note: [FORMULATION.md](FORMULATION.md).

## Train and infer

```bash
python -m pip install -r requirements.txt
python scripts/train_ming.py --help
python scripts/infer_ming.py --help
python scripts/train_ming.py --dummy
pytest -q
```

`--dummy` runs the in-repo CPU UNI loop (2 steps, tiny RGBA PNGs, no Hub download) and calls `winning_formulation()`. A run without `--dummy` or `--live` is refused before any download. `--live` calls the stub in `ming/live.py` and still does not download. Reproduction outline: [REPRODUCE.md](REPRODUCE.md).

Infer is the same code with `--load_lora`, which skips the train loop:

```bash
python scripts/infer_ming.py --dummy --load_lora models/layer-ming-design_lora
```

LoRA rank on the sidecar is `winning_formulation().spec["adapter_rank"]`. The CPU step size uses the stamp's `g_lr`. Neither value is copied into this repo as a second recipe.

## Starter concept

UNI card: bare captions, `attributes` for unused-token bookkeeping (not prefixed onto the caption), a canary `negative` that is not a teacher, and the fruit-bowl control prompt.

| Card | Plus | Files |
|---|---|---|
| Layer separation | flattened design → separate RGBA layers (six-layer card plan, plus a poster row) | [`configs/ming/prompts-layer.yaml`](configs/ming/prompts-layer.yaml) |

No schedule here is a published result.

## ComfyUI

The custom node is `comfy_ming.py` (`NTC/Ming`, class `MingDesignLayerLora`, mapping `NTCMingDesignLayerLora`). Details: [COMFYUI.md](COMFYUI.md).

## License

Code in this repository is [MIT](LICENSE). The Ming weights are not included. They are also MIT, copyright inclusionAI, and using them requires that attribution. See [NOTICE](NOTICE).
