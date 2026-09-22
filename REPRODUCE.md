# Reproduce

What this tree can run today, and what it cannot.

## Honest scope

| | In this repo | Not in this repo |
|---|---|---|
| Train / infer entrypoints (`ming/train.py`, `scripts/train_ming.py`, `scripts/infer_ming.py`) | yes | |
| Design-layer card, load plan, CPU UNI loop, Comfy node | yes | |
| `winning_formulation()` via the `particle-sliders-core` pin | yes | |
| `inclusionAI/Ming-Image-0.1-Design-Layer` weights | | on the Hub only |
| A finished GPU slider or calibrated Comfy strength | | not produced |
| RoutedMLP / GradRegularizer source | | `particle-sliders-core` only |

Do not describe a GPU run from this checkout as a finished Ming slider. Weight loading is stubbed. This scaffold does not download Hub weights.

## CPU smoke (no Hub download)

```bash
python -m pip install -r requirements.txt
python scripts/train_ming.py --help
python scripts/infer_ming.py --help
pytest -q
python scripts/train_ming.py --dummy --save_dir /tmp/ming-dummy
python scripts/infer_ming.py --dummy --load_lora models/layer-ming-design_lora --save_dir /tmp/ming-infer
```

`pytest` and `--dummy` do not import `diffusers` and do not call Hugging Face. A command without `--dummy` or `--live` raises before any download. `--live` raises `NotImplementedError` from `ming.live.load_ming_pipeline` before any download. Infer without `--load_lora` exits with an argparse error so a sample command cannot start the train loop.

The shared game is the requirements pin, not a sibling checkout variable:

```text
particle-sliders-core @ git+https://github.com/HyperGAN/particle-sliders.git@b3a100e5184d70a94f1a7562b71c10b81f009591#subdirectory=packages/particle-sliders-core
```

That commit is branch `cursor/particle-sliders-shared-core-030c` on [particle-sliders #132](https://github.com/HyperGAN/particle-sliders/pull/132). When #132 merges, move the pin to the merge commit on `main`. Do not point this product at `concept-slider-core`.

## Train CLI

```bash
python scripts/train_ming.py --dummy \
  --prompts_file configs/ming/prompts-layer.yaml \
  --config_file configs/ming/config-layer.yaml
```

Defaults, unless you override them:

```bash
python scripts/train_ming.py \
  --model_id inclusionAI/Ming-Image-0.1-Design-Layer \
  --task layer-decompose \
  --sample_steps 12 \
  --sample_guidance 2.0 \
  --resolution 1024 \
  --num_layers 6 \
  --dtype bfloat16 \
  --dummy
```

Resolution note from the model card: working-resolution bucket **1024** (recommended) or **512** for faster layer decomposition. The output preserves the input image's aspect ratio. `--dummy` writes 8×8 RGBA PNGs and records the requested resolution on the sidecar. It does not allocate a 1024 canvas.

The sidecar records `formulation.architecture` = `gmix`, `formulation.entry` = `winning_formulation`, the live stamp id, and `provisional`. Rank is `winning_formulation().spec["adapter_rank"]`.

## Infer CLI

```bash
python scripts/infer_ming.py \
  --dummy \
  --prompts_file configs/ming/prompts-layer.yaml \
  --load_lora models/layer-ming-design_lora
```

`--load_lora` skips the train loop and writes `samples/final_meta.json` plus one tiny PNG per neutral caption and the fruit-bowl control, at scales `0 / 0.25 / 0.5 / 1.0`. On `--dummy` the adapter path is recorded and not downloaded. There is no checkpoint in git.

The fruit-bowl line is a check that the slider did not move an unrelated subject. It is not a teacher. The negative prompt on each row is a canary, not a teacher.

## What a later live run will load

`ming/live.py` names the Hub folders and stops. A live loader, when it exists, should follow the companion repository:

```bash
python infer.py \
  --model inclusionAI/Ming-Image-0.1-Design-Layer \
  --task layer-decompose \
  --input-image path/to/design.png \
  --prompt path/to/layer-plan.txt \
  --resolution 1024 \
  --output-dir outputs/layers
```

Card settings for that run: 12 sampling steps, CFG 2.0, BF16, about 80 GiB VRAM. Provide an input image plus a layer plan, or omit the prompt and set a layer count. This product does not vendor that infer script.
