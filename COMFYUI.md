# ComfyUI: Ming design-layer

Use a Ming design-layer diffusion model. This repo does not vendor the inclusionAI weights (on the order of 80 GiB of BF16 parameters on the card's validated GPU), and it does not vendor a trained LoRA.

Install this repository as a custom node and restart ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/HyperGAN/ming-particle-sliders.git
python -m pip install -r ming-particle-sliders/requirements.txt
```

The node module is `comfy_ming.py`, registered from `__init__.py`.

## Node

| | |
|---|---|
| Class | `MingDesignLayerLora` |
| Mapping | `NTCMingDesignLayerLora` |
| Category | `NTC/Ming` |
| Display | Ming Design-Layer LoRA (ntc-ai) |

Strength 0 returns the cloned model with no adapter. Strength must be between 0 and 5. The node checks that the diffusion class name contains `ming` and that the chosen file is a safetensors LoRA (it looks for `lora_A` / `lora_B` style keys).

## Sample settings

These are the model-card settings, not a calibrated slider:

- **12 steps**
- **CFG 2.0**
- Resolution bucket **1024** (recommended) or **512**. The output preserves the input aspect ratio.
- **BF16**
- About **80 GiB** VRAM for the validated configuration

The diffusers class on the Hub transformer is `DiffusionTransformer`. A Comfy wrapper has to expose `ming` in its class name before this node will attach a LoRA. The Hub folders are `transformer/`, `vae/`, `scheduler/`, `connector/`, `mllm/`, and `mlp/`. Attribute **inclusionAI**. The weights are MIT and are not in this repo.

## Where adapters come from

Nothing in this repo is a trained LoRA yet. `python scripts/train_ming.py --dummy` writes a CPU sidecar, not a Comfy file. When a LoRA exists, put it in `ComfyUI/models/loras/` and select it on **Ming Design-Layer LoRA**.

There is no calibrated strength 1.0, because no slider has been trained on a GPU and scored. Strength 0 leaves the base checkpoint unchanged.
