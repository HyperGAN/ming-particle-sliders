> **Calculated memory planning:** training **132 GiB VRAM / 64 GiB host RAM** (512×512, B=2, BF16 staged DiT; FP32 LoRA/critic); single RAW response **20 GiB VRAM / 64 GiB host RAM** (512×512). See [MEMORY.md](MEMORY.md) for checkpointed/phase variants, attention fallback and exact limits. These are conservative calculated budgets, not measured peaks.

> **Current source: core 0.15.2 / ParticleGAN 0.11.1.** This is a fresh cohort; older versioned evidence below remains historical. The [earlier adoption record](https://github.com/HyperGAN/ming-particle-sliders/blob/0405b08e33c988508bc93fb8cada1caec1fd766a/docs/pg095-adoption.md) documents the captured runtime.

# Ming-Image Design-Layer particle sliders

**Saved finished-output LR decision:** G0.012 and G0.036 tie at RAW32
(mean ON−OFF 0.04166666666666666 over four cases). No unique rate or default
change is selected. The G0.001 arm is technical missing at28/32, not zero.
See [the exact scope, options and instrument caveat](docs/finished-output-learning-rates.md)
and [portable numbers](docs/finished-output-rate.json). These original-source
research results do not qualify the package's default native/sampler paths.


See the [learning-rate controls](LEARNING-RATES.md) and [archived CPU qualification](https://github.com/HyperGAN/ming-particle-sliders/blob/b7fcee2d35848570f3e92777020685995bcf6ccb/docs/evidence/cpu-qualification.json).

Product repository for particle sliders on
[inclusionAI/Ming-Image-0.1-Design-Layer](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design-Layer).
The layout follows [anima-particle-sliders](https://github.com/HyperGAN/anima-particle-sliders)
and [krea2-particle-sliders](https://github.com/HyperGAN/krea2-particle-sliders):
this repo owns the Hub id, the ComfyUI node, and the train/infer entrypoints.
[particle-sliders-core](https://github.com/HyperGAN/particle-sliders/tree/ec1161f5d74481be3540cb0dfa63d0e8f8b54e90/packages/particle-sliders-core)
owns gmix and `winning_formulation()`.

The installed CLI and ComfyUI surfaces remain a product scaffold. An
experimental [staged velocity binding](docs/native-binding.md) supplies native
model glue to the real shared response/GAN APIs. Its isolated current candidate lane executes an actual HyperGAN develop/HNDL field critic on tiny
CPU fixtures. Pretrained Ming, multiframe training and production sampling
remain unqualified. No slider weights or sample gallery are available.

New training and evaluation must use raw current weights, as must exports and
inference. Do not add weight EMA, checkpoint averaging or an EMA comparison.
The explicit native game binding exports RAW state and exposes no host EMA
setting or EMA-selection interface.
Historical source cohorts and qualification records retain their original scope.

## Base model

`inclusionAI/Ming-Image-0.1-Design-Layer` decomposes a flattened design image
into transparent RGBA layers. The companion code is
[inclusionAI/Ming-Image](https://github.com/inclusionAI/Ming-Image). The model
card states an MIT license. Upstream layer-decomposition sampling is 12 steps,
CFG 2.0, and a 1024 working-resolution bucket (512 is the faster bucket), in
BF16, on a GPU with 80 GiB. Those settings belong to the base model. This
repository does not redistribute its weights.

## Shared core

```bash
python -m pip install .
```

This installs the stable product and the exact core dependency recorded in
[`requirements.txt`](requirements.txt). The core installs `particlegan`
0.11.1. Train, infer, and
Comfy call `particle_sliders.winning_formulation` and then `stamp.require()`
to check metadata. They do not construct or execute a GAN, routed adapter,
critic, or regularizer. Every check also verifies the installed core version,
immutable origin and locked source hashes. The wheel includes its lock records.
The formulation note is [FORMULATION.md](FORMULATION.md). The pin record is
[core.lock.json](core.lock.json).

This package uses shared core 0.15.2 with ParticleGAN 0.11.1. The historical core 0.3.0 scaffold pin is preserved separately in
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

## Installed cohorts

Install the default CLI and ComfyUI scaffold from the project root. Build the
isolated native candidate with `scripts/build_native_candidate.py` and install
the resulting wheel in a separate environment. Both packages pin core 0.15.2
and ParticleGAN 0.11.1. The [prior installation record](https://github.com/HyperGAN/ming-particle-sliders/blob/0405b08e33c988508bc93fb8cada1caec1fd766a/docs/pg094-current.md)
retains the earlier source and checkpoint boundaries.

## Scoped cached initialization

The additional `ming_native_scoped` package provides explicit cached fresh,
whole-FULL resume and RAW ordered-native response APIs. It is a separate
installed helper owner beside the exact isolated `ming_native` distribution;
the default CLI/Comfy scaffold and every existing LR/cohort remain unchanged.
Four new installed software cases passed locally with one tiny update.
Pretrained Ming, authentic native cache availability, finished RGBA rendering,
stable CLI/default use and public packaging remain unqualified.
See [the scoped API](SCOPED.md) for the actual API, exact dependency/ownership
boundary and fresh-only initialization policy.

## Scheduled strength on an owned ordered-image pipeline

Call `ming.scheduled_render.render(pipeline, scale_adapter=network.scaled,
step_gains=gains, num_inference_steps=len(gains), **inputs)` on the same
staged Ming pipeline and attached RAW adapter used for your constant
comparison. One gain applies to each native denoiser call, including the
batched guidance branches. The output retains the pipeline's ordered frames
and the returned trace records native timesteps and actual strengths.
This API requires a caller-owned
pipeline with native model admission and conditioning. The stable and native
Ming distributions use separate environments. In an existing native runtime,
load the standalone `ming/scheduled_render.py` from this checkout with
`runpy.run_path` and use its `render` function with the existing native owner.

The four-case native CRISP Layer 1 RAW256 study tied early-heavy and constant
strength at +0.04167 mean intended effect at nominal 0.5. Late-heavy scored
0. All twelve finished Layer 1 outputs passed independent integrity. Use
constant strength for this tested profile.

A separate four-seed panel compared constant, first-quarter emphasis and
middle-half emphasis at equal integrated strength 0.5. All three tied at
zero on each seed. All 16 Layer 1 images passed integrity, and three of
the four neutral images reached the scorer's ceiling. Constant remains
the best tested choice alongside the tied schedules.
[Per-case evidence](evidence/crisp-layer1-raw256-scheduled-strength.json)

### Activation gating on the ordered two-frame sampler

Use `LatentGatePlan` with the same attached RAW adapter and native sampler to
test CFP, CLABS, or a matched random control. Import it from
`ming.scheduled_render`, or take both `LatentGatePlan` and `render` from the
same `runpy.run_path` result in the isolated native runtime. Supply a built
wheel from core [`b1f780d`](https://github.com/HyperGAN/particle-sliders/commit/b1f780decc516808cfd3019d5e5b674457161f86).
Ming verifies and loads the standalone shared gate module from that wheel.
The wheel need not be installed over the checkpoint's original core.

```python
from ming.scheduled_render import LatentGatePlan, render

gate = LatentGatePlan(
    core_wheel=verified_core_wheel,
    method="cfp",
    parameter=calibrated_threshold,
    direction=fitted_ordered_direction,
    calibration_sha256=calibration_record_sha256,
)
output, trace = render(
    pipeline, scale_adapter=raw_adapter.scaled, gate_plan=gate,
    strength=calibrated_nominal_strength, num_frames_per_prompt=2,
    num_inference_steps=native_step_count, **native_inputs,
)
```

The readout takes the FP32 spatial mean of the live pre-edit latent in
`[channel, frame]` order. In the cached-reference consumer, frame zero is the
original reconstruction and frame one is the requested layer. A CFP or CLABS
direction is a unit tensor of shape `[channels, 2]` fitted for this readout.
The conditional latent produces one decision per positive-width Euler
interval. That decision scales the attached RAW edit for both ordered frames
and the same batched guidance branches as a fixed schedule. The native final
zero-width call still runs at scale zero and advances no gate state. The
trace records it separately from eligible steps and includes projections,
probabilities, effective strengths, realized firing rate and the sum of
absolute strengths weighted by the native sigma spans. Replay on the same
pipeline by dividing `trace["effective_strengths"]` by the nominal strength
and supplying the resulting `step_gains`.

For CLABS, supply the calibrated bias as `parameter`, an integer RNG `seed`,
and the fitted `mean`, `std`, `epsilon`, `setpoint`, `proportional_gain`,
`integral_gain` and `beta` in `clabs`. For `method="random"`, use the matched
target firing rate as `parameter` with an explicit seed and no direction.
Fit on matched attribute and neutral pre-edit trajectories with identical
reference, frame order, initial noise, solver and geometry. Calibrate the
threshold or bias in active RAW rollouts on separate cases using the shared
calibration protocol. A calibration rollout can return the shared
`GateUsage` with this trace's `fired` and `eligible` counts. Compare finished
held-out Layer 1 results against matched OFF, constant and random arms at their
realized budgets, with the separate output integrity check.
[Shared steering equations and workflow](https://github.com/HyperGAN/particle-sliders/blob/b1f780decc516808cfd3019d5e5b674457161f86/docs/strength-schedules.md#cfp-and-clabs)

The retained CRISP RAW256 schedule panels provide finished images, final
latents and fixed-strength traces. The prepared teacher cache provides
paired response fields at frozen input rows. Fitting and active calibration
still need matching pre-edit two-frame trajectories. The native software
fixture verifies the hook using the locked cached-reference consumer, actual
Euler scheduler and a tiny attached LoRA. Adaptive finished-image gain is
unmeasured. The retained constant qualification is unchanged.
