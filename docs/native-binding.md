# Experimental staged Ming velocity binding

`ming.native` imports the installed shared package. It never loads a model,
chooses a device, constructs a local GAN loss or starts training on import.
The existing commands and Comfy node continue to refuse native execution.

## Native contract

Source inspected: [inclusionAI/Ming-Image at 3515cd7](https://github.com/inclusionAI/Ming-Image/tree/3515cd7b1e3049fc517eed7d35020251cd06519a).
`diffusion/generator.py::ConditionedTransformer.forward` takes a list of
`[C,F,H,W]` latents, precomputed embedding lists, optional second embeddings
and preencoded reference latent lists. The wrapper retains Ming's frozen
conditioning MLP. `diffusion/pipeline.py` converts raw scheduler time to
`(1000-t)/1000`, selects target frames and negates the model prediction.
The binding performs these coordinate operations on the genuine spatial field.
It neither converts a flat vector to an image nor changes PG admission.

Only single-timestep, unipolar frozen-positive responses are supported.
Frozen neutral and positive teachers execute with adapter scale zero. The
student uses neutral conditioning and the shared scale context. Scale zero is
the frozen native forward; the context restores scales after errors.
Negative semantic axes and Ming's production CFG/normalization are pending;
guidance other than 1 and all trajectory schedules explicitly refuse.

Callers stage a frozen official `ConditionedTransformer` and attach the
installed shared `particle_sliders.native_lora.LoRANetwork`. The binding checks
that attached layer forwards belong to that model, and registers only FP32
adapter leaves through `image_lora_parameters`. Native callers must provide
`base_model_id`, an immutable 40-character weight `revision`, and the exact
`upstream_commit` in `model_identity`. Official wrapper/transformer file hashes
are checked; this is source admission, not independent verification that the
staged tensors match a supplied weight revision. CPU `fixture=True` is explicit
in export/checkpoint provenance and cannot qualify a GPU. The shared
checkpoint hook is bound on the inner native transformer.

## Exact candidate environment

Default installation retains public core 0.11.0/PG 0.9.3 and original historical
lock records. The experimental HyperGAN/HNDL game is a new environment:
core 0.12.0 joint commit `4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231`, official
PG 0.9.4, HNDL 0.10.0 and exact HyperGAN develop
`5f368a3770cf5bdae0448995c9f76e29af8049f7`. The core candidate is not public.
Its tested wheel SHA256 is
`9afad3afc1f6c653a574074dc14f0da1f1c43975c31ca45c4eac049df0b0b4ad`.
The complete admission record is
[native-candidate.lock.json](../ming/native-candidate.lock.json).

For coordinated qualification only, install the exact candidate wheel and its
`hypergan-hndl` dependencies in a separate virtual environment, then install
the product wheel with `--no-deps` and select
`cohort='native-candidate-0.12'` in `MingResponseAdapter`. This manual setup is
not a normal supported product installer: package metadata still selects
public core 0.11. The CLI stamp refuses in the candidate environment. A normal
native installer awaits shared review/publication and a qualified product cohort.

`build_hndl_game(adapter, prepared_rows, conditions, total_steps=...)` constructs
the real shared `HNDLFieldCritic` through HyperGAN's verified common builder,
then `ConditionalPG09`. Shared code owns the actual public BISPF recipe,
adversarial/penalty terms, optimizer ownership and stepping. No math is copied.
Critic conditions are caller-supplied frozen protocol features. Only
`[1,C,1,H,W]` is admitted: removing the singleton frame axis is reversible.
Multiframe/layer training needs a separately qualified shared HNDL field
binding and remains blocked.

Shared `save_adapter`/`load_adapter` produce versioned numerical adapter
artifacts, with backend/weight identity rejection before changing leaves.
Shared game packets own optimizer/random-stream resume. Neither is an ordinary
interoperable Ming LoRA, pretrained checkpoint or renderer. An old PG/source
packet must resume in its original pinned environment.
