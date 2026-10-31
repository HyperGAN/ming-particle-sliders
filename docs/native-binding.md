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

The default wheel retains public core 0.11.0/PG 0.9.3 and all original core
lock identities. The opt-in wheel is a different local version of the same
product distribution: `0.1.1+native012`. Install it in a separate environment.
It resolves exact public draft core commit
`4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231` from
[core PR148](https://github.com/HyperGAN/particle-sliders/pull/148), official
PG 0.9.4, HNDL 0.10.0 and HyperGAN develop
`5f368a3770cf5bdae0448995c9f76e29af8049f7`. PR148's exact head and public
source were independently verified; it remains an unmerged draft.
The product's default wheel and historical checkpoints are not repinned.

Build and install the candidate with ordinary dependency resolution:

```bash
python -m pip install 'build>=1,<2' 'setuptools>=84.2.1'
python scripts/build_native_candidate.py --output-dir dist-native
# In a separate environment, with its intended CPU/native Torch staged:
python -m pip install dist-native/ming_particle_sliders-0.1.1+native012-py3-none-any.whl
python -m pip check
```

The builder only creates a wheel; it does not download, install, train or publish.
Its metadata selects the exact core VCS commit, PG, HNDL and HyperGAN candidate
rather than the default core 0.11 dependency. There is no `--no-deps` install
workaround. The optional `--core-wheel` argument checks the original archived
qualification wheel against its recorded hash; a core wheel is not needed to
build from the now-public source ref. Normal VCS rebuilds can have different
ZIP hashes while matching the exact commit and all locked installed sources.

`MingResponseAdapter(..., cohort='native-candidate-0.12')` still requires an
explicit new cohort. Runtime admission checks actual PG Python source closure
against the hash-verified core release manifest, exact HG/HNDL versions and HG
VCS origin, declared ordinary/train dependency constraints, HNDL source closure,
and clean HG build provenance/builder hashes. The default CLI stamp refuses
candidate adoption; no native CLI or production renderer is enabled.
The full admission record is
[native-candidate.lock.json](../ming/native-candidate.lock.json).

Use the adapter as a context manager or call `close()` after all backward work
finishes. It restores borrowed per-module checkpoint callbacks/flags and exact
training modes, including on constructor failure. It leaves the caller's attached
LoRA in place. Concurrent bindings of the same model are rejected.

Official `patchify_and_embed` concatenates targets before reference frames;
`unpatchify` returns their full count when `multi_frame_output=True`, otherwise
one frame. The binding requires that explicit capability and checks the exact
returned frame count before selecting target frames. Frozen input encodings,
references, latents and critic features cannot require gradients. The base must
be uniformly FP32 or BF16 and frozen; adapter leaves must be FP32.

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
