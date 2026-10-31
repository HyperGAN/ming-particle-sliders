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

## Isolated candidate distribution

The stable wheel remains `ming-particle-sliders==0.1.1`, namespace `ming`, with
its public core 0.11.0/PG 0.9.3 cohort and stable CLI/Comfy modules. It ships
only `ming/core.lock.json` and its own identity declaration. Passing a native
profile to this installed product refuses; it cannot select another lock.

The native experiment is a different distribution and namespace:
`ming-particle-sliders-native==0.1.1.dev12`, importing `ming_native`.
It contains no `ming` files, `comfy_ming` module or console entry points.
It ships its own `ming_native/core.lock.json` and identity declaration, with
exact public core4abc VCS/PG/HNDL/HyperGAN dependencies. It cannot satisfy a
stable `ming-particle-sliders==0.1.1` requirement. Normal environments must
remain separate because the two exact shared-core versions differ; both
products explicitly refuse unsafe force-co-installation.

Build and normally resolve the native artifact in a separate environment:

```bash
python -m pip install 'build>=1,<2' 'setuptools>=84.2.1'
python scripts/build_native_candidate.py --output-dir dist-native
# In a new environment with the intended CPU Torch:
python -m pip install dist-native/ming_particle_sliders_native-0.1.1.dev12-py3-none-any.whl
python -m pip check
```

Use `from ming_native.native import MingResponseAdapter, build_hndl_game`.
The default `cohort=None` uses only this distribution's declared native cohort;
an explicit incompatible profile refuses. The source build input is
[profiles/native-candidate.lock.json](../profiles/native-candidate.lock.json).
Core PR148's exact4abc source was verified public; the artifact remains an
experimental cohort, independent of whether that PR later merges.
PG version comes from this lock when building dependencies.

Runtime admission binds installed distribution name/version/namespace, package
resources, the single lock digest, and declared exact dependencies before
validating runtime sources. Actual PG Python closure, exact HG/HNDL versions and
HG VCS origin, ordinary/train constraints including direct Git requirements,
HNDL closure and clean HG builder provenance are checked. Missing dependencies,
malformed provenance, unexpected Git URLs and source mismatches raise actionable
`CohortError` exceptions. No unknown Git dependency is accepted by version alone.

Use a context manager or call `close()` after backward work finishes. Ownership
leases cover the entire module tree, including inner transformers shared by
different wrappers. Cleanup attempts every mode/callback/flag/ownership action
and aggregates failures. Original setup/body exceptions remain visible alongside
cleanup failures. Failed cleanup blocks re-borrowing and can be retried after
the caller fixes the failing state restoration.

Only unipolar guidance1 is admitted. `negative=None` is supported and unused;
neutral and positive encodings remain required and frozen. No objective or
strength default changes. Official target-before-reference ordering and exact
frame capability/count validation remain unchanged by this packaging fix.

The optional `--core-wheel` builder argument validates the original archived
qualification wheel; it is not required for normal exact public VCS resolution.
The builder only creates artifacts, with no install, training or publication.

Candidate tests run with `python scripts/run_candidate_tests.py`. The runner
copies the exact native/artifact test files byte-for-byte outside the stable
Comfy root package, then tests the installed `ming_native` API. It introduces
no `ming` alias and no stable Comfy module into the candidate environment.

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
