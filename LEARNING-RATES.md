# Explicit learning rates

Current finished-output decision and sole selection method:
[scoped learning-rate guide](docs/finished-output-learning-rates.md) and
[portable rates/case scores](docs/finished-output-rate.json). Use RAW weights,
no weight EMA, equal-case intended-effect ON−matched OFF, and reuse completed
work; no additional fresh confirmation. The option/default and CPU integration
descriptions below retain their own source-cohort scope.


The isolated native `ming_native.native.build_hndl_game` accepts `lr`,
`d_lr_mult`, and `prior_lr_mult` keyword arguments. Omitted overrides preserve
ParticleGAN 0.9.4 defaults (base 0.012, critic multiplier 1.5). Explicit base
and critic values must be finite and positive. Any non-null prior multiplier
is refused because this adapter-only game has no trainable particle bank.

The shared game owns optimizer construction and records the resolved recipe
and implementation source in its checkpoint identity. Changed rates require a
new run; old packets need their original source/dependency environment. The
stable Ming metadata CLI remains an honest scaffold with no native trainer,
so no fake CLI LR controls or common training profile are added.

This option-control change establishes installed CPU optimizer/resume conformance;
the separately captured research comparison above has its own exact scope. Historical locks, profile-history bytes, and the stable `ming`
namespace are unchanged. The new optional controls do not qualify pretrained
loading, GPU updates, a full sampler, or image quality.
