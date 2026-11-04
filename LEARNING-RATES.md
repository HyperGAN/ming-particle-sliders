# Explicit learning rates

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

This change establishes installed CPU optimizer/resume conformance, not a tuned
native LR. Historical locks, profile-history bytes, and the stable `ming`
namespace are unchanged. The new optional controls do not qualify pretrained
loading, GPU updates, a full sampler, or image quality.
