# Explicit ordered-target-frame critic

The isolated `ming-particle-sliders-native==0.2.0.dev2` distribution adds
`ming_native.ordered_frames`. It supports software response games with two to
five fixed ordered target frames. The stable package, existing singleton-frame
native API and its refusal of multiframe games retain their existing behavior.
No pretrained model, production CFG, decomposition sampler or RGBA renderer is
qualified by this change; no learning-rate value is recommended.

The new `OrderedFrameResponseAdapter` requires the explicit
`feature_profile='ming-ordered-target-frames-v1'`. Its metadata gives exports and
checkpoints a distinct critic-feature identity, so they cannot be loaded into
the older adapter profile as though their training protocols were identical.
It inherits the exact-source, frozen-base, native target-before-reference frame,
time/sign, ownership, checkpointing and cleanup contracts of the staged binding.

`ming_native.ordered_frames.build_hndl_game` requires the same explicit profile,
the new adapter, at least two prepared rows with identical finite FP32 native
`[1,C,F,H,W]` response geometry, and frozen protocol conditions. It uses the
unchanged shared core at `ddb41423` and official ParticleGAN 0.9.4, with the
existing G/D learning-rate options. There is no particle bank. The prior-rate
option continues to refuse. Omitted rates retain the shared recipe defaults.

The HNDL feature graph groups channel/frame coordinates as
`packed_index = channel * target_frames + frame` inside the critic before its
spatial convolutions. Every target frame has a distinct ordered input
coordinate. The game, targets, responses and PG input-gradient penalty retain
their native five axes. This critic feature mapping does not move frames into
the batch, average frames, or reclassify a flat hidden vector as an image. The
verified HyperGAN/HNDL builder supplies the actual graph; shared PG supplies all
losses, penalties, optimizer ownership and updates.

CPU validation covers an independent exact Torch graph oracle, gradients through
every frame, finite second-order penalty backward, explicit profile/geometry/rate
refusals, frozen-base and RNG preservation, real shared-game continuation, and
distinct numerical-export identity. All 92 installed candidate checks pass.
A separate three-process probe uses the actual upstream `DiffusionTransformer`
and `ConditionedTransformer` classes at `3515cd7`, with random tiny CPU weights,
two target frames and one reference frame. Two updates match one-plus-one in
full checkpoint state and final metrics. Explicit random initialization of the
tiny constructor's empty learned padding storage is fixture setup, not a
pretrained-weight conversion. See [the measured CPU receipt](ordered-frame-cpu-readiness-20261104.json).

The prior native product's exact lock is retained at
`profiles/history/native-dev1-pg094-ddb41423.lock.json`. Resume old packets in
their original product/source environment; this version does not migrate them.
