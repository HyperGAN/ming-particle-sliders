# Ming capability audit

Audit baseline: HyperGAN/ming-particle-sliders main
`93cb5bfa9e6585da3dc0a89327fb56dc276e8cc5` (2026-10-17).

| Surface | Executed path | Capability |
|---|---|---|
| Train script, module, installed `ming-train` | `ming.train.main` → `ming.surfaces.stamp` → shared `winning_formulation` / `require` | Metadata dry-run; live mode exits |
| Infer script, module, installed `ming-infer` | `ming.infer.main` → same shared metadata check | Metadata dry-run; live mode exits |
| ComfyUI `MingParticleSlider.load` | Same metadata check; finite range validation; unpublished-weight guard | Always refuses native execution |

No native model forward, adapter/bank, teacher path, optimizer, MSE or GAN
update, or sampling loop exists in this product. There are no active
weights to audit. Conditioning, sign semantics, serialization, and resume
are absent. Base-model sampling constants are descriptive metadata.

The product is installable as a wheel; source-free command execution
verifies only the implemented metadata checks and refusal behavior.
Shared-package functionality and Ming-native functionality are separate
qualification gates.

Before enabling live Ming execution:

1. Implement and verify the exact native model/projection/adapter contract
   against frozen teacher responses and conditioning, including sign and
   zero-strength behavior.
2. Use the reviewed installed shared GAN API and its immutable dependency
   lineage. The requested released ParticleGAN target is 0.9.1; the shared
   owner controls its adoption. Never silently upgrade an active run.
3. Prove GAN-only updates, frozen base weights, correct active adapter
   parameters, and sampling of those trained parameters.
4. Prove product checkpoint save/load and deterministic resume through the
   shared contract, with mismatched identity and legacy artifacts rejected.
5. Obtain coordinator admission for native/GPU qualification, then publish
   reviewed weights and provenance before setting `HUB_PUBLISHED` true.

HyperGAN develop diagnostics and the optional HNDL 0.8.2 conditional
readout critic belong to the shared implementation. Ming currently has no
responses, critic dimensions, or native training path for either to act
on; adding those imports here would not provide executable reuse.
