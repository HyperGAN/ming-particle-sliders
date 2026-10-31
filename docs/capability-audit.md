# Ming capability audit

Baseline main: `93cb5bfa9e6585da3dc0a89327fb56dc276e8cc5`. Existing draft
baseline: `17950bc9d1a8b3f4bcaf144793b695292efee2a7`. Updated 2026-10-31.

| Surface | Executed path | Qualification |
|---|---|---|
| Train/infer scripts, modules and installed commands | Exact public core 0.11/PG 0.9.3 source admission, shared stamp/require | Metadata dry-run; live commands refuse |
| ComfyUI node | Same admission/stamp, finite strength range, unpublished-weight guard | Native execution refuses, including strength zero |
| Experimental `MingResponseAdapter` | Staged native wrapper → native 5D latent velocity; shared LoRA ownership and frozen response preparation | Tiny CPU fixtures; native pretrained behavior unrun |
| Explicit `build_hndl_game` candidate | Core 0.12/PG 0.9.4 → verified HyperGAN develop/HNDL 0.10 field critic → shared BISPF game/optimizers/checkpoint | Actual installed graphs and tiny CPU game; singleton frame only |
| Adapter exports | Shared safetensors save/load with exact backend/frozen-model identity | Tiny CPU roundtrip and rejection before mutation |
| Candidate checkpoints | Shared exact game packets, optimizers and streams | Tiny CPU deterministic continuation; old PG cohort refused without mutation |

The candidate uses joint local core commit
`4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231`, HyperGAN develop
`5f368a3770cf5bdae0448995c9f76e29af8049f7`, and HNDL 0.10.0. It is now the public unmerged draft PR148
and does not replace the default dependency or qualify a historical checkpoint.
The packaged [candidate lock](../profiles/native-candidate.lock.json) records its
exact tested wheel and source hashes. Old public and historical identities
are preserved.

Before enabling native product commands:

1. Qualify the exact pretrained transformer/conditioner weights and source
   closure, input encoding, reference latents, and actual target layers.
2. Qualify multiframe/layer output. The shared HNDL field critic accepts a 2D
   field; this binding only removes a singleton frame axis and rejects multiple
   frames. It does not flatten or relabel a wide vector.
3. Qualify native CFG, scheduler stepping, VAE RGBA decoding and layer ordering.
   The binding rejects trajectories and guidance other than 1; no generic
   Euler renderer substitutes for Ming's production sampler.
4. Review and qualify the exact public shared draft before enabling native
   product execution. A separate experimental candidate wheel now resolves
   its exact public VCS dependencies normally. Keep core 0.11 and historical
   checkpoint environments isolated; no automatic migration is supported.
5. Obtain coordinated GPU/pretrained qualification and weight-publication
   authorization before enabling live commands or `HUB_PUBLISHED`.

No model weights were downloaded, no GPU was present in this executor, and no
training of a pretrained model, rendering, deployment or publication ran.
CPU fixtures demonstrate software ownership and transaction contracts only.

Review follow-up: [review-fixes-20261031.md](review-fixes-20261031.md) records
source closure, frame-count, input ownership, checkpoint cleanup and coherent
candidate packaging corrections. Public source availability is no longer a
resolver blocker; reviewed native quality and multiframe game support remain gates.

The current packaging/lifecycle corrections are recorded in
[isolation-review-20261031.md](isolation-review-20261031.md). The older same-name
`+native012` wheel recipe is superseded by a separate `ming_native` distribution.
