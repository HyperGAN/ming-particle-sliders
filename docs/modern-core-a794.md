> Historical dev13/corea794 qualification. Current source uses the separate
> [PG094/ddb41423 candidate](pg094-current.md). Receipts below are unchanged.

# Active modern candidate at core a794e3b4

The active experimental distribution is `ming-particle-sliders-native==0.1.1.dev13`,
importing `ming_native`. It pins the exact shared core commit
`a794e3b4a8813b012f2c4faa3da82b78d4a90214`, with core version 0.12.0, official
ParticleGAN 0.9.4, HyperGAN commit `5f368a3770cf5bdae0448995c9f76e29af8049f7`
and HNDL 0.10.0. No training objective, bank architecture, model license, strength
law or native admission gate changes here.

The stable `ming-particle-sliders==0.1.1` distribution retains its public core
0.11 and PG 0.9.3 cohort. Its Python runtime, identity, lock and dependency files
are unchanged. The earlier native preview at dev12 and core
`4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231` remains an old cohort; its original
lock is preserved under `profiles/history/`. Neither old cohort is relabeled.

## Artifact and source identity

`profiles/native-candidate.lock.json` was regenerated once from the approved
Git tree and a built core wheel. Its recursive mapping contains all 51 tracked
source/data files in the `particle_sliders` and `concept_slider_core` package
roots. This includes package data and `LICENSE`, rather than filtering by a few
file suffixes. The built core wheel was checked against the same complete map.
Runtime admission rejects changed, missing and unexpected nested files; Python
bytecode caches are excluded. Exact Git origin and declared dependency checks
remain enforced.

Only the active native builder uses `profiles/native_cohort.py` and
`profiles/native_runtime.py`. They preserve the reviewed product binding while
changing the candidate identity and enforcing the new complete core closure.
Keeping these inputs separate leaves the stable `ming` package unchanged.
The native wheel still has its own namespace, one lock and no stable console or
Comfy entry points. Its new version/profile cannot silently select an old packet.

## Reproducible build and install

Build from this repository or its source distribution, then use a separate CPU
qualification environment. Normal dependency resolution is required:

```bash
python -m pip install 'setuptools>=84.2.1' 'build>=1,<2'
python scripts/build_native_candidate.py --output-dir dist-native
python -m pip install 'dist-native/ming_particle_sliders_native-0.1.1.dev13-py3-none-any.whl[test]'
python -m pip check
python -I scripts/run_candidate_tests.py
```

For the CPU lane, stage the official CPU Torch wheel before resolving this
artifact. The qualification used Torch 2.6.0+cpu on Python 3.12.15, with no
borrowed site-packages and no CUDA build. No full native model extras or weights
are needed. Core is resolved from the exact VCS URL embedded in the native wheel.
A rebuilt core wheel may have different ZIP timestamps; exact commit provenance
and the entire installed source/data closure are checked independently.

The qualification-only lock generator is `scripts/lock_native_core.py`. It reads
only the approved commit from a supplied Git repository and verifies a supplied
reference core wheel. Running it for a different artifact would create a new
identity requiring review; it is not part of routine installation.

## Qualification boundaries

Product conformance covers exact installed origin/bytes, recursive closure
corruption, native adapter ownership, scale-zero behavior, raw responses,
checkpoint continuation and original cleanup/failure-retry contracts. Additional
CPU tests compose the product binding with the real shared `ResponseSession` to
exercise EMA off, raw and EMA exports, exact resume, observation/RNG preservation
and transactional restore failure followed by retry. These are shared-session
software fixtures. They do not add a public Ming EMA/sampler entry point or
qualify pretrained/native EMA behavior.

A separate process generates a genuine packet using the unchanged dev12 preview and core at `4abc0eb`
installation. The modern process verifies the same packet hash and refuses it
without changing game state. Earlier stable PG 0.9.3 packets are checked separately.
The packet headers are not rewritten. Published JSON receipts and the reproducible
probe distinguish these checks from the synthetic corrupted-header unit test.
No numerical packet weights are published.

The coordinator supplied the upstream software qualification for this exact core,
including a bounded synthetic CUDA fixture. This task runs CPU product checks
only. That upstream result is not native Ming, width, audio, image-quality or
production rendering admission. Multiframe HNDL, production CFG/scheduler/RGBA,
pretrained behavior and GPU model runs remain gated or unrun.

The draft remains source-only. No workflow/access/visibility changes, release,
deployment, model-weight publication or HNDL optimization are included. The new
exact product head requires Claude actual-diff review and RETURN before landing.

## Recorded results

The installed dev13 candidate passed **67 CPU tests in 6.69 seconds** from the
extracted product source distribution and temporary, byte-identical test copies.
The repository checkout was absent from runtime import paths. Normal dependency
resolution and `pip check` passed. All 51 imported core source/data files matched
the locked mapping, and installed VCS provenance recorded the exact approved
commit. No borrowed site-packages were used.

Published receipts are under `docs/evidence/modern-a794/`: `qualification.json`,
`preview-producer.json`, `preview-refusal.json`, and `stable-pg093-refusal.json`.
The dev12 producer ran its actual old installation. The consumers checked the
same packet bytes without editing headers, and both refusals left game state
unchanged. These are CPU fixture records, not historical user checkpoints or
native model results.
