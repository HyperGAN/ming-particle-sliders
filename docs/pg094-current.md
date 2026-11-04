# Current ParticleGAN 0.9.4 software candidate

The default package `ming-particle-sliders==0.2.0` and the separate experimental
`ming-particle-sliders-native==0.2.0.dev2` now both use shared core 0.12.0 at
`ddb41423bcefc1fe76cd94d0f9b367353b3ea137`, including its merged unified training
and isolated execution APIs. Official ParticleGAN is 0.9.4 (release source
`330eb94ee748cdcad131b0275d8de75fd5a7f96f`). Each package carries its own distinct
profile and complete 574-file core source/data closure. Exact immutable origin,
installed dependency, source/data and official PG release checks remain enforced.

The default package is still a metadata scaffold. `ming-train --dry-run` and
`ming-infer --dry-run` work; live CLI and Comfy operations refuse. Updating its
metadata does not supply the missing pretrained loader, sampler or native
qualification. The stable namespace cannot select the isolated native game.

The isolated native candidate retains the staged Ming velocity adapter and real
HyperGAN/HNDL field critic. Shared `ConditionalPG09` owns objectives, optimizers,
checkpoints and updates. This is an adapter-only GAN with `particles=None`;
it does not implement the full ParticleGAN SBNP particle bank. No local substitute
GAN, invented bank, model download or GPU training is added. The native profile is
`native-pg094-ddb41423`; old packets retain their original metadata and require
their original environment.

## Separate environments

Python >=3.11 and the selected core's actual dependency constraints are required.
Install CPU Torch first for CPU qualification, then allow ordinary dependency
resolution. Do not install both product distributions in one environment.

```bash
python -m pip install '.[test]'
ming-train --dry-run
ming-infer --dry-run
python -m pytest -q
```

For the native candidate, build in the source checkout, then install and test
in a second environment:

```bash
python scripts/build_native_candidate.py --output-dir dist-native
python -m pip install 'dist-native/ming_particle_sliders_native-0.2.0.dev2-py3-none-any.whl[test]'
python -m pip check
python -I scripts/run_candidate_tests.py
```

The native wheel owns `ming_native` only and has no stable console or Comfy
entry points. Its real HNDL stack remains HyperGAN
`5f368a3770cf5bdae0448995c9f76e29af8049f7` and HNDL 0.10.0. The original staged
model ownership, cleanup, frozen conditioning and single-frame HNDL requirements
remain enforced. A shared executor fixture is software evidence only; it must
explicitly set `dummy=true` and cannot admit a pretrained or native profile.

## Historical reproduction and gates

The earlier public 0.1.1/core0.11/PG0.9.3 lock is preserved byte-for-byte at
`profiles/history/public-0.1.1-core3c25.lock.json`. The prior native dev13/corea794
lock is preserved at `profiles/history/native-dev13-corea794.lock.json`; the dev12
lock and all prior receipts remain unchanged. Reproduce those cohorts from their
original product commit `55f40380de8b02e31e67327eb84e7d41f763c32f` and the exact
archived core identity. Do not install this new wheel into an old run environment.
The scripts producing old packets still require the actual original installation;
old receipt files do not claim validation of this new source.

Pretrained Ming, production CFG/scheduler/RGBA rendering, multiframe HNDL and
native quality remain unqualified. No weights are trained or published. The new
actual product diff requires Claude review and RETURN before landing, along
with the existing native admission gates. This task prepares a source PR only.

## Recorded CPU results

Separate fresh Python 3.11.17 / Torch 2.9.1+cpu environments passed normal exact
VCS dependency resolution and `pip check`, without borrowed site-packages or CUDA
initialization. The default package passed 63 tests with 31 isolated native-lane
skips. The installed native wheel passed all 68 tests from temporary byte-identical
test copies with the checkout absent from runtime imports.

The merged worker fixture covered plan/check, explicit dummy=false refusal, periodic
checkpoints at steps 1–4, partial training to step 2, exact resume to step 4 and EMA
export. This invokes the installed worker with temporary in-process registration;
it does not ship a production profile or demonstrate the outer executor's process
isolation. Existing native tests cover raw/EMA inference-only loading, state/RNG
neutral observation, transactional restore, cleanup/retry and closure corruption.
See [the current receipt](evidence/pg094-ddb41423/qualification.json).


The isolated dev2 product also adds the explicit [ordered-target-frame critic binding](ordered-frame-binding.md), with a distinct artifact identity and native five-axis PG responses. Its CPU checks do not qualify pretrained layered outputs, CFG, RGBA rendering or an LR value. The singleton-frame binding retains its existing multiframe refusal.
