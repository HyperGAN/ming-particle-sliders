# Installed CPU validation, 2026-10-31

The two lanes use separate virtual environments and installed product wheels.
Python 3.12.15 / Torch 2.14.1+cpu; CUDA disabled and Hugging Face offline.
Torch and selected ordinary dependencies are reused through a task-local,
read-only dependency view. This is not a clean full native dependency install.
No model weights, pretrained execution, GPU or rendering checks ran.

| Lane | Installed cohort | Result |
|---|---|---|
| Public | Core 0.11.0 / PG 0.9.3; unchanged public core SHA and historical records | 36 tests passed |
| Explicit native candidate | Core 0.12.0 joint candidate / PG 0.9.4 / actual HyperGAN develop 5f368a3 / HNDL 0.10.0 | 18 tests passed |

The candidate tests include the same 15 bounded native adapter cases also run
in the public lane; counts are lane results, not 54 distinct tests. They test
native time/sign/geometry, scale-zero equality, exact FP32 adapter ownership,
frozen base parameters, frozen conditioning/reference snapshots, export/load
identity rejection, context cleanup, and per-forward checkpoint scale capture.
The three candidate-only cases execute the real shared HyperGAN/HNDL graph and
PG game, prove generator/critic changes and deterministic exact continuation,
reject a mismatched packet without mutation, reject multiframe critic geometry,
and ensure the default public stamp refuses silent candidate adoption.

An additional two-process receipt creates an actual checkpoint with installed
core 0.11/PG 0.9.3, then attempts its restore with the candidate. The candidate
refuses with instructions to resume in the original pinned environment, with
all model/critic/optimizer/random-stream state unchanged.

Source-free module and console dry-runs pass from `/tmp` in the public lane.
The Comfy custom-node loader is exercised separately without adding its source
directory to `sys.path`. Runtime admission rejects wrong core version, origin,
and source content. Installed lock resources match the preserved root record.
Both lane environments report no broken requirements. HyperGAN's declared
`foldlitedb==0.1.2` dependency is installed in the candidate environment.

Installed tests use:

```bash
python -I -m pytest -q -o pythonpath= --import-mode=importlib tests \
  --ignore=tests/test_hndl_candidate.py
# In the explicitly staged, exact candidate environment:
python -I -m pytest -q -o pythonpath= --import-mode=importlib \
  tests/test_native.py tests/test_hndl_candidate.py
```

Wheel and sdist builds succeed with setuptools 84.2.1. The sdist carries the
lock records, source/custom-node entry, scripts and native contract docs.
A CPU-only CI workflow was prepared for the public installed-package lane on
Python 3.11/3.12 with official CPU Torch 2.6.0. GitHub refused its publication
because the current OAuth credential lacks workflow scope. The proposal remains
a separate local handoff artifact; no permissions were changed. Remote CI and
Python 3.11 have not run. No workflow is added by this change, and there is no
merge-triggered release, model download, weight upload or deployment workflow.
