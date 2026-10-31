# Source-review corrections, 2026-10-31

These corrections follow review of head673f136. They do not change the shared
objective or strength defaults, loosen PG admission, or enable native commands.

| Review observation | Verified source and correction |
|---|---|
| PG version/unused hash admission | Public and candidate admission now hash every actual PG Python source and compare its full closure to the hash-verified shared release manifest. The recorded PG wheel identity is checked against that manifest. A same-version changed PG source tree refuses. |
| HG/HNDL exact candidate pins | Admission checks HG2.0.0/HNDL0.10.0, exact HG VCS commit, actual ordinary/train dependency constraints, actual HNDL source closure and shared verified clean-HG builder closure. Tests reject altered versions/origin/constraints and HNDL sources. |
| First-F target slicing | Correct ordering confirmed in exact official `patchify_and_embed`: target then reference frames. `unpatchify` returns all frames or only the first according to explicit `multi_frame_output`. Binding now checks exact output count and capability; tests include two targets/two references and single-frame output with two references. |
| Conditions and rows | Reject wrong types/ranks, mismatched row counts, nonfinite features/targets, non-FP32 critic features, and trainable encodings/references/features/cached targets before critic graph allocation. Input encoding validation runs before the shared preparation snapshot can detach it. |
| Bare CUDA device | Actual staged parameter devices already have concrete indices. A small normalization helper additionally resolves a bare `cuda` device through `current_device`; its CPU unit test does not claim CUDA execution. |
| Borrowed checkpoint hooks | Context manager/explicit close restore callback presence/identity, checkpoint flags and heterogeneous model training modes. Constructor errors restore too; overlapping borrows refuse. Caller-owned LoRA stays attached. Close only after backward work completes. |
| Candidate packaging selecting core0.11 | Separate product wheel version0.1.1+native012 declares exact core4abc VCS/PG0.9.4/HNDL0.10/HG5f368 dependencies. Default wheel retains core0.11. Normal pip resolution and pip check succeed without a no-dependencies workaround. |
| Mixed dtype/base freeze | Base parameters must be frozen and uniformly FP32/BF16; adapter leaves remain FP32. A genuine BF16 tiny CPU base proves finite adapter gradients and unchanged frozen base state. Mixed base dtype, unfrozen base and BF16 adapter leaves refuse. |
| License packaging | Original wheel already contained LICENSE and NOTICE. Metadata now declares SPDX MIT and both license files explicitly; rebuilt public/candidate archives are checked. No upstream model source or weights are vendored into the product. |

Official source inspected: inclusionAI/Ming-Image commit
`3515cd7b1e3049fc517eed7d35020251cd06519a`, transformer SHA256
`aefd220223ba80a59ecd9251a1bd512c4633cb68f76b1ce14de83a388c97eec4`.
A separate receipt executes the exact AST bodies of official patchify/unpatchify
and coordinate-grid methods over target counts1/2, reference counts0/1/3 and
both output-capability values (12 cases). Only geometry methods execute, with
identity token responses; no pretrained forward or model loading is implied.

Core PR148 was independently queried as open/draft/unmerged at exact head
`4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231`. The public commit tree is
`4634fc0b555e5331e37dbac1a0f48a28da2b734c`, and public packaging source matches
the previously qualified joint candidate. No lock was retargeted from a report.

A fresh task-local venv began without core, PG, HG, HNDL or product installed.
Normal pip installed the candidate wheel and fetched/built actual core4abc and
HyperGAN5f368 public VCS sources, plus PG0.9.4/HNDL0.10/foldlitedb0.1.2.
It reused a selected read-only ordinary CPU dependency view (including Torch),
so this is not a wholly independent Torch/native dependency installation.
Installed core direct_url records exact4abc and the package subdirectory;
full locked installed-source validation succeeds. Both candidate and public
pip checks report no broken requirements.

The source-only draft still has no published workflow: OAuth workflow scope
remains unavailable, and no denied workflow operation is retried. No release,
merge, deployment or weight publication occurs. Pretrained/GPU behavior, native
CFG/scheduler/RGBA output and multiframe HNDL game qualification remain blocked.

Final installed CPU results for this correction:

- Public core0.11/PG0.9.3 product wheel: **51 passed in10.43s**.
- Normally resolved public-source candidate wheel: **48 passed in3.35s**.
- Twenty-nine native adapter cases overlap; these are99 lane executions,
  70 distinct cases. The candidate adds19 stack/conditions/rows/metadata/game cases.
- Exact official geometry-method execution: **12 cases passed**.
- Actual public PG0.9.3 checkpoint refuses in candidate with no state mutation.
- Source-free public commands and candidate default-stamp refusal pass;
  both environments report no broken requirements.

```sh
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  .venv/bin/python -I -m pytest -q -o pythonpath= --import-mode=importlib \
  ming-worktree/tests --ignore=ming-worktree/tests/test_hndl_candidate.py
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  .venv-public-candidate/bin/python -I -m pytest -q -o pythonpath= \
  --import-mode=importlib ming-worktree/tests/test_native.py \
  ming-worktree/tests/test_hndl_candidate.py
.venv-public-candidate/bin/python -m pip install --no-cache-dir \
  --report evidence/public-candidate-resolver-final-report.json \
  evidence/review-native-wheels/ming_particle_sliders-0.1.1+native012-py3-none-any.whl
.venv-public-candidate/bin/python -I -m pip check
```

Wheel/sdist license/source/metadata verification and complete resolver/provenance
receipts accompany the coordinator's handoff packet. No skip is counted as a
qualified check. Python3.11, full pretrained dependencies and GPU tests remain unrun.
