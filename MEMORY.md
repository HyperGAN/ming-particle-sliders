# Memory requirements

| Operation / profile | GPU count | Calculated VRAM per GPU (GiB) | Calculated host RAM (GiB) | Input limits / precision |
|---|---:|---:|---:|---|
| Train staged512px no checkpointing | 1 | 132 | 64 | 512×512; ≤256 tokens; B=2; BF16 staged DiT; FP32 LoRA/critic/activation envelope; no checkpointing |
| Sample RAW response512px | 1 | 20 | 64 | 512×512; ≤256 tokens; B=1; BF16 staged DiT; FP32 LoRA/critic/activation envelope; no checkpointing |
| External native-pipeline planning only | 1 | 80 | 136 | 1024×1024;one frame; ≤1024 tokens; B=1; BF16 staged DiT; FP32 LoRA/critic/activation envelope; no checkpointing |

These are **calculated planning budgets, not measured peaks or hard runtime ceilings**. GiB means 2³⁰ bytes. They describe the finite profiles above, one worker and one learned slider, with RAW weights and **no model-weight EMA**. A larger image, token sequence, image/reference count, logical batch, cache or number of simultaneous workers needs a new calculation; there is no finite universal maximum for unbounded inputs.

## Scope and residency

Base: **`inclusionAI/Ming-Image-0.1-Design-Layer@ff1b22d2bdd0ae6597c0024a42cac7923b24cf92`**. Only one already staged native DiT, one rank16 adapter and one field critic are counted. Preencoded conditioning is supplied by the caller; text/vision encoder and VAE are not secretly included. The pinned base revision is a documentation calculation example, not a package-default model/admission or a cross-cohort migration.

Ming CLI/Comfy remain scaffold surfaces; this is the actual experimental MingResponseAdapter API only. The finite profile has one target frame, no references,≤256 caption tokens,512px latent16×1×64×64 and batch2, critic width≤32 and caller-supplied condition feature width≤106. Multi-frame/reference layer decomposition changes token count and has no universal finite maximum. Production rendering/CFG are not implemented by this binding.

The table's efficient-SDPA rows require a flash/efficient implementation. An automatic backend can fall back to dense math; use the dense row where provided, or add the dense term below before planning capacity. A planning row does not change code, precision, optimizer, rates, initialization, checkpoints or qualification.

## Frozen and trainable ledger

- staged native DiT: **12,309,817,472 bytes** (11.464 GiB). 6,154,908,736 parameters.
- index tensor bytes; equals4P: **24,619,634,944 bytes** (22.929 GiB).

From the captured30 main+2 noise-refiner+2 context-refiner architecture, P=34×(4D²+3DI+4D+2h)+32×(256×4D+4D)+input/output/time/caption/pad terms = 6,154,908,736. The captured index total equals4P=24,619,634,944 bytes; equality alone does not prove every stored dtype. Resident bytes here follow the staged bfloat16 dtype enforced by the chosen binding. No MLLM/TE/VAE is loaded by this response binding.

Trainable allowance: **16,811,680 parameters**. 34 attention modules×4 projections×16×(3840+3840) +≤100,000 critic parameters = ≤16,811,680; ordinary LoRA, no particle bank. FP32 trainable weights and gradients require `4A+4A` bytes. Frozen base parameters have no gradient or optimizer tensor allocation. Routed particle-bank parameters, where present, are counted once; a frozen response cache is not a trainable bank.

The current `bispf_optimized` PG0.11.1 recipe uses **DualNorm, momentum 0, direction_blend**; it is not assumed to be Adam. Persistent per-parameter moment tensors are0 in this profile; scalar counters are small. The peak ledger additionally reserves **`48A+32MiB` bytes** for protected normals, displacement/direction copies, FP64 promotion and thin-matrix SVD/Gram workspace. This is a conservative temporary-allocation allowance, not a claim that the optimizer uses no memory. Selecting historical `bispf`/Adam instead adds its actual state (typically another`8A` bytes for two FP32 moments); nonzero momentum or a different solver requires recalculation. Weight EMA adds0 bytes and must stay disabled.

## Numeric peak ledger (GiB before headroom)

| Profile | Frozen GPU weights | Activation / decode | Trainable + optimizer | Other GPU allowance | Host loading/cache subtotal |
|---|---:|---:|---:|---:|---:|
| Train staged512px no checkpointing | 11.464 | 89.648 | 0.908 | 0.000 | 47.858 |
| Sample RAW response512px | 11.464 | 0.952 | 0.063 | 0.000 | 47.858 |
| External native-pipeline planning only | 49.237 | 12.000 | 0.908 | 0.000 | 104.129 |

CPU rows place their active tensors in host RAM; their frozen/activation entries are not GPU residency. Sequential phase sub-ledgers are in the source-facts report.

Cache bound for the training profile: **32 rows**, each with three FP32 native fields, plus **32 cached prompts**. Tensor-only cache allowance: `25,165,824+41,943,040` bytes; masks/indices and serialization overhead remain covered by the explicit host subtotal and reserve. These bounds are inputs to this calculation, not new runtime limits.

## Activation, cache and host calculation

The auditable inputs and per-profile subtotals are in [source-facts.json](docs/memory/source-facts.json). For a transformer envelope, let `N=image tokens+bounded text tokens`, `D=hidden width`, `I=FFN width`, `H=attention heads`, `L=covered blocks`, and `B=simultaneously retained native graphs`:

- One-block inference allowance: `4B(16ND+6NI)` bytes. The4-byte coefficient deliberately covers FP32 intermediates even for BF16 weights;16 and6 allow Q/K/V/output, normalization/residual, gated FFN and adapter intermediates.
- Uncheckpointed training allowance: `4LB(16ND+6NI)`; checkpointed allowance: `4(4LBND+2B(16ND+6NI))`. Checkpointing saves layer inputs and recomputes block internals; it does not remove the frozen model, adapter or critic.
- Dense math fallback adds `8BH N²` bytes for inference and `16LBH N²` for uncheckpointed training, or `32BH N²` for the checkpointed current-block envelope. These are conservative materialized-score/probability allowances; flash/efficient SDPA avoids that quadratic storage. GQA expansion uses the query-head count for this bound.
- Decode allowance is separately based on bounded output/tile area and channel width; profile notes record the bound. Peak is the **maximum of sequential phases**, with the actual overlapping model weights added inside each phase. Sampling steps are sequential and do not multiply resident peak.
- Frozen CPU rows are bounded by the profile (three FP32 native fields per cached row, plus explicitly bounded prompt embeddings). Loader staging/serialization copies and the largest published shard are included in host subtotals. Host cache/page residency and kernel workspaces vary, so these are planning allowances, not an allocator theorem.

Final GPU budget is `ceil_to_4GiB(1.25×GPU subtotal+2GiB)`; host is `ceil_to_4GiB(1.25×host subtotal+4GiB)`. The explicit25% plus runtime reserve covers allocator fragmentation, CUDA context/libraries, kernel workspaces, small FP32 islands and Python/serialization overhead. These chosen reserve multipliers are explicit planning assumptions; architecture metadata alone cannot prove every allocator or fused-kernel peak. It is not borrowed from scheduler admission caps, card size or global free-memory floors. Do not add every row's peak together for sequential phases; parallel workers do add their simultaneous requirements.

This profile's text-state allowance is **41,943,040 bytes** (0.039 GiB): no encoder/KV in staged response API; optional external MLLM KV only. It is already included in the phase or host allowance, not added twice. Text-only encoding has no generated-token loop. If the actual encoder returns KV state, it is explicitly included as `2×layers×B×KV_heads×tokens×head_dim×dtype_bytes`; cache-disabled encoders allocate no persistent autoregressive KV. Recurrent/linear-attention state is accounted separately where applicable. Arbitrary output-token generation and image-conditioned encoders are outside these profiles.

## Sources and reproducibility

- [ming/native.py](ming/native.py)
- [ming/constants.py](ming/constants.py)
- [docs/native-binding.md](docs/native-binding.md)
- [Captured architecture, published file/index sizes and arithmetic](docs/memory/source-facts.json) retains exact model revisions, metadata SHA256s and direct Hugging Face config URLs. No weights were downloaded or opened for this calculation.
- Optimizer facts: the pinned ParticleGAN0.11.1 `recipes.py` (`bispf_optimized`), `optim/dualnorm.py` momentum/state branch, and `optim/direction_blend.py` FP64 normals. These are part of the dependency selected by the repository's existing core lock; their source facts are recorded in the combined calculation report.

Documentation was calculated from source and published metadata on 2026-11-14. No model construction, import of Torch/Diffusers, forward pass, benchmark, sampling, training or new GPU work was performed.
