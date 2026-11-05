> Current adoption: [published PG0.9.5 / core0.13.0](docs/pg095-adoption.md). Versioned earlier receipts below retain their original source scope; none establishes a PG0.9.5 native result.

# Formulation metadata

`ming-particle-sliders` owns the intended Ming product identifiers, ComfyUI
registration, fail-closed train/infer commands and experimental staged native
velocity glue. Game mathematics stay shared. The import below is a metadata
check, independent of the opt-in candidate GAN binding:

```python
from ming.surfaces import stamp

metadata = stamp()  # winning_formulation(), then require(metadata.as_dict())
```

The immutable dependency is recorded in `requirements.txt` and
`core.lock.json`. Its legacy `winning_formulation()` returns the `gmix`
architecture and provisional `particle-gmix-1600-v2` formulation. The
product does not copy shared game math or override those metadata fields.
Validating a stamp against its own dictionary checks the shared schema; it
cannot validate absent native training behavior or an absent checkpoint.

`MingResponseAdapter` binds an already staged native forward, shared native
LoRA leaves and frozen embeddings/reference latents to the shared response
contract. `build_hndl_game` uses actual shared `ConditionalPG09` and the
verified HyperGAN develop/HNDL field critic in a separate explicit core 0.12
candidate environment. Its objective is the public BISPF preset, with no local
loss, optimizer, regularizer or particle-bank imitation. This is an adapter GAN;
its native velocity field is never labeled a rendered image or ordinary Ming
LoRA. See [native binding scope](docs/native-binding.md).

The shared package owns generic optimizers, GAN losses, sampling contracts,
and checkpoint/resume mechanics. Product code should supply native model
glue and provenance, with regressions covering the actual forward and
parameter changes. Do not vendor a core or ParticleGAN checkout, silently
retarget old checkpoints, or add a reconstruction-loss training fallback.
