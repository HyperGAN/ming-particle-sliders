# Formulation metadata

`ming-particle-sliders` owns the intended Ming product identifiers, ComfyUI
registration, and fail-closed train/infer commands. It has no native Ming
training game or adapter execution. The shared import below is an actual
metadata check, not evidence of a working GAN backend:

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

A future Ming trainer must bind a real native forward, a routed trainable
adapter, frozen teacher responses, and meaningful conditioning to the
qualified shared GAN API. It must use the reviewed released ParticleGAN
dependency owned by that shared package. An import or constructor alone
does not establish this path. The required evidence is in
[the capability audit](docs/capability-audit.md).

The shared package owns generic optimizers, GAN losses, sampling contracts,
and checkpoint/resume mechanics. Product code should supply native model
glue and provenance, with regressions covering the actual forward and
parameter changes. Do not vendor a core or ParticleGAN checkout, silently
retarget old checkpoints, or add a reconstruction-loss training fallback.
