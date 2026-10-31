"""Experimental staged Ming velocity binding. No loaders, renderer or local game.

Only single-timestep, unipolar frozen-positive responses are supported. Native
production sampling/CFG and pretrained/GPU behavior remain unqualified.
"""
import hashlib
import inspect
from pathlib import Path

import torch

from particle_sliders.pg09_backends import FrozenResponseAdapter, bind_native_gradient_checkpointing
from particle_sliders.pg09_images import image_lora_parameters

from .cohort import require_cohort
from .surfaces import BASE_MODEL_ID

UPSTREAM_COMMIT = '3515cd7b1e3049fc517eed7d35020251cd06519a'
UPSTREAM_SOURCES = {
    'generator': 'a31f006c123febdcfdfeb92b1b55f2716b154a98e3c3e13bd7bf825148e9901e',
    'transformer': 'aefd220223ba80a59ecd9251a1bd512c4633cb68f76b1ce14de83a388c97eec4',
}


class MingResponseAdapter(FrozenResponseAdapter):
    """Reuse shared native LoRA and response contracts on an already staged model.

    ``fixture=True`` is restricted to CPU and is always recorded in provenance.
    Production callers must stage the exact official ConditionedTransformer and
    its transformer, plus explicit immutable base-weight identity themselves.
    This class does not establish that those weights or a sampler are qualified.
    """
    family = 'ming_design_layer_velocity'
    time_coordinates = 'raw scheduler t in [0,1000]; model (1000-t)/1000; negate output'

    def __init__(self, model, native_lora, *, model_identity=None, fixture=False, cohort='public-0.11'):
        require_cohort(cohort)
        if type(fixture) is not bool:
            raise ValueError('fixture must be explicit boolean')
        if fixture:
            if any(p.device.type != 'cpu' for p in (*model.parameters(), *native_lora.parameters())):
                raise ValueError('fixture mode is CPU only')
            identity = dict(model_identity or {}, qualification='CPU fixture only')
        else:
            identity = dict(model_identity or {})
            revision = identity.get('revision', '')
            if (identity.get('base_model_id') != BASE_MODEL_ID or not isinstance(revision, str) or len(revision) != 40 or
                    any(c not in '0123456789abcdef' for c in revision) or
                    identity.get('upstream_commit') != UPSTREAM_COMMIT):
                raise ValueError('explicit immutable Ming base-weight/source identity required')
            inner = getattr(model, 'transformer', None)
            if inner is None:
                raise ValueError('exact official Ming ConditionedTransformer wrapper required')
            for obj, name in ((model, 'generator'), (inner, 'transformer')):
                source = inspect.getsourcefile(type(obj))
                if source is None:
                    raise ValueError('inspectable exact official Ming source required')
                if hashlib.sha256(Path(source).read_bytes()).hexdigest() != UPSTREAM_SOURCES[name]:
                    raise ValueError('staged Ming source differs from the locked official ' + name)
        children = {id(m) for m in model.modules()}
        for layer in native_lora.unet_loras:
            original = getattr(layer, 'org_forward', None)
            owner = getattr(original, '__self__', None)
            if id(owner) not in children or getattr(owner.forward, '__self__', None) is not layer:
                raise ValueError('native LoRA must be attached to this actual staged model')
        network = image_lora_parameters(model, native_lora)
        self.transformer = model
        self.fixture = fixture
        self.cohort = cohort
        super().__init__(network, model_identity=identity)
        # Ming exposes the Diffusers checkpoint hook on the inner transformer.
        bind_native_gradient_checkpointing(getattr(model, 'transformer', model), network)

    def prepare(self, row):
        if row.schedule is not None or row.guidance != 1:
            raise ValueError('Ming binding supports raw single-timestep velocity only; native sampler/CFG pending')
        if row.scale != 1 or row.strength != 1:
            raise ValueError('Ming binding supports unipolar positive targets at +1 only')
        if row.latent.ndim != 5 or row.latent.shape[0] != 1:
            raise ValueError('Ming requires its genuine [1,C,F,H,W] latent field')
        if row.latent.shape[-1] % 2 or row.latent.shape[-2] % 2:
            raise ValueError('Ming native 2x2 spatial patches require even H and W')
        return super().prepare(row)

    def _velocity(self, condition, latent, timestep):
        if (not isinstance(condition, dict) or set(condition) - {'embeds', 'embeds_2', 'reference_latents'}
                or 'embeds' not in condition):
            raise ValueError('Ming requires frozen native embeds and optional embeds_2/reference_latents')
        if timestep is None or timestep.numel() != 1 or not torch.isfinite(timestep).all() or not 0 <= timestep.item() <= 1000:
            raise ValueError('one finite raw scheduler timestep in [0,1000] required')
        dtype = next(self.transformer.parameters()).dtype
        device = latent.device
        def tensor(key, ndim):
            value = condition.get(key)
            if value is None and key != 'embeds':
                return None
            if (not isinstance(value, torch.Tensor) or value.ndim != ndim or value.shape[0] != 1 or min(value.shape) < 1
                    or not value.is_floating_point() or not torch.isfinite(value).all()):
                raise ValueError(key + ' requires a finite single-example native tensor')
            return value.to(device=device, dtype=dtype)
        embeds, embeds_2, refs = tensor('embeds', 3), tensor('embeds_2', 3), tensor('reference_latents', 5)
        if refs is not None and (refs.shape[1] != latent.shape[1] or refs.shape[-2:] != latent.shape[-2:]):
            raise ValueError('reference latent channels and spatial geometry must match target')
        output = self.transformer(
            list(latent.to(dtype).unbind(0)), (1000 - timestep.reshape(1).to(device).float()) / 1000,
            list(embeds.unbind(0)), return_dict=False,
            encoder_hidden_states_2=None if embeds_2 is None else list(embeds_2.unbind(0)),
            ref_hidden_states=None if refs is None else list(refs.unbind(0)),
        )[0]
        if (not isinstance(output, (list, tuple)) or len(output) != 1 or
                output[0].ndim != 4 or output[0].shape[0] != latent.shape[1] or
                output[0].shape[1] < latent.shape[2] or output[0].shape[-2:] != latent.shape[-2:]):
            raise ValueError('native Ming output must preserve target channels/frames/spatial field')
        return -torch.stack([output[0][:, :latent.shape[2]].float()])

    def metadata(self):
        return dict(super().metadata(), protocol='ming-staged-velocity-v1',
                    core_cohort=self.cohort,
                    upstream_commit=UPSTREAM_COMMIT, upstream_sources=UPSTREAM_SOURCES,
                    response_coordinates='native [1,C,F,H,W] latent velocity',
                    particle_bank='none: shared native LoRA adapter GAN only',
                    fixture=self.fixture, render_qualified=False, native_quality_qualified=False,
                    conditioning='frozen embeddings and preencoded reference latents',
                    sampler='unavailable; no trajectory or CFG binding')


class MingHNDLCritic(torch.nn.Module):
    """Bind a singleton-frame native field to the real shared HyperGAN/HNDL graph."""
    score_bound = 0

    def __init__(self, shape, condition_width, width=32):
        super().__init__()
        require_cohort('native-candidate-0.12')
        if len(shape) != 5 or shape[0] != 1 or shape[2] != 1:
            raise ValueError('Ming HNDL critic currently requires one genuine latent frame')
        from particle_sliders.fibo_hndl import HNDLFieldCritic
        self.native_shape = tuple(shape[1:])
        self.critic = HNDLFieldCritic(shape[1], condition_width, shape[-2:], width)

    def forward(self, response, condition):
        if tuple(response.shape[1:]) != self.native_shape:
            raise ValueError('Ming HNDL critic native field geometry differs')
        return self.critic(response.squeeze(2), condition)


def build_hndl_game(adapter, prepared_rows, conditions, *, total_steps, seed=7, batch_size=2, critic_width=32):
    """Opt-in new cohort only. Shared PG owns every objective, optimizer and step.

    No native model is loaded here. Conditions must be supplied by the product
    caller from its actual frozen protocol, never invented from row indices.
    """
    require_cohort('native-candidate-0.12')
    if adapter.cohort != 'native-candidate-0.12':
        raise ValueError('Ming HyperGAN/HNDL game requires an explicitly new native candidate cohort')
    rows = tuple(prepared_rows)
    if not rows:
        raise ValueError('frozen Ming response rows required')
    if type(seed) is not int or not 0 <= seed < 2**63 - 100:
        raise ValueError('invalid seed')
    from particle_sliders.pg09 import ConditionalPG09
    from particle_sliders.field_critic import seed_initialization_rng
    device = next(adapter.network.parameters()).device
    devices = [device.index] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        seed_initialization_rng(seed, devices)
        critic = MingHNDLCritic(tuple(rows[0].target.shape), conditions.shape[1], critic_width)
        critic.to(device)
        return ConditionalPG09(adapter, rows, critic, conditions=conditions,
                               total_steps=total_steps, seed=seed, batch_size=batch_size,
                               response_domain='continuous_field')
