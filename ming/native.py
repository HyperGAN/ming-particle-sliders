"""Experimental staged Ming velocity binding. No loaders, renderer or local game.

Only single-timestep, unipolar frozen-positive responses are supported. Native
production sampling/CFG and pretrained/GPU behavior remain unqualified.
"""
import hashlib
import inspect
from pathlib import Path
import weakref

import torch

from particle_sliders.pg09_backends import (
    FrozenResponseAdapter, PreparedResponse, bind_native_gradient_checkpointing,
)
from particle_sliders.pg09_images import image_lora_parameters

from .cohort import require_cohort
from .surfaces import BASE_MODEL_ID

UPSTREAM_COMMIT = '3515cd7b1e3049fc517eed7d35020251cd06519a'
UPSTREAM_SOURCES = {
    'generator': 'a31f006c123febdcfdfeb92b1b55f2716b154a98e3c3e13bd7bf825148e9901e',
    'transformer': 'aefd220223ba80a59ecd9251a1bd512c4633cb68f76b1ce14de83a388c97eec4',
}


def _cuda_rng_devices(device):
    device = torch.device(device)
    if device.type != 'cuda':
        return []
    return [device.index if device.index is not None else torch.cuda.current_device()]


def _check_condition(condition, latent):
    allowed = {'embeds': 3, 'embeds_2': 3, 'reference_latents': 5}
    if not isinstance(condition, dict) or set(condition) - allowed.keys() or 'embeds' not in condition:
        raise ValueError('Ming requires frozen native embeds and optional embeds_2/reference_latents')
    for key, value in condition.items():
        if value is None and key != 'embeds':
            continue
        if (not isinstance(value, torch.Tensor) or value.ndim != allowed[key] or
                value.shape[0] != 1 or min(value.shape) < 1 or value.requires_grad or
                not value.is_floating_point() or not torch.isfinite(value).all()):
            raise ValueError(key + ' requires a finite frozen single-example native tensor')
        if key == 'reference_latents' and (value.shape[1] != latent.shape[1] or value.shape[-2:] != latent.shape[-2:]):
            raise ValueError('reference latent channels and spatial geometry must match target')


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
        parameters = list(model.parameters())
        if not parameters or any(p.requires_grad for p in parameters):
            raise ValueError('Ming base parameters must all be frozen')
        dtypes = {p.dtype for p in parameters}
        if len(dtypes) != 1 or next(iter(dtypes)) not in (torch.float32, torch.bfloat16):
            raise ValueError('Ming base must have one FP32 or BF16 dtype')
        devices = {p.device for p in (*parameters, *native_lora.parameters())}
        if len(devices) != 1:
            raise ValueError('Ming base and adapter must share one explicit staged device')
        inner = getattr(model, 'transformer', model)
        self.multi_frame_output = getattr(inner, 'multi_frame_output', None)
        if type(self.multi_frame_output) is not bool:
            raise ValueError('explicit native multi_frame_output capability required')
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
        borrowed_by = getattr(model, '_ming_response_binding', None)
        if borrowed_by is not None and borrowed_by() is not None and not borrowed_by()._closed:
            raise ValueError('Ming model is already borrowed by an open response binding')
        # Borrowed module modes and public checkpoint callbacks must be restored.
        self._borrowed = [(module, module.training,
                           {key: (key in module.__dict__, module.__dict__.get(key))
                            for key in ('_gradient_checkpointing_func', 'gradient_checkpointing', '_ming_response_binding')})
                          for module in model.modules()]
        self.transformer = model
        self.fixture = fixture
        self.cohort = cohort
        self._closed = False
        model._ming_response_binding = weakref.ref(self)
        try:
            network = image_lora_parameters(model, native_lora)
            super().__init__(network, model_identity=identity)
            bind_native_gradient_checkpointing(inner, network)
        except BaseException:
            self.close()
            raise

    def close(self):
        if self._closed:
            return
        for module, training, state in self._borrowed:
            module.training = training
            for key, (present, value) in state.items():
                if present:
                    setattr(module, key, value)
                elif key in module.__dict__:
                    delattr(module, key)
        self._closed = True

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *exc):
        self.close()

    def _require_open(self):
        if self._closed:
            raise RuntimeError('Ming response binding is closed')

    def velocity(self, condition, latent, timestep, *, scale=1.):
        self._require_open()
        return super().velocity(condition, latent, timestep, scale=scale)

    def prepare(self, row):
        self._require_open()
        if row.schedule is not None or row.guidance != 1:
            raise ValueError('Ming binding supports raw single-timestep velocity only; native sampler/CFG pending')
        if row.scale != 1 or row.strength != 1:
            raise ValueError('Ming binding supports unipolar positive targets at +1 only')
        if (not isinstance(row.latent, torch.Tensor) or row.latent.ndim != 5 or row.latent.shape[0] != 1 or
                min(row.latent.shape) < 1 or not row.latent.is_floating_point() or not torch.isfinite(row.latent).all()):
            raise ValueError('Ming requires its genuine [1,C,F,H,W] latent field')
        if row.latent.shape[-1] % 2 or row.latent.shape[-2] % 2:
            raise ValueError('Ming native 2x2 spatial patches require even H and W')
        if row.latent.requires_grad or (row.timestep is not None and row.timestep.requires_grad):
            raise ValueError('Ming latent and timestep must be frozen')
        if not self.multi_frame_output and row.latent.shape[2] != 1:
            raise ValueError('single-frame native capability cannot return multiple target frames')
        for condition in (row.neutral, row.positive, row.negative):
            _check_condition(condition, row.latent)
        return super().prepare(row)

    def _velocity(self, condition, latent, timestep):
        _check_condition(condition, latent)
        if timestep is None or timestep.numel() != 1 or not torch.isfinite(timestep).all() or not 0 <= timestep.item() <= 1000:
            raise ValueError('one finite raw scheduler timestep in [0,1000] required')
        dtype = next(self.transformer.parameters()).dtype
        device = latent.device
        def tensor(key):
            value = condition.get(key)
            if value is None and key != 'embeds':
                return None
            return value.to(device=device, dtype=dtype)
        embeds, embeds_2, refs = tensor('embeds'), tensor('embeds_2'), tensor('reference_latents')
        if refs is not None and (refs.shape[1] != latent.shape[1] or refs.shape[-2:] != latent.shape[-2:]):
            raise ValueError('reference latent channels and spatial geometry must match target')
        output = self.transformer(
            list(latent.to(dtype).unbind(0)), (1000 - timestep.reshape(1).to(device).float()) / 1000,
            list(embeds.unbind(0)), return_dict=False,
            encoder_hidden_states_2=None if embeds_2 is None else list(embeds_2.unbind(0)),
            ref_hidden_states=None if refs is None else list(refs.unbind(0)),
        )[0]
        # Official patchify concatenates [targets, references] along F; unpatchify
        # returns all those frames when multi_frame_output=True, otherwise one.
        expected_frames = latent.shape[2] + (0 if refs is None else refs.shape[2]) if self.multi_frame_output else 1
        if (not isinstance(output, (list, tuple)) or len(output) != 1 or
                output[0].ndim != 4 or output[0].shape[0] != latent.shape[1] or
                output[0].shape[1] != expected_frames or output[0].shape[-2:] != latent.shape[-2:]):
            raise ValueError('native Ming output must preserve target channels/frames/spatial field')
        return -torch.stack([output[0][:, :latent.shape[2]].float()])

    def metadata(self):
        return dict(super().metadata(), protocol='ming-staged-velocity-v1',
                    core_cohort=self.cohort,
                    multi_frame_output=self.multi_frame_output,
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
    adapter._require_open()
    rows = tuple(prepared_rows)
    if len(rows) < 2 or any(not isinstance(r, PreparedResponse) for r in rows):
        raise ValueError('at least two prepared frozen Ming response rows required')
    if any(not isinstance(t, torch.Tensor) for r in rows for t in (r.target, r.neutral)):
        raise ValueError('Ming response rows require frozen tensors')
    shape = tuple(rows[0].target.shape)
    if any(tuple(t.shape) != shape or t.dtype != torch.float32 or t.requires_grad or not torch.isfinite(t).all()
           for r in rows for t in (r.target, r.neutral)):
        raise ValueError('Ming response rows must share finite frozen geometry')
    if (not isinstance(conditions, torch.Tensor) or conditions.ndim != 2 or
            conditions.shape[0] != len(rows) or conditions.shape[1] < 1 or
            conditions.dtype != torch.float32 or conditions.requires_grad or not torch.isfinite(conditions).all()):
        raise ValueError('Ming conditions require finite frozen FP32 [row_count, feature_width] tensor')
    conditions = conditions.detach().clone()
    if type(seed) is not int or not 0 <= seed < 2**63 - 100:
        raise ValueError('invalid seed')
    from particle_sliders.pg09 import ConditionalPG09
    from particle_sliders.field_critic import seed_initialization_rng
    device = next(adapter.network.parameters()).device
    devices = _cuda_rng_devices(device)
    with torch.random.fork_rng(devices=devices):
        seed_initialization_rng(seed, devices)
        critic = MingHNDLCritic(tuple(rows[0].target.shape), conditions.shape[1], critic_width)
        critic.to(device)
        return ConditionalPG09(adapter, rows, critic, conditions=conditions,
                               total_steps=total_steps, seed=seed, batch_size=batch_size,
                               response_domain='continuous_field')
