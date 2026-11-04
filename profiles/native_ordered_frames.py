"""Explicit ordered-frame critic binding; shared PG owns the native 5D game.

This profile qualifies software mechanics only. Pretrained Ming conditioning,
CFG sampling, layer decomposition, rendering and learning rates remain pending.
The existing singleton-frame binding retains its original admission rules.
"""
import hashlib
import torch
from .cohort import require_cohort
from .native import MingResponseAdapter, PreparedResponse, _cuda_rng_devices, learning_rate_overrides

FEATURE_PROFILE = 'ming-ordered-target-frames-v1'
GRAPH = '''field = reshape(response, $packed_channels, $height, $width_px)
field = conv(field, $width, kernel_size=1, name="field0")
field = silu(field)
field = conv(field, $width, kernel_size=3, padding=1, name="field2")
context = linear(condition, $double_width, name="condition")
gain, bias = split(context, $width)
ones = constant(gain, $width, value=1.0)
gain = add(gain, ones)
gain = reshape(gain, $width, 1, 1)
bias = reshape(bias, $width, 1, 1)
field = broadcast_mul(field, gain)
field = broadcast_add(field, bias)
field = silu(field)
field = mean(field, 3)
field = mean(field, 2)
score = linear(field, $width, name="score0")
score = silu(score)
linear(score, 1, name="score2")
'''


def _feature_identity():
    return dict(profile=FEATURE_PROFILE, graph_sha256=hashlib.sha256(GRAPH.encode()).hexdigest(),
        mapping='packed_index=channel*target_frames+frame',
        native_response='[B,C,F,H,W]; all target frames retained by PG and its input penalty',
        target_frames='fixed ordered2..5, identical count/geometry in every row',
        qualification='Software response mechanics only; pretrained layered outputs and rates unqualified')


def _require_profile(profile):
    if profile != FEATURE_PROFILE:
        raise ValueError('explicit '+FEATURE_PROFILE+' feature profile required')


class OrderedFrameResponseAdapter(MingResponseAdapter):
    """Distinct artifact identity for the opt-in ordered-target-frame critic."""

    def __init__(self, model, native_lora, *, feature_profile, **kwargs):
        _require_profile(feature_profile)
        if getattr(getattr(model, 'transformer', model), 'multi_frame_output', None) is not True:
            raise ValueError('ordered-frame adapter requires native multi_frame_output=True')
        self.feature_profile = feature_profile
        super().__init__(model, native_lora, **kwargs)

    def metadata(self):
        return dict(super().metadata(), ordered_frame_critic=_feature_identity())


class OrderedFrameCritic(torch.nn.Module):
    """Group fixed channel/frame coordinates inside the critic feature graph.

    Responses and PG penalties retain their native five axes. Frames are never
    moved into the batch or averaged together. Each frame has distinct learned
    input coordinates before the spatial convolutions.
    """
    score_bound = 0

    def __init__(self, shape, condition_width, *, feature_profile, width=32):
        super().__init__()
        _require_profile(feature_profile)
        require_cohort('native-pg094-ddb41423')
        if (not isinstance(shape, (tuple, list)) or len(shape) != 5 or shape[0] != 1
                or any(type(v) is not int or v < 1 for v in (*shape, condition_width, width))
                or not 2 <= shape[2] <= 5):
            raise ValueError('native [1,C,F,H,W] with2..5 ordered target frames and positive integer widths required')
        from particle_sliders.fibo_hndl import verified_builder
        builder, identity = verified_builder()
        self.native_shape = tuple(shape[1:])
        self.condition_width = condition_width
        self.graph = builder(GRAPH,
            input_shape={'response': ['B', *shape[1:]], 'condition': ['B', condition_width]},
            output_shape=['B', 1], parameters=dict(packed_channels=shape[1]*shape[2],
                height=shape[3], width_px=shape[4], width=width, double_width=2*width))
        self.feature_policy = dict(_feature_identity(), native_response_shape=list(shape), builder=identity)

    def forward(self, response, condition):
        if (not isinstance(response, torch.Tensor) or response.ndim != 5
                or tuple(response.shape[1:]) != self.native_shape
                or not isinstance(condition, torch.Tensor)
                or condition.shape != (response.shape[0], self.condition_width)):
            raise ValueError('ordered native target-frame or condition geometry changed')
        return self.graph(response=response, condition=condition).squeeze(-1)


def build_hndl_game(adapter, prepared_rows, conditions, *, feature_profile=None,
                    total_steps, seed=7, batch_size=2, critic_width=32,
                    lr=None, d_lr_mult=None, prior_lr_mult=None):
    """Opt-in native 5D field game with the unchanged official PG0.9.4 recipe."""
    rates = learning_rate_overrides(lr=lr, d_lr_mult=d_lr_mult, prior_lr_mult=prior_lr_mult)
    _require_profile(feature_profile)
    require_cohort('native-pg094-ddb41423')
    if (not isinstance(adapter, OrderedFrameResponseAdapter)
            or adapter.feature_profile != feature_profile
            or adapter.multi_frame_output is not True
            or adapter.cohort != 'native-pg094-ddb41423'):
        raise ValueError('ordered-frame game requires its explicitly bound native adapter')
    adapter._require_open()
    rows = tuple(prepared_rows)
    if len(rows) < 2 or any(not isinstance(r, PreparedResponse) for r in rows):
        raise ValueError('at least two prepared frozen Ming response rows required')
    if any(not isinstance(t, torch.Tensor) for r in rows for t in (r.target, r.neutral)):
        raise ValueError('Ming response rows require frozen tensors')
    shape = tuple(rows[0].target.shape)
    if (len(shape) != 5 or shape[0] != 1 or not 2 <= shape[2] <= 5
            or any(tuple(t.shape) != shape or t.dtype != torch.float32 or t.requires_grad or not torch.isfinite(t).all()
                   for r in rows for t in (r.target, r.neutral))):
        raise ValueError('Ming ordered rows require identical finite frozen FP32 native5D geometry with2..5 target frames')
    if (not isinstance(conditions, torch.Tensor) or conditions.ndim != 2
            or conditions.shape[0] != len(rows) or conditions.shape[1] < 1
            or conditions.dtype != torch.float32 or conditions.requires_grad or not torch.isfinite(conditions).all()):
        raise ValueError('Ming conditions require finite frozen FP32 [row_count, feature_width] tensor')
    if type(seed) is not int or not 0 <= seed < 2**63 - 100:
        raise ValueError('invalid seed')
    from particle_sliders.pg09 import ConditionalPG09
    from particle_sliders.field_critic import seed_initialization_rng
    device = next(adapter.network.parameters()).device
    devices = _cuda_rng_devices(device)
    with torch.random.fork_rng(devices=devices):
        seed_initialization_rng(seed, devices)
        critic = OrderedFrameCritic(shape, conditions.shape[1], feature_profile=feature_profile, width=critic_width).to(device)
        return ConditionalPG09(adapter, rows, critic, conditions=conditions.detach().clone(),
            total_steps=total_steps, seed=seed, batch_size=batch_size,
            response_domain='continuous_field', recipe_overrides=rates)
