"""Tiny CPU contract probes, never pretrained/native quality qualification."""
from copy import deepcopy
from dataclasses import replace
from importlib.metadata import distribution, PackageNotFoundError
from importlib import import_module

import pytest
import torch
from torch import nn

try:
    distribution('ming-particle-sliders-native')
except PackageNotFoundError:
    NAMESPACE = 'ming'
else:
    NAMESPACE = 'ming_native'
NATIVE = import_module(NAMESPACE + '.native')
COHORT = import_module(NAMESPACE + '.cohort')
MingResponseAdapter = NATIVE.MingResponseAdapter
from particle_sliders.native_lora import LoRANetwork
from particle_sliders.pg09_backends import ResponseRow, FlowSchedule, save_adapter, load_adapter

PROFILE = COHORT.product_identity()['profile']


class TinyMing(nn.Module):
    multi_frame_output = True
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)
        self.calls = []

    def forward(self, hidden_states, timestep, encoder_hidden_states, *, return_dict,
                encoder_hidden_states_2=None, ref_hidden_states=None):
        self.calls.append((timestep.detach().clone(), encoder_hidden_states, ref_hidden_states))
        values = []
        for i, x in enumerate(hidden_states):
            value = x.permute(1, 2, 3, 0)
            if getattr(self, 'is_gradient_checkpointing', False) and torch.is_grad_enabled():
                value = self._gradient_checkpointing_func(self.proj, value)
            else:
                value = self.proj(value)
            y = value.permute(3, 0, 1, 2)
            y = y + encoder_hidden_states[i].mean() + timestep[i]
            if encoder_hidden_states_2 is not None:
                y = y + encoder_hidden_states_2[i].mean()
            if ref_hidden_states is not None:
                y = y + ref_hidden_states[i].mean()
                y = torch.cat([y, ref_hidden_states[i]], dim=1)
            if not self.multi_frame_output:
                y = y[:, :1]
            values.append(y)
        return (values,)


class CheckpointMing(TinyMing):
    is_gradient_checkpointing = True

    def enable_gradient_checkpointing(self, *, gradient_checkpointing_func):
        self._gradient_checkpointing_func = gradient_checkpointing_func


def fixture(*, fixture_mode=True, profile=PROFILE, model_type=TinyMing):
    with torch.random.fork_rng():
        torch.manual_seed(41)
        model = model_type().requires_grad_(False)
        lora = LoRANetwork(model, rank=2, alpha=2, target_replace=[model_type.__name__])
    adapter = MingResponseAdapter(model, lora, fixture=fixture_mode, cohort=profile)
    return model, lora, adapter


def row(offset=0.):
    return ResponseRow(
        neutral={'embeds': torch.ones(1, 2, 3) * offset},
        positive={'embeds': torch.ones(1, 2, 3) * (offset + .2)},
        negative={'embeds': torch.zeros(1, 2, 3)},
        latent=torch.linspace(-.5, .5, 128).reshape(1, 2, 1, 8, 8),
        timestep=torch.tensor(250.),
    )


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for left, right in zip(a, b):
            equal(left, right)
    else:
        assert a == b


def test_native_time_sign_zero_and_adapter_gradient_ownership():
    model, lora, adapter = fixture()
    data = row()
    base = deepcopy(model.state_dict())
    with torch.no_grad():
        for layer in lora.unet_loras:
            layer.lora_up.weight.fill_(.1)
    with adapter.network.scaled(0):
        native = -torch.stack(model(list(data.latent.unbind(0)), torch.tensor([.75]),
                                   list(data.neutral['embeds'].unbind(0)), return_dict=False)[0])
    zero = adapter.velocity(data.neutral, data.latent, data.timestep, scale=0.)
    assert torch.equal(zero, native)
    assert torch.equal(model.calls[-1][0], torch.tensor([.75]))
    generated = adapter.velocity(data.neutral, data.latent, data.timestep)
    assert generated.shape == data.latent.shape
    assert not torch.equal(generated, zero)
    generated.sum().backward()
    assert all(p.grad is None for p in model.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.network.parameters())
    assert any(p.grad.abs().sum() > 0 for p in adapter.network.parameters())
    assert {id(p) for p in adapter.network.parameters()} == {id(p) for p in lora.parameters()}
    equal(base, model.state_dict())
    assert adapter.network.scale == 1.


def test_frozen_condition_reference_snapshot_and_export_restore(tmp_path):
    model, lora, adapter = fixture()
    data = row()
    for conditioning in (data.neutral, data.positive, data.negative):
        conditioning['embeds_2'] = torch.ones(1, 2, 3)
        conditioning['reference_latents'] = torch.ones(1, 2, 2, 8, 8)
    prepared = adapter.prepare(data)
    assert not prepared.target.requires_grad
    before = adapter.generated(prepared).detach()
    data.neutral['embeds'].fill_(99.)
    data.neutral['reference_latents'].fill_(99.)
    assert torch.equal(before, adapter.generated(prepared))
    path = tmp_path / 'fixture.safetensors'
    saved = deepcopy(adapter.network.state_dict())
    save_adapter(adapter, path, model_fingerprint='tiny-frozen-base-41')
    with torch.no_grad():
        for p in adapter.network.parameters():
            p.add_(1.)
    changed = deepcopy(adapter.network.state_dict())
    with pytest.raises(ValueError, match='protocol, frozen model'):
        load_adapter(adapter, path, model_fingerprint='wrong-base')
    equal(changed, adapter.network.state_dict())
    load_adapter(adapter, path, model_fingerprint='tiny-frozen-base-41')
    equal(saved, adapter.network.state_dict())
    assert adapter.metadata()['native_quality_qualified'] is False
    assert adapter.metadata()['fixture'] is True


@pytest.mark.parametrize('change,match', [
    ({'guidance': 2.}, 'single-timestep'),
    ({'schedule': FlowSchedule.linear(2, time_scale=1000), 'timestep': None}, 'single-timestep'),
    ({'latent': torch.ones(1, 128)}, 'genuine'),
    ({'latent': torch.ones(1, 2, 1, 7, 8)}, 'patches'),
    ({'timestep': torch.tensor(float('nan'))}, 'finite timestep'),
    ({'timestep': torch.tensor(1001.)}, 'scheduler timestep'),
    ({'neutral': {'embeds': torch.ones(1, 2, 3), 'unknown': True}}, 'frozen native'),
    ({'neutral': {'embeds': torch.ones(1, 2, 3), 'reference_latents': torch.ones(1, 3, 1, 8, 8)}}, 'geometry'),
    ({'scale': -1.}, 'unipolar'),
])
def test_unqualified_contracts_refuse(change, match):
    _, _, adapter = fixture()
    with pytest.raises(ValueError, match=match):
        adapter.prepare(replace(row(), **change))


def test_fixture_does_not_qualify_a_native_model():
    with pytest.raises(ValueError, match='immutable Ming'):
        fixture(fixture_mode=False)


def test_lora_must_be_attached_to_actual_model():
    model, lora, _ = fixture()
    other = TinyMing().requires_grad_(False)
    with pytest.raises(ValueError, match='actual staged model'):
        MingResponseAdapter(other, lora, fixture=True, cohort=PROFILE)


def test_scale_restores_after_native_forward_error(monkeypatch):
    model, lora, adapter = fixture()
    before = [layer.multiplier for layer in lora.unet_loras]
    def broken(*args, **kwargs):
        raise RuntimeError('fixture forward failed')
    monkeypatch.setattr(model, 'forward', broken)
    with pytest.raises(RuntimeError, match='fixture forward failed'):
        adapter.velocity(row().neutral, row().latent, row().timestep, scale=0)
    assert [layer.multiplier for layer in lora.unet_loras] == before
    assert adapter.network.scale == 1.


def test_native_checkpoint_recomputation_captures_each_forward_scale():
    adapters = []
    for model_type in (TinyMing, CheckpointMing):
        _, lora, adapter = fixture(model_type=model_type)
        with torch.no_grad():
            for layer in lora.unet_loras:
                layer.lora_up.weight.fill_(.1)
        data = row()
        outputs = [adapter.velocity(data.neutral, data.latent, data.timestep, scale=s) for s in (.5, 1.)]
        # Backward occurs after both temporary scale contexts have exited.
        sum(value.sum() for value in outputs).backward()
        assert adapter.network.scale == 1.
        adapters.append(adapter)
    for plain, checkpointed in zip(adapters[0].network.parameters(), adapters[1].network.parameters()):
        assert torch.equal(plain.grad, checkpointed.grad)


def test_multiple_target_and_reference_frames_follow_official_order():
    model, _, adapter = fixture()
    data = replace(row(), latent=row().latent.repeat(1, 1, 2, 1, 1))
    refs = torch.cat([torch.full_like(row().latent, 10.), torch.full_like(row().latent, 20.)], dim=2)
    for condition in (data.neutral, data.positive, data.negative):
        condition['reference_latents'] = refs.clone()
    prepared = adapter.prepare(data)
    with adapter.network.scaled(0):
        raw = model(list(data.latent.unbind(0)), torch.tensor([.75]),
                    list(data.positive['embeds'].unbind(0)), return_dict=False,
                    ref_hidden_states=list(refs.unbind(0)))[0][0]
    assert raw.shape[1] == 4  # target0,target1,reference0,reference1
    assert torch.equal(raw[:, 2:], refs[0])
    assert torch.equal(prepared.target, -raw[:, :2].unsqueeze(0))
    assert torch.equal(model.calls[-1][2][0], refs[0])


def test_wrong_native_output_frame_count_refuses(monkeypatch):
    model, _, adapter = fixture()
    data = row()
    data.neutral['reference_latents'] = torch.ones(1, 2, 2, 8, 8)
    original = model.forward
    def truncated(*a, **kw):
        values = original(*a, **kw)[0]
        return ([values[0][:, :-1]],)
    monkeypatch.setattr(model, 'forward', truncated)
    with pytest.raises(ValueError, match='preserve target'):
        adapter.velocity(data.neutral, data.latent, data.timestep)


class SingleFrameMing(TinyMing):
    multi_frame_output = False


def test_official_single_frame_capability_with_multiple_references():
    model, _, adapter = fixture(model_type=SingleFrameMing)
    data = row()
    for c in (data.neutral, data.positive, data.negative):
        c['reference_latents'] = torch.ones(1, 2, 2, 8, 8)
    assert adapter.prepare(data).target.shape == data.latent.shape
    model.calls.clear()
    with pytest.raises(ValueError, match='single-frame native capability'):
        adapter.prepare(replace(data, latent=data.latent.repeat(1, 1, 2, 1, 1)))
    assert not model.calls


@pytest.mark.parametrize('key', ['embeds', 'embeds_2', 'reference_latents'])
def test_native_encodings_require_frozen_tensors(key):
    _, _, adapter = fixture()
    data = row()
    shape = (1, 2, 1, 8, 8) if key == 'reference_latents' else (1, 2, 3)
    data.neutral[key] = torch.ones(shape, requires_grad=True)
    with pytest.raises(ValueError, match='native tensor'):
        adapter.prepare(data)


def test_borrowed_checkpoint_callback_and_modes_restore_on_close():
    model = CheckpointMing().requires_grad_(False)
    original = object()
    model._gradient_checkpointing_func = original
    model.train()
    model.proj.eval()
    modes = [m.training for m in model.modules()]
    lora = LoRANetwork(model, rank=2, alpha=2, target_replace=['CheckpointMing'])
    with MingResponseAdapter(model, lora, fixture=True, cohort=PROFILE) as adapter:
        assert model._gradient_checkpointing_func is not original
        adapter.prepare(row())
    assert model._gradient_checkpointing_func is original
    assert [m.training for m in model.modules()] == modes
    adapter.close()  # idempotent; caller-owned LoRA remains attached
    with pytest.raises(RuntimeError, match='closed'):
        adapter.prepare(row())


def test_checkpoint_binding_failure_restores_borrowed_state(monkeypatch):
    native = NATIVE
    model = CheckpointMing().requires_grad_(False)
    lora = LoRANetwork(model, rank=2, alpha=2, target_replace=['CheckpointMing'])
    def fail(root, network):
        root._gradient_checkpointing_func = object()
        root.gradient_checkpointing = True
        raise RuntimeError('checkpoint hook rejected')
    monkeypatch.setattr(native, 'bind_native_gradient_checkpointing', fail)
    with pytest.raises(RuntimeError, match='checkpoint hook rejected'):
        MingResponseAdapter(model, lora, fixture=True, cohort=PROFILE)
    assert '_gradient_checkpointing_func' not in model.__dict__
    assert 'gradient_checkpointing' not in model.__dict__
    assert model.training is True


def test_bf16_frozen_base_owns_only_fp32_adapter_gradients():
    model = TinyMing().to(torch.bfloat16).requires_grad_(False)
    base = deepcopy(model.state_dict())
    lora = LoRANetwork(model, rank=2, alpha=2, target_replace=['TinyMing'])
    with MingResponseAdapter(model, lora, fixture=True, cohort=PROFILE) as adapter:
        prepared = adapter.prepare(row())
        generated = adapter.generated(prepared)
        assert generated.dtype == torch.float32
        generated.sum().backward()
        assert all(p.dtype == torch.float32 and p.grad is not None for p in adapter.network.parameters())
        assert all(p.grad is None and not p.requires_grad for p in model.parameters())
        equal(base, model.state_dict())


@pytest.mark.parametrize('damage', ['unfrozen', 'mixed-base', 'adapter-bf16'])
def test_unqualified_dtype_and_base_ownership_refuse(damage):
    model = TinyMing().requires_grad_(False)
    lora = LoRANetwork(model, rank=2, alpha=2, target_replace=['TinyMing'])
    if damage == 'unfrozen': model.proj.weight.requires_grad_(True)
    elif damage == 'mixed-base': model.proj.weight.data = model.proj.weight.data.bfloat16()
    else: lora.to(torch.bfloat16)
    with pytest.raises(ValueError, match='frozen|one FP32|float32'):
        MingResponseAdapter(model, lora, fixture=True, cohort=PROFILE)
    assert model.training is True


def test_bare_cuda_rng_normalization_without_cuda_execution(monkeypatch):
    _cuda_rng_devices = NATIVE._cuda_rng_devices
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 3)
    assert _cuda_rng_devices(torch.device('cuda')) == [3]
    assert _cuda_rng_devices(torch.device('cuda:1')) == [1]
    assert _cuda_rng_devices(torch.device('cpu')) == []


def test_overlapping_borrow_is_rejected_without_changing_first_hook():
    model, lora, first = fixture(model_type=CheckpointMing)
    callback = model._gradient_checkpointing_func
    with pytest.raises(ValueError, match='already borrowed'):
        MingResponseAdapter(model, lora, fixture=True, cohort=PROFILE)
    assert model._gradient_checkpointing_func is callback
    first.prepare(row())
    first.close()
    with MingResponseAdapter(model, lora, fixture=True, cohort=PROFILE) as second:
        second.prepare(row())


def test_unipolar_negative_none_is_optional_and_unused():
    _, _, adapter = fixture()
    data = row()
    normal = adapter.prepare(data)
    without_negative = adapter.prepare(replace(data, negative=None))
    assert torch.equal(normal.target, without_negative.target)
    assert torch.equal(normal.neutral, without_negative.neutral)
    assert torch.equal(adapter.generated(normal), adapter.generated(without_negative))
    assert 'optional None' in adapter.metadata()['negative_condition']
    with pytest.raises(ValueError, match='single-timestep'):
        adapter.prepare(replace(data, negative=None, guidance=2.))


class WrapperMing(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.transformer = inner
    def forward(self, *args, **kwargs):
        return self.transformer(*args, **kwargs)


def test_different_wrappers_cannot_borrow_the_same_inner_model():
    inner = CheckpointMing().requires_grad_(False)
    lora = LoRANetwork(inner, rank=2, alpha=2, target_replace=['CheckpointMing'])
    first_root, second_root = WrapperMing(inner), WrapperMing(inner)
    with MingResponseAdapter(first_root, lora, fixture=True, cohort=PROFILE) as first:
        callback = inner._gradient_checkpointing_func
        with pytest.raises(ValueError, match='inner model is already borrowed'):
            MingResponseAdapter(second_root, lora, fixture=True, cohort=PROFILE)
        assert inner._gradient_checkpointing_func is callback
        first.prepare(row())
    with MingResponseAdapter(second_root, lora, fixture=True, cohort=PROFILE) as second:
        second.prepare(row())


class FailingRestoreMing(CheckpointMing):
    fail_mode = False
    fail_hook = False
    original_callback = None
    def __setattr__(self, name, value):
        if name == 'training' and value is True and self.fail_mode:
            raise RuntimeError('mode restoration refused')
        if name == '_gradient_checkpointing_func' and value is self.original_callback and self.fail_hook:
            raise RuntimeError('hook restoration refused')
        super().__setattr__(name, value)


def test_cleanup_attempts_every_action_aggregates_failures_and_can_retry():
    model = FailingRestoreMing().requires_grad_(False)
    original = object()
    model.original_callback = original
    model._gradient_checkpointing_func = original
    model.gradient_checkpointing = False
    lora = LoRANetwork(model, rank=2, alpha=2, target_replace=['FailingRestoreMing'])
    adapter = MingResponseAdapter(model, lora, fixture=True, cohort=PROFILE)
    model.gradient_checkpointing = True
    model.fail_mode = model.fail_hook = True
    with pytest.raises(BaseExceptionGroup, match='all restoration actions attempted') as failure:
        adapter.close()
    assert len(failure.value.exceptions) == 2
    assert model.gradient_checkpointing is False  # action after failing callback still ran
    assert model.proj.training is True  # another module still restored
    assert adapter._closed and adapter._cleanup_failed
    with pytest.raises(ValueError, match='already borrowed'):
        MingResponseAdapter(model, lora, fixture=True, cohort=PROFILE)
    model.fail_mode = model.fail_hook = False
    adapter.close()
    assert model._gradient_checkpointing_func is original and model.training
    assert not adapter._cleanup_failed
    assert '_ming_response_binding' not in model.__dict__
    assert '_ming_response_binding' not in model.proj.__dict__


def test_body_and_cleanup_errors_both_survive():
    model = FailingRestoreMing().requires_grad_(False)
    lora = LoRANetwork(model, rank=2, alpha=2, target_replace=['FailingRestoreMing'])
    adapter = MingResponseAdapter(model, lora, fixture=True, cohort=PROFILE)
    with pytest.raises(BaseExceptionGroup, match='body and cleanup both failed') as failure:
        with adapter:
            model.fail_mode = True
            raise ValueError('body failure retained')
    assert isinstance(failure.value.exceptions[0], ValueError)
    assert 'body failure retained' in str(failure.value.exceptions[0])
    model.fail_mode = False
    adapter.close()


def test_constructor_and_cleanup_errors_both_survive(monkeypatch):
    model = FailingRestoreMing().requires_grad_(False)
    lora = LoRANetwork(model, rank=2, alpha=2, target_replace=['FailingRestoreMing'])
    def fail(root, network):
        root.fail_mode = True
        raise ValueError('original setup failure retained')
    monkeypatch.setattr(NATIVE, 'bind_native_gradient_checkpointing', fail)
    with pytest.raises(BaseExceptionGroup, match='setup and cleanup both failed') as failure:
        MingResponseAdapter(model, lora, fixture=True, cohort=PROFILE)
    assert isinstance(failure.value.exceptions[0], ValueError)
    assert 'original setup failure retained' in str(failure.value.exceptions[0])
    assert model.proj.training is True
    owner = model._ming_response_binding
    model.fail_mode = False
    owner.close()
