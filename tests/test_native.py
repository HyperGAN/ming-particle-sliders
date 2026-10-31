"""Tiny CPU contract probes, never pretrained/native quality qualification."""
from copy import deepcopy
from dataclasses import replace
from importlib.metadata import version

import pytest
import torch
from torch import nn

from ming.native import MingResponseAdapter
from particle_sliders.native_lora import LoRANetwork
from particle_sliders.pg09_backends import ResponseRow, FlowSchedule, save_adapter, load_adapter

PROFILE = 'native-candidate-0.12' if version('particle-sliders-core') == '0.12.0' else 'public-0.11'


class TinyMing(nn.Module):
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
