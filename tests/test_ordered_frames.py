"""Ordered native response geometry and actual shared PG0.9.4 software checks."""
from copy import deepcopy
from dataclasses import replace
import pytest
import torch
from tests.test_native import TinyMing, fixture, row, equal
from particle_sliders.native_lora import LoRANetwork
from particle_sliders.pg09_backends import save_adapter, load_adapter

ordered = pytest.importorskip('ming_native.ordered_frames', reason='isolated native ordered-frame product required')


def make():
    caller = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(41)
        model = TinyMing().requires_grad_(False)
        lora = LoRANetwork(model, rank=2, alpha=2, target_replace=['TinyMing'])
        adapter = ordered.OrderedFrameResponseAdapter(model, lora, fixture=True,
            feature_profile=ordered.FEATURE_PROFILE)
        rows = []
        for offset in (0., .3):
            r = row(offset)
            refs = torch.randn(1, 2, 1, 8, 8)
            r = replace(r, latent=r.latent.repeat(1, 1, 2, 1, 1),
                neutral=dict(r.neutral, reference_latents=refs),
                positive=dict(r.positive, reference_latents=refs),
                negative=None)
            rows.append(adapter.prepare(r))
    assert torch.equal(caller, torch.get_rng_state())
    return model, adapter, rows, torch.tensor([[0., .25], [.3, .25]])


def game(**kwargs):
    model, adapter, rows, conditions = make()
    return model, adapter, ordered.build_hndl_game(adapter, rows, conditions,
        feature_profile=ordered.FEATURE_PROFILE, total_steps=4, critic_width=4,
        lr=.0012, d_lr_mult=15., **kwargs)


def test_actual_graph_fixed_frame_order_exact_torch_oracle_and_all_frame_penalty_gradients():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(71)
        critic = ordered.OrderedFrameCritic((1, 2, 2, 8, 8), 3,
            feature_profile=ordered.FEATURE_PROFILE, width=4)
        response = torch.randn(2, 2, 2, 8, 8, requires_grad=True)
        condition = torch.randn(2, 3)
    actual = critic(response, condition)
    graph = critic.graph
    value = torch.nn.functional.silu(graph['field0'](response.reshape(2, 4, 8, 8)))
    value = graph['field2'](value)
    gain, bias = graph['condition'](condition).chunk(2, dim=1)
    value = torch.nn.functional.silu(value*(1+gain[:, :, None, None])+bias[:, :, None, None]).mean(3).mean(2)
    expected = graph['score2'](torch.nn.functional.silu(graph['score0'](value))).squeeze(-1)
    assert torch.equal(actual, expected)
    assert not torch.equal(actual, critic(response.flip(2), condition))
    gradient = torch.autograd.grad(actual.sum(), response, create_graph=True)[0]
    assert all(gradient[:, :, i].abs().sum() > 0 for i in (0, 1))
    gradient.square().sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in critic.parameters())
    assert critic.feature_policy['mapping'] == 'packed_index=channel*target_frames+frame'


def test_real_shared_game_exact_continuation_keeps_native5D_and_frozen_base(monkeypatch):
    torch.set_num_threads(1)
    caller = torch.get_rng_state().clone()
    model, adapter, current = game()
    assert torch.equal(caller, torch.get_rng_state())
    frozen = deepcopy(model.state_dict())
    initial_g = deepcopy(adapter.network.state_dict())
    initial_d = deepcopy(current.critic.state_dict())
    assert current.response_shape == (1, 2, 2, 8, 8)
    assert current.prior is None
    assert set(current.base_lrs[0]) == {.0012} and set(current.base_lrs[1]) == {.018}
    def forbidden(*args, **kwargs): raise AssertionError('No MSE fallback')
    monkeypatch.setattr(torch.nn.functional, 'mse_loss', forbidden)
    current.step()
    packet = deepcopy(current.state_dict())
    _, _, restored = game()
    restored.load_state_dict(packet)
    equal(current.step(), restored.step())
    equal(current.state_dict(), restored.state_dict())
    assert any(not torch.equal(v, adapter.network.state_dict()[k]) for k, v in initial_g.items())
    assert any(not torch.equal(v, current.critic.state_dict()[k]) for k, v in initial_d.items())
    equal(frozen, model.state_dict())
    assert all(p.grad is None for p in model.parameters())
    assert adapter.metadata()['ordered_frame_critic']['profile'] == ordered.FEATURE_PROFILE
    assert adapter.metadata()['native_quality_qualified'] is False


def test_distinct_artifact_identity_refuses_singleton_profile_adapter(tmp_path):
    _, adapter, current = game()
    current.step()
    path = tmp_path/'ordered.safetensors'
    save_adapter(adapter, path, model_fingerprint='tiny-ordered-base')
    _, _, other = fixture()
    with pytest.raises(ValueError, match='protocol, frozen model'):
        load_adapter(other, path, model_fingerprint='tiny-ordered-base')
    _, same, _, _ = make()
    load_adapter(same, path, model_fingerprint='tiny-ordered-base')
    equal(adapter.network.state_dict(), same.network.state_dict())


@pytest.mark.parametrize('shape', [(1,2,1,8,8), (1,2,6,8,8), (2,2,2,8,8), (1,2,True,8,8), (1,2,2,0,8)])
def test_critic_refuses_unbound_frame_geometry(shape):
    with pytest.raises(ValueError, match='ordered target frames'):
        ordered.OrderedFrameCritic(shape, 2, feature_profile=ordered.FEATURE_PROFILE)


@pytest.mark.parametrize('options', [dict(lr=True),dict(lr=0),dict(lr=float('nan')),
    dict(d_lr_mult=-1),dict(d_lr_mult=float('inf')),dict(prior_lr_mult=1)])
def test_invalid_rates_refuse_before_graph(options, monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError('No graph should be allocated')
    monkeypatch.setattr(ordered, 'OrderedFrameCritic', forbidden)
    with pytest.raises(ValueError, match='positive|particle bank'):
        ordered.build_hndl_game(None, [], None, feature_profile=ordered.FEATURE_PROFILE, total_steps=4, **options)


def test_profile_and_invalid_rows_refuse_before_graph(monkeypatch):
    _, adapter, rows, conditions = make()
    def forbidden(*args, **kwargs): raise AssertionError('No graph should be allocated')
    monkeypatch.setattr(ordered, 'OrderedFrameCritic', forbidden)
    with pytest.raises(ValueError, match='explicit'):
        ordered.build_hndl_game(adapter, rows, conditions, total_steps=4)
    with pytest.raises(ValueError, match='identical finite frozen'):
        ordered.build_hndl_game(adapter, [rows[0],replace(rows[1], target=rows[1].target[:, :, :1])],
            conditions, feature_profile=ordered.FEATURE_PROFILE, total_steps=4)
    with pytest.raises(ValueError, match='conditions require'):
        ordered.build_hndl_game(adapter, rows, conditions[:1], feature_profile=ordered.FEATURE_PROFILE, total_steps=4)


def test_fixed_frame_geometry_refuses_changed_batch_feature_shape():
    critic = ordered.OrderedFrameCritic((1,2,2,8,8), 2, feature_profile=ordered.FEATURE_PROFILE, width=4)
    with pytest.raises(ValueError, match='geometry changed'):
        critic(torch.ones(1,2,1,8,8),torch.ones(1,2))
    with pytest.raises(ValueError, match='geometry changed'):
        critic(torch.ones(1,2,2,8,8),torch.ones(1,3))


def test_single_output_native_capability_refuses_before_binding_ownership():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(41)
        model = TinyMing().requires_grad_(False)
        model.multi_frame_output = False
        lora = LoRANetwork(model, rank=2, alpha=2, target_replace=['TinyMing'])
    with pytest.raises(ValueError, match='multi_frame_output=True'):
        ordered.OrderedFrameResponseAdapter(model, lora, fixture=True, feature_profile=ordered.FEATURE_PROFILE)
    assert all(not hasattr(m, '_ming_response_binding') for m in model.modules())
