"""Explicit candidate lane: real installed HyperGAN develop/HNDL graphs on CPU."""
from copy import deepcopy
from importlib.metadata import version

import pytest
import torch

from ming.native import MingHNDLCritic, build_hndl_game
from tests.test_native import fixture, row, equal

pytestmark = pytest.mark.skipif(version('particle-sliders-core') != '0.12.0',
                                reason='separate exact core0.12/PG0.9.4/HyperGAN/HNDL environment required')


def game():
    model, lora, adapter = fixture(profile='native-candidate-0.12')
    rows = [adapter.prepare(row(offset)) for offset in (0., .3)]
    conditions = torch.tensor([[0., .25], [.3, .25]])  # actual fixture embedding/time features
    return model, adapter, build_hndl_game(adapter, rows, conditions,
        total_steps=4, critic_width=4)


def test_real_hypergan_hndl_shared_game_step_and_deterministic_restore(monkeypatch):
    torch.set_num_threads(1)
    initial_rng = torch.get_rng_state().clone()
    model, adapter, current = game()
    assert torch.equal(initial_rng, torch.get_rng_state())
    base = deepcopy(model.state_dict())
    before = deepcopy(adapter.network.state_dict())
    critic_before = deepcopy(current.critic.state_dict())
    identity = current.critic.critic._identity
    assert identity['profile'] == 'hypergan-hndl-v2'
    assert identity['hypergan_commit'] == '5f368a3770cf5bdae0448995c9f76e29af8049f7'
    assert identity['hndl_version'] == '0.10.0'
    def forbidden(*a, **kw):
        raise AssertionError('no MSE training fallback')
    monkeypatch.setattr(torch.nn.functional, 'mse_loss', forbidden)
    metrics = current.step()
    assert metrics['g_loss'] == pytest.approx(metrics['g_adv'] + metrics['prior_regularization'])
    assert metrics['d_loss'] == pytest.approx(metrics['d_adv'] + metrics['d_pen'])
    assert any(not torch.equal(v, adapter.network.state_dict()[k]) for k, v in before.items())
    assert any(not torch.equal(v, current.critic.state_dict()[k]) for k, v in critic_before.items())
    equal(base, model.state_dict())
    assert all(p.grad is None for p in model.parameters())
    packet = deepcopy(current.state_dict())
    _, _, restored = game()
    restored.load_state_dict(packet)
    equal(current.step(), restored.step())
    equal(current.state_dict(), restored.state_dict())
    unchanged = deepcopy(restored.state_dict())
    bad = deepcopy(packet)
    bad['config']['pg_version'] = '0.9.3'
    with pytest.raises(ValueError, match='original pinned'):
        restored.load_state_dict(bad)
    equal(unchanged, restored.state_dict())


def test_candidate_does_not_silently_retarget_the_default_stamp():
    from ming.surfaces import stamp
    with pytest.raises(RuntimeError, match='exact packaged'):
        stamp()


def test_multiframe_field_is_refused_without_flattening():
    with pytest.raises(ValueError, match='one genuine latent frame'):
        MingHNDLCritic((1, 2, 3, 8, 8), 2)
