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


@pytest.mark.parametrize('damage', ['hndl-version', 'hg-version', 'hg-origin', 'constraint'])
def test_candidate_checks_actual_hg_hndl_and_declared_constraints(monkeypatch, damage):
    from importlib.metadata import distribution
    from ming import cohort
    class Changed:
        def __init__(self, name): self.actual = distribution(name); self.name = name
        @property
        def version(self):
            if (damage == 'hndl-version' and self.name == 'hndl' or
                    damage == 'hg-version' and self.name == 'hypergan'): return '0.0.0'
            return self.actual.version
        @property
        def requires(self):
            return ['torch>=999'] if damage == 'constraint' and self.name == 'hypergan' else self.actual.requires
        def read_text(self, name):
            return '{}' if damage == 'hg-origin' and self.name == 'hypergan' else self.actual.read_text(name)
        def locate_file(self, name): return self.actual.locate_file(name)
    monkeypatch.setattr(cohort, 'distribution', Changed)
    with pytest.raises(RuntimeError, match='HyperGAN|constraint'):
        cohort.require_cohort('native-candidate-0.12')


def test_candidate_checks_actual_hndl_source_closure(monkeypatch, tmp_path):
    import hndl
    from ming.cohort import require_cohort
    fake = tmp_path / '__init__.py'
    fake.write_text('# modified same-version HNDL implementation\n')
    monkeypatch.setattr(hndl, '__file__', str(fake))
    with pytest.raises(ValueError, match='HNDL source closure'):
        require_cohort('native-candidate-0.12')


@pytest.mark.parametrize('damage', ['list', 'nan', 'rows', 'requires-grad', 'integer', 'rank'])
def test_invalid_game_conditions_refuse_before_graph_allocation(monkeypatch, damage):
    import ming.native as native
    _, _, adapter = fixture(profile='native-candidate-0.12')
    rows = [adapter.prepare(row(o)) for o in (0., .3)]
    conditions = torch.ones(2, 2)
    if damage == 'list': conditions = [[1., 1.], [1., 1.]]
    elif damage == 'nan': conditions[0, 0] = float('nan')
    elif damage == 'rows': conditions = conditions[:1]
    elif damage == 'requires-grad': conditions.requires_grad_(True)
    elif damage == 'integer': conditions = conditions.long()
    else: conditions = conditions.flatten()
    def forbidden(*a, **kw): raise AssertionError('critic must not allocate')
    monkeypatch.setattr(native, 'MingHNDLCritic', forbidden)
    with pytest.raises(ValueError, match='finite frozen FP32'):
        build_hndl_game(adapter, rows, conditions, total_steps=4)


def test_candidate_wheel_metadata_matches_its_runtime():
    from importlib.metadata import distribution
    from packaging.requirements import Requirement
    product = distribution('ming-particle-sliders')
    assert product.version.endswith('+native012')
    requirements = {r.name: r for r in map(Requirement, product.requires) if r.marker is None}
    assert requirements['particle-sliders-core'].url == (
        'git+https://github.com/HyperGAN/particle-sliders.git@'
        '4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231#subdirectory=packages/particle-sliders-core')
    assert str(requirements['particlegan'].specifier) == '==0.9.4'
    assert str(requirements['hndl'].specifier) == '==0.10.0'
    assert requirements['hypergan'].url.endswith('5f368a3770cf5bdae0448995c9f76e29af8049f7')


@pytest.mark.parametrize('damage', ['one-row', 'target-grad', 'neutral-nan', 'target-shape'])
def test_invalid_prepared_rows_refuse_before_graph_allocation(monkeypatch, damage):
    from dataclasses import replace
    import ming.native as native
    _, _, adapter = fixture(profile='native-candidate-0.12')
    rows = [adapter.prepare(row(o)) for o in (0., .3)]
    if damage == 'one-row': rows = rows[:1]
    elif damage == 'target-grad': rows[0] = replace(rows[0], target=rows[0].target.requires_grad_())
    elif damage == 'neutral-nan': rows[0].neutral.fill_(float('nan'))
    else: rows[0] = replace(rows[0], target=rows[0].target[:, :, :, :2])
    def forbidden(*a, **kw): raise AssertionError('critic must not allocate')
    monkeypatch.setattr(native, 'MingHNDLCritic', forbidden)
    with pytest.raises(ValueError, match='prepared frozen|finite frozen'):
        build_hndl_game(adapter, rows, torch.ones(len(rows), 2), total_steps=4)
