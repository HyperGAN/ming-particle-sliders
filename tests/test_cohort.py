"""Installed resource/source admission and historical default separation."""
from importlib.metadata import distribution
from importlib.resources import files
import json
from pathlib import Path

import pytest

from ming import cohort


def test_packaged_public_lock_is_exact_root_lock():
    assert files('ming').joinpath('core.lock.json').read_bytes() == (Path(__file__).resolve().parents[1] / 'core.lock.json').read_bytes()
    assert cohort.require_cohort()['commit'] == 'ddb41423bcefc1fe76cd94d0f9b367353b3ea137'


@pytest.mark.parametrize('damage', ['version', 'origin', 'source'])
def test_runtime_refuses_unreviewed_core_before_the_stamp(monkeypatch, tmp_path, damage):
    actual = distribution('particle-sliders-core')
    corrupt = tmp_path / 'source.py'
    corrupt.write_text('# changed source\n')
    class Distribution:
        version = '0.0.0' if damage == 'version' else actual.version
        def read_text(self, name):
            return '{}' if damage == 'origin' else actual.read_text(name)
        def locate_file(self, name):
            return corrupt if damage == 'source' else actual.locate_file(name)
    monkeypatch.setattr(cohort, 'distribution', lambda name: Distribution() if name == 'particle-sliders-core' else distribution(name))
    with pytest.raises(RuntimeError, match='cohort'):
        cohort.require_cohort()


def test_stable_declares_one_profile_and_ships_only_its_own_lock():
    assert cohort.product_identity()['distribution'] == 'ming-particle-sliders'
    assert cohort.product_identity()['profile'] == 'public-pg094-ddb41423'
    assert not files('ming').joinpath('native-candidate.lock.json').is_file()
    with pytest.raises(ValueError, match='no implicit upgrade'):
        cohort.require_cohort('native-candidate-0.12')


def test_pg_same_version_modified_sources_refuse(monkeypatch, tmp_path):
    actual = distribution('particlegan')
    fake_root = tmp_path / 'particlegan'
    fake_root.mkdir()
    (fake_root / '__init__.py').write_text('# same version, different implementation\n')
    class ChangedPG:
        version = actual.version
        def locate_file(self, name): return fake_root
    monkeypatch.setattr(cohort, 'distribution', lambda n: ChangedPG() if n == 'particlegan' else distribution(n))
    with pytest.raises(RuntimeError, match='ParticleGAN source closure'):
        cohort.require_cohort()


def test_missing_core_dependency_has_actionable_error(monkeypatch):
    from importlib.metadata import PackageNotFoundError
    def missing(name):
        if name == 'particle-sliders-core': raise PackageNotFoundError(name)
        return distribution(name)
    monkeypatch.setattr(cohort, 'distribution', missing)
    with pytest.raises(cohort.CohortError, match="missing dependency 'particle-sliders-core'.*own environment"):
        cohort.require_cohort()


@pytest.mark.parametrize('damage', ['product-version', 'declared-core-url'])
def test_installed_product_identity_and_declared_dependency_are_bound(monkeypatch, damage):
    actual = distribution('ming-particle-sliders')
    class Changed:
        version = '99.0' if damage == 'product-version' else actual.version
        requires = ['particle-sliders-core @ git+https://github.com/HyperGAN/particle-sliders.git@main#subdirectory=packages/particle-sliders-core'] if damage == 'declared-core-url' else actual.requires
        def locate_file(self, name): return actual.locate_file(name)
    monkeypatch.setattr(cohort, 'distribution', lambda n: Changed() if n == 'ming-particle-sliders' else distribution(n))
    with pytest.raises(cohort.CohortError, match='product version|declared core cohort'):
        cohort.require_cohort()


def test_identity_cannot_redeclare_another_lock_digest(monkeypatch):
    original = cohort._json
    def changed(text, label):
        result = original(text, label)
        if label == 'product identity': result['lock_sha256'] = '0' * 64
        return result
    monkeypatch.setattr(cohort, '_json', changed)
    with pytest.raises(cohort.CohortError, match='identity contradicts'):
        cohort.product_identity()
