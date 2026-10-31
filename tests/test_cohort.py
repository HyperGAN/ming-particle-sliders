"""Installed resource/source admission and historical default separation."""
from importlib.metadata import distribution
from importlib.resources import files
import json
from pathlib import Path

import pytest

from ming import cohort


def test_packaged_public_lock_is_exact_root_lock():
    assert files('ming').joinpath('core.lock.json').read_bytes() == (Path(__file__).resolve().parents[1] / 'core.lock.json').read_bytes()
    assert cohort.require_cohort()['commit'] == '3c25f785d14073e6f627c0fa9d2297a041c79913'


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


def test_profiles_are_explicit_and_candidate_is_unpublished():
    with pytest.raises(ValueError, match='no implicit upgrade'):
        cohort.require_cohort('latest')
    candidate = json.loads(files('ming').joinpath('native-candidate.lock.json').read_text())
    assert candidate['commit'] == '4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231'
    assert 'unpublished' in candidate['status']
