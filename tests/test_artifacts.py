"""Actual wheel names, namespaces, resources and unsafe co-install defense."""
from email import message_from_bytes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZipFile

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
import pytest

ROOT = Path(os.environ.get('MING_SOURCE_ROOT', Path(__file__).resolve().parents[1]))


@pytest.fixture(scope='module')
def wheels(tmp_path_factory):
    staging = tmp_path_factory.mktemp('ming-packaging-source')
    for name in ('ming', 'profiles', 'scripts'):
        shutil.copytree(ROOT / name, staging / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('pyproject.toml', 'requirements.txt', 'README.md', 'LICENSE', 'NOTICE', 'comfy_ming.py'):
        shutil.copy2(ROOT / name, staging / name)
    output = tmp_path_factory.mktemp('ming-artifacts')
    subprocess.run([sys.executable, '-I', '-m', 'build', '--wheel', '--no-isolation', '--outdir', str(output), str(staging)], check=True, capture_output=True)
    subprocess.run([sys.executable, '-I', str(staging / 'scripts/build_native_candidate.py'), '--output-dir', str(output)], check=True, capture_output=True)
    return output / 'ming_particle_sliders-0.2.0-py3-none-any.whl', output / 'ming_particle_sliders_native-0.2.0.dev3-py3-none-any.whl'


def metadata(archive):
    return message_from_bytes(archive.read(next(n for n in archive.namelist() if n.endswith('.dist-info/METADATA'))))


def test_built_wheels_have_distinct_identity_namespace_entrypoints_and_one_lock(wheels):
    with ZipFile(wheels[0]) as stable, ZipFile(wheels[1]) as native:
        a, b = set(stable.namelist()), set(native.namelist())
        assert not a & b
        assert metadata(stable)['Name'] == 'ming-particle-sliders'
        assert metadata(native)['Name'] == 'ming-particle-sliders-native'
        assert metadata(native)['Version'] not in SpecifierSet('==0.2.0')
        assert not any(n.startswith(('ming/', 'comfy_ming')) or n.endswith('entry_points.txt') for n in b)
        assert not any('native-candidate.lock' in n or n.startswith('ming_native/') for n in a)
        for archive, namespace, profile in ((stable, 'ming', 'public-pg095-3dd672a7'), (native, 'ming_native', 'native-pg095-3dd672a7')):
            assert [n for n in archive.namelist() if n.endswith('.lock.json')] == [namespace + '/core.lock.json']
            identity = json.loads(archive.read(namespace + '/identity.json'))
            assert identity['profile'] == profile
            assert identity['namespace'] == namespace
            assert set(metadata(archive).get_all('License-File')) == {'LICENSE', 'NOTICE'}
        lock = json.loads(native.read('ming_native/core.lock.json'))
        requirements = {r.name: r for r in map(Requirement, metadata(native).get_all('Requires-Dist')) if r.marker is None}
        assert str(requirements['particlegan'].specifier) == '==' + lock['particlegan_version']


def test_actual_candidate_wheel_cannot_satisfy_stable_requirement(wheels, tmp_path):
    # Resolver identity probe only: no dependencies/network involved.
    only_native = tmp_path / 'only-native'
    only_native.mkdir()
    shutil.copy2(wheels[1], only_native)
    result = subprocess.run([sys.executable, '-I', '-m', 'pip', '--isolated', 'install', '--dry-run', '--ignore-installed', '--no-deps', '--no-index', '--find-links', str(only_native), 'ming-particle-sliders==0.2.0'], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'No matching distribution found for ming-particle-sliders==0.2.0' in result.stderr


def test_actual_unsafe_coinstall_is_disjoint_and_both_identities_refuse(wheels, tmp_path):
    # Deliberately bypass dependency resolution in a disposable target to test
    # the defense against a user force-installing incompatible distributions.
    target = tmp_path / 'forced-target'
    subprocess.run([sys.executable, '-I', '-m', 'pip', '--isolated', 'install', '--no-deps', '--no-index', '--target', str(target), *map(str, wheels)], check=True, capture_output=True)
    probe = '''
import importlib,sys
sys.path.insert(0,sys.argv[1])
a=importlib.import_module('ming.cohort')
b=importlib.import_module('ming_native.cohort')
assert a is not b
for module in (a,b):
    try: module.product_identity()
    except module.CohortError as exc: assert 'separate environments; both are installed' in str(exc)
    else: raise AssertionError('unsafe co-install must refuse')
'''
    result = subprocess.run([sys.executable, '-I', '-c', probe, str(target)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
