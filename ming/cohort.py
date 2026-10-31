"""Fail closed on the packaged metadata cohort; never retarget old artifacts."""
import hashlib
from importlib.metadata import distribution
from importlib.resources import files
import json

from packaging.requirements import Requirement


def _require_pg_sources(installed, lock):
    name = 'pg' + lock['particlegan_version'].replace('.', '') + '_release.json'
    manifest = json.loads(installed.locate_file('particle_sliders/' + name).read_text())
    if (manifest['version'] != lock['particlegan_version'] or
            manifest['wheel_sha256'] != lock['particlegan_wheel_sha256']):
        raise RuntimeError('Ming ParticleGAN release manifest differs from its cohort')
    root = distribution('particlegan').locate_file('particlegan')
    actual = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in root.rglob('*.py')}
    if actual != manifest['python_sources']:
        raise RuntimeError('Ming ParticleGAN source closure differs from its cohort')


def _require_native_stack(lock):
    if (distribution('hypergan').version != lock['hypergan_version'] or
            distribution('hndl').version != lock['hndl_version']):
        raise RuntimeError('Ming requires the exact candidate HyperGAN/HNDL versions')
    origin = json.loads(distribution('hypergan').read_text('direct_url.json') or '{}')
    if (origin.get('url', '').removesuffix('.git') != 'https://github.com/HyperGAN/HyperGAN' or
            origin.get('vcs_info', {}).get('commit_id') != lock['hypergan_commit']):
        raise RuntimeError('Ming requires the exact candidate HyperGAN VCS origin')
    # Check declared ordinary/train dependencies too, not only source identity.
    for name, extras in (('hypergan', ('', 'train')), ('hndl', ('',)),
                         ('particle-sliders-core', ('', 'hypergan-hndl'))):
        for text in distribution(name).requires or ():
            requirement = Requirement(text)
            if requirement.marker and not any(requirement.marker.evaluate({'extra': e}) for e in extras):
                continue
            if requirement.specifier and not requirement.specifier.contains(distribution(requirement.name).version):
                raise RuntimeError('Ming candidate dependency constraint is unsatisfied: ' + text)
    from particle_sliders.fibo_hndl import verified_builder
    _, identity = verified_builder()  # Exact HNDL closure, clean HG provenance and builder hashes.
    if (identity['hypergan_commit'] != lock['hypergan_commit'] or
            identity['hndl_version'] != lock['hndl_version']):
        raise RuntimeError('Ming candidate builder contradicts its cohort')


def require_cohort(profile='public-0.11'):
    names = {'public-0.11': 'core.lock.json', 'native-candidate-0.12': 'native-candidate.lock.json'}
    if profile not in names:
        raise ValueError('unknown Ming cohort; no implicit upgrade')
    lock = json.loads(files('ming').joinpath(names[profile]).read_text())
    installed = distribution(lock['package'])
    if (installed.version != lock['version'] or
            distribution('particlegan').version != lock['particlegan_version']):
        raise RuntimeError('Ming requires its exact packaged core/ParticleGAN cohort')
    origin = json.loads(installed.read_text('direct_url.json') or '{}')
    if 'vcs_info' in origin:
        qualified = (origin['vcs_info'].get('commit_id') == lock['commit'] and
                     origin.get('url', '').removesuffix('.git') == lock['repository'] and
                     origin.get('subdirectory') == lock['subdirectory'])
    else:
        qualified = origin.get('archive_info', {}).get('hashes', {}).get('sha256') == lock['shared_wheel_sha256']
    if not qualified or origin.get('dir_info', {}).get('editable'):
        raise RuntimeError('Ming shared core origin differs from its packaged cohort')
    for name, expected in lock['source_sha256'].items():
        if hashlib.sha256(installed.locate_file(name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Ming shared core source differs from its packaged cohort: ' + name)
    _require_pg_sources(installed, lock)
    if profile == 'native-candidate-0.12':
        _require_native_stack(lock)
    return lock
