"""Fail closed on the packaged metadata cohort; never retarget old artifacts."""
import hashlib
from importlib.metadata import distribution
from importlib.resources import files
import json


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
    return lock
