"""Installed-product identity and immutable runtime cohort admission."""
import hashlib
from importlib.metadata import distribution, PackageNotFoundError
from importlib.resources import files
import json
from pathlib import Path


class CohortError(RuntimeError):
    """Install the selected product in its own exact dependency environment."""


try:
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
except ImportError as exc:
    raise CohortError('Ming requires packaging; install the selected product with dependencies in its own environment') from exc


def _dist(name):
    try:
        return distribution(name)
    except PackageNotFoundError as exc:
        raise CohortError(f'Ming is missing dependency {name!r}; install the selected product with dependencies in its own environment') from exc


def _json(text, label):
    try:
        value = json.loads(text or '{}')
    except (ValueError, TypeError) as exc:
        raise CohortError('Ming has malformed ' + label) from exc
    if not isinstance(value, dict):
        raise CohortError('Ming requires an object for ' + label)
    return value


def product_identity():
    """Bind the resource namespace, installed distribution and single lock."""
    try:
        identity = _json(files(__package__).joinpath('identity.json').read_text(), 'product identity')
        expected = {
            'ming': ('ming-particle-sliders', '0.2.0', 'public-pg094-ddb41423', 'ming-particle-sliders-native',
                     '1c2a65511e2349ef5a292253f02886afdd9a456db645e0eb5dc674bb95317f08'),
        }
        namespace = __package__.rsplit('.', 1)[-1]
        if namespace not in expected or tuple(identity.get(k) for k in ('distribution', 'version', 'profile', 'incompatible_distribution', 'lock_sha256')) != expected[namespace]:
            raise CohortError('Ming product identity contradicts its installed namespace')
        if identity.get('namespace') != namespace or identity.get('schema') != 'ming-product-cohort-v1':
            raise CohortError('Ming product identity schema/namespace differs')
        product = _dist(identity['distribution'])
        if product.version != identity['version']:
            raise CohortError('Ming installed product version contradicts its declared cohort')
        try:
            other = distribution(identity['incompatible_distribution'])
        except PackageNotFoundError:
            other = None
        if other is not None:
            raise CohortError('Ming stable and native distributions require separate environments; both are installed')
        # Also supports Comfy's source-package alias, only when its source and
        # resources exactly match the installed canonical package.
        for name in ('identity.json', 'core.lock.json', 'cohort.py'):
            local = files(__package__).joinpath(name).read_bytes()
            installed = product.locate_file(namespace + '/' + name).read_bytes()
            if local != installed:
                raise CohortError('Ming imported product differs from its installed distribution: ' + name)
        lock_bytes = files(__package__).joinpath('core.lock.json').read_bytes()
        if hashlib.sha256(lock_bytes).hexdigest() != identity.get('lock_sha256'):
            raise CohortError('Ming product lock contradicts its declared identity')
        return identity
    except (OSError, ModuleNotFoundError) as exc:
        raise CohortError('Ming product identity/resources are missing; reinstall the selected wheel in its own environment') from exc


def _vcs_identity(installed, repository, commit, subdirectory=None):
    origin = _json(installed.read_text('direct_url.json'), 'VCS origin')
    return (origin.get('url', '').removesuffix('.git') == repository.removesuffix('.git') and
            origin.get('vcs_info', {}).get('vcs') == 'git' and
            origin.get('vcs_info', {}).get('commit_id') == commit and
            origin.get('subdirectory') == subdirectory and not origin.get('dir_info', {}).get('editable'))


def _require_pg_sources(installed, lock):
    name = 'pg' + lock['particlegan_version'].replace('.', '') + '_release.json'
    manifest = _json(installed.locate_file('particle_sliders/' + name).read_text(), 'PG release manifest')
    if (manifest['version'] != lock['particlegan_version'] or
            manifest['wheel_sha256'] != lock['particlegan_wheel_sha256']):
        raise CohortError('Ming ParticleGAN release manifest differs from its cohort')
    root = _dist('particlegan').locate_file('particlegan')
    actual = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.py')}
    if actual != manifest['python_sources']:
        raise CohortError('Ming ParticleGAN source closure differs from its cohort')


def _validate_requirement(requirement, lock):
    installed = _dist(requirement.name)
    if requirement.url:
        known = {
            'particle-sliders-core': (lock['repository'], lock['commit'], lock['subdirectory']),
        }
        if 'hypergan_commit' in lock:
            known['hypergan'] = ('https://github.com/HyperGAN/HyperGAN', lock['hypergan_commit'], None)
        key = canonicalize_name(requirement.name)
        if key not in known:
            raise CohortError('Ming candidate has an unqualified direct dependency: ' + str(requirement))
        repository, commit, subdirectory = known[key]
        expected = 'git+' + repository.removesuffix('.git') + '.git@' + commit
        if subdirectory:
            expected += '#subdirectory=' + subdirectory
        matched = _vcs_identity(installed, repository, commit, subdirectory)
        if key == 'particle-sliders-core' and not matched:
            origin = _json(installed.read_text('direct_url.json'), 'core archive origin')
            matched = (origin.get('archive_info', {}).get('hashes', {}).get('sha256') == lock['shared_wheel_sha256'] and
                       not origin.get('dir_info', {}).get('editable'))
        if requirement.url != expected or not matched:
            raise CohortError('Ming candidate Git dependency differs from the exact cohort: ' + requirement.name)
    elif requirement.specifier and not requirement.specifier.contains(installed.version):
        raise CohortError('Ming candidate dependency constraint is unsatisfied: ' + str(requirement))


def _require_native_stack(lock):
    if (_dist('hypergan').version != lock['hypergan_version'] or _dist('hndl').version != lock['hndl_version']):
        raise CohortError('Ming requires the exact candidate HyperGAN/HNDL versions')
    if not _vcs_identity(_dist('hypergan'), 'https://github.com/HyperGAN/HyperGAN', lock['hypergan_commit']):
        raise CohortError('Ming requires the exact candidate HyperGAN VCS origin')
    for name, extras in (('hypergan', ('', 'train')), ('hndl', ('',)), ('particle-sliders-core', ('', 'hypergan-hndl'))):
        for text in _dist(name).requires or ():
            requirement = Requirement(text)
            if requirement.marker and not any(requirement.marker.evaluate({'extra': e}) for e in extras):
                continue
            _validate_requirement(requirement, lock)
    from particle_sliders.fibo_hndl import verified_builder
    _, identity = verified_builder()
    if (identity['hypergan_commit'] != lock['hypergan_commit'] or identity['hndl_version'] != lock['hndl_version']):
        raise CohortError('Ming candidate builder contradicts its cohort')


def require_cohort(profile=None):
    identity = product_identity()
    declared = identity['profile']
    if profile is not None and profile != declared:
        raise ValueError(f'Ming installed product declares {declared}; no implicit upgrade or cross-cohort selection')
    try:
        lock = _json(files(__package__).joinpath('core.lock.json').read_text(), 'cohort lock')
        requirements = [Requirement(t) for t in _dist(identity['distribution']).requires or ()]
        core = [r for r in requirements if canonicalize_name(r.name) == 'particle-sliders-core' and r.marker is None]
        expected_core = f'git+{lock["repository"]}.git@{lock["commit"]}#subdirectory={lock["subdirectory"]}'
        if len(core) != 1 or core[0].url != expected_core:
            raise CohortError('Ming installed product dependency contradicts its declared core cohort')
        if declared == 'native-pg094-ddb41423':
            runtime = {canonicalize_name(r.name): r for r in requirements if r.marker is None}
            if (str(runtime['particlegan'].specifier) != '==' + lock['particlegan_version'] or
                    str(runtime['hndl'].specifier) != '==' + lock['hndl_version']):
                raise CohortError('Ming native product metadata contradicts its declared PG/HNDL cohort')
        for requirement in requirements:
            if requirement.marker is None or requirement.marker.evaluate({'extra': ''}):
                _validate_requirement(requirement, lock)
        installed = _dist(lock['package'])
        if installed.version != lock['version'] or _dist('particlegan').version != lock['particlegan_version']:
            raise CohortError('Ming requires its exact packaged core/ParticleGAN cohort')
        origin = _json(installed.read_text('direct_url.json'), 'shared core origin')
        qualified = (_vcs_identity(installed, lock['repository'], lock['commit'], lock['subdirectory']) if 'vcs_info' in origin else
                     origin.get('archive_info', {}).get('hashes', {}).get('sha256') == lock['shared_wheel_sha256'])
        if not qualified or origin.get('dir_info', {}).get('editable'):
            raise CohortError('Ming shared core origin differs from its packaged cohort')
        actual = {}
        for package in lock['source_roots']:
            root = installed.locate_file(package)
            for path in root.rglob('*'):
                if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc', '.pyo'):
                    name = package + '/' + str(path.relative_to(root))
                    actual[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != lock['source_sha256']:
            raise CohortError('Ming shared core recursive source/data closure differs from its packaged cohort')
        _require_pg_sources(installed, lock)
        if declared == 'native-pg094-ddb41423':
            _require_native_stack(lock)
        return lock
    except CohortError:
        raise
    except (ImportError, OSError, KeyError, TypeError, ValueError) as exc:
        raise CohortError('Ming cohort source/dependency admission failed; reinstall the exact selected product environment: ' + str(exc)) from exc
