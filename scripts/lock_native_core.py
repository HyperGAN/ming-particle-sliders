"""Regenerate the current native source closure from exact Git and reference wheel.

Qualification only; never install, download, train or publish. Changing the lock
also requires updating the product cohort's bound identity and fresh review.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tomllib
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '3dd672a7f9b8c169daef90fd63e4bb42b9ddc423'
PREFIX = 'packages/particle-sliders-core/'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-repo', type=Path, required=True)
    parser.add_argument('--core-wheel', type=Path, required=True)
    args = parser.parse_args(argv)
    def git(*items):
        return subprocess.check_output(['git', '-C', str(args.core_repo), *items])
    closure = {}
    for name in git('ls-tree', '-r', '--name-only', COMMIT, PREFIX + 'src').decode().splitlines():
        relative = name.removeprefix(PREFIX + 'src/')
        if relative.split('/')[0] in ('particle_sliders', 'concept_slider_core'):
            closure[relative] = hashlib.sha256(git('show', COMMIT + ':' + name)).hexdigest()
    with ZipFile(args.core_wheel) as archive:
        packaged = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()
                    if name.startswith(('particle_sliders/', 'concept_slider_core/')) and not name.endswith('/')}
    if packaged != closure:
        raise SystemExit('built core wheel differs from the recursive Git source/data closure')
    pyproject = git('show', COMMIT + ':' + PREFIX + 'pyproject.toml')
    metadata = tomllib.loads(pyproject.decode())['project']
    path = ROOT / 'profiles/native-candidate.lock.json'
    lock = json.loads(path.read_text())
    if 'particlegan==' + lock['particlegan_version'] not in metadata['dependencies']:
        raise SystemExit('core PG dependency differs; do not infer a new cohort')
    lock.update(commit=COMMIT, version=metadata['version'],
                shared_wheel_sha256=hashlib.sha256(args.core_wheel.read_bytes()).hexdigest(),
                source_sha256=dict(sorted(closure.items())),
                source_roots=['concept_slider_core', 'particle_sliders'],
                source_closure_schema='recursive-package-files-v1',
                core_git_tree=git('rev-parse', COMMIT + '^{tree}').decode().strip(),
                core_pyproject_sha256=hashlib.sha256(pyproject).hexdigest())
    path.write_text(json.dumps(lock, indent=2) + '\n')
    print(json.dumps(dict(commit=COMMIT, files=len(closure), lock_sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
