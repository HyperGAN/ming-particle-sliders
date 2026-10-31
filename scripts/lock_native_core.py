"""Regenerate only the approved modern candidate closure from Git + built wheel.

Run for qualification, not routine installation. It does not download or publish artifacts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tomllib
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
COMMIT = 'a794e3b4a8813b012f2c4faa3da82b78d4a90214'
PREFIX = 'packages/particle-sliders-core/'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--core-repo', type=Path, required=True)
    p.add_argument('--core-wheel', type=Path, required=True)
    args = p.parse_args(argv)
    def git(*items):
        return subprocess.check_output(['git', '-C', str(args.core_repo), *items])
    paths = git('ls-tree', '-r', '--name-only', COMMIT, PREFIX + 'src').decode().splitlines()
    closure = {}
    for name in paths:
        relative = name.removeprefix(PREFIX + 'src/')
        if relative.split('/')[0] in ('particle_sliders', 'concept_slider_core'):
            closure[relative] = hashlib.sha256(git('show', COMMIT + ':' + name)).hexdigest()
    with ZipFile(args.core_wheel) as archive:
        packaged = {n: hashlib.sha256(archive.read(n)).hexdigest() for n in archive.namelist()
                    if n.startswith(('particle_sliders/', 'concept_slider_core/')) and not n.endswith('/')}
    if packaged != closure:
        raise SystemExit('built core wheel differs from the recursive Git source/data closure')
    pyproject = git('show', COMMIT + ':' + PREFIX + 'pyproject.toml')
    metadata = tomllib.loads(pyproject.decode())['project']
    previous_path = ROOT / 'profiles/history/native-dev12-core4abc.lock.json'
    previous_bytes = previous_path.read_bytes()
    lock = json.loads(previous_bytes)
    if 'particlegan==' + lock['particlegan_version'] not in metadata['dependencies']:
        raise SystemExit('approved core PG dependency differs; do not infer a new cohort')
    lock.update(commit=COMMIT, version=metadata['version'],
                shared_wheel_sha256=hashlib.sha256(args.core_wheel.read_bytes()).hexdigest(),
                source_sha256=dict(sorted(closure.items())),
                source_roots=['concept_slider_core', 'particle_sliders'],
                source_closure_schema='recursive-package-files-v1',
                core_git_tree=git('rev-parse', COMMIT + '^{tree}').decode().strip(),
                core_pyproject_sha256=hashlib.sha256(pyproject).hexdigest(),
                native_product_version='0.1.1.dev13', native_profile='native-a794e3b4',
                status='active approved modern software candidate; native gates unchanged',
                qualification='CPU product software fixtures only; no new native/pretrained/GPU admission',
                previous_preview=dict(product_version='0.1.1.dev12', core_commit='4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231',
                                      lock_sha256=hashlib.sha256(previous_bytes).hexdigest(),
                                      lock_path='profiles/history/native-dev12-core4abc.lock.json'))
    lock.pop('public_pr', None)
    path = ROOT / 'profiles/native-candidate.lock.json'
    path.write_text(json.dumps(lock, indent=2) + '\n')
    print(json.dumps(dict(commit=COMMIT, files=len(closure), lock_sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
