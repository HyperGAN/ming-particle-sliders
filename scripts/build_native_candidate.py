"""Build the isolated native distribution; never install, train or publish."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-wheel', type=Path, help='Optionally verify the archived qualification wheel')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    lock_path = ROOT / 'profiles/native-candidate.lock.json'
    lock = json.loads(lock_path.read_text())
    if args.core_wheel is not None and hashlib.sha256(args.core_wheel.read_bytes()).hexdigest() != lock['shared_wheel_sha256']:
        raise SystemExit('core wheel does not match the exact native candidate artifact')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='ming-native-build-') as directory:
        staging = Path(directory)
        package = staging / 'ming_native'
        package.mkdir()
        for name in ('native.py', 'cohort.py', 'constants.py'):
            shutil.copy2(ROOT / 'ming' / name, package / name)
        (package / '__init__.py').write_text('"""Isolated experimental Ming native API; no stable CLI/Comfy imports."""\n')
        shutil.copy2(lock_path, package / 'core.lock.json')
        identity = dict(schema='ming-product-cohort-v1', distribution='ming-particle-sliders-native',
                        version='0.1.1.dev12', namespace='ming_native', profile='native-candidate-0.12',
                        incompatible_distribution='ming-particle-sliders',
                        lock_sha256=hashlib.sha256(lock_path.read_bytes()).hexdigest())
        (package / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
        for name in ('LICENSE', 'NOTICE', 'README.md'):
            shutil.copy2(ROOT / name, staging / name)
        project = tomllib.loads((ROOT / 'pyproject.toml').read_text())
        (staging / 'pyproject.toml').write_text(f'''[build-system]
requires = ["setuptools>=84.2.1"]
build-backend = "setuptools.build_meta"
[project]
name = "ming-particle-sliders-native"
version = "0.1.1.dev12"
description = "Isolated experimental staged Ming native response API"
requires-python = ">=3.11,<3.15"
license = "MIT"
license-files = ["LICENSE", "NOTICE"]
readme = "README.md"
dynamic = ["dependencies"]
[project.optional-dependencies]
test = {json.dumps(project['project']['optional-dependencies']['test'])}
[tool.setuptools]
packages = ["ming_native"]
[tool.setuptools.dynamic]
dependencies = {{file = ["requirements.txt"]}}
[tool.setuptools.package-data]
ming_native = ["core.lock.json", "identity.json"]
''')
        (staging / 'requirements.txt').write_text(
            f'particle-sliders-core @ git+{lock["repository"]}.git@{lock["commit"]}#subdirectory={lock["subdirectory"]}\n'
            f'particlegan=={lock["particlegan_version"]}\n'
            f'hndl=={lock["hndl_version"]}\n'
            f'hypergan @ git+https://github.com/HyperGAN/HyperGAN.git@{lock["hypergan_commit"]}\n'
            'packaging>=23,<27\n')
        subprocess.run([sys.executable, '-m', 'build', '--wheel', '--no-isolation',
                        '--outdir', str(output), str(staging)], check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
